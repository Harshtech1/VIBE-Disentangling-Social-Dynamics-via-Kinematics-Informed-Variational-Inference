#!/usr/bin/env python3
"""VIBE-GECV-RECON-v1.0 three-video validation; never processes the full corpus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModel, Dinov2Model, HubertModel, Wav2Vec2FeatureExtractor

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataloading.vibe_dataset import VIBEDataset, vibe_collate_fn
from data_preprocessing.reconstruction_contract import (
    K_RECON, T_RECON, bin_pool, map_local_tracks, select_k8_duration_confidence_id,
    shared_interval_synchrony, shared_time_grid,
)
from data_preprocessing.backbone_compat import load_hubert_base
from losses.loss_function import VIBE_Loss
from model.vibe_model import VIBE_Transformer
from phase5c_gecv_3video_diagnostic import (
    CANONICAL_LABELS, PILOT, decode_and_sample, locate_and_describe, locate_video,
    run_ffprobe, smooth,
)
from ultralytics import YOLO

D = 768
GLOBAL_ID = "OpenGVLab/VideoMAEv2-Base"
DINO_ID = "facebook/dinov2-base"
HUBERT_ID = "facebook/hubert-base-ls960"


class Blocked(RuntimeError):
    pass


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def serialise(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def report_value(value: Any) -> Any:
    """Summarise arrays for JSON while returning full values to the pipeline."""
    if isinstance(value, np.ndarray):
        return {"array": {"shape": list(value.shape), "dtype": str(value.dtype)}}
    if isinstance(value, torch.Tensor):
        return {"tensor": {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): report_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"tuple": [report_value(item) for item in value]}
    if isinstance(value, list):
        return [report_value(item) for item in value]
    return value


def trajectory_summary(trajectories: dict[int, list[tuple[int, np.ndarray]]]) -> dict[str, Any]:
    return {
        "track_ids": sorted(int(track_id) for track_id in trajectories),
        "track_lengths": {str(track_id): len(points) for track_id, points in trajectories.items()},
    }


def stage_report_value(name: str, value: Any) -> Any:
    """Keep diagnostics useful without persisting every tracked bounding box."""
    if name == "track" and isinstance(value, tuple) and len(value) == 2:
        trajectories, confidence = value
        summary = trajectory_summary(trajectories)
        summary["mean_detection_confidence"] = {
            str(track_id): float(np.mean(scores)) for track_id, scores in confidence.items() if scores
        }
        return summary
    if name == "trajectory_smoothing" and isinstance(value, dict):
        return trajectory_summary(value)
    return report_value(value)


def clip_identifier(clip: dict[str, Any]) -> str:
    """Handle runtime and initial NOT_RUN_GPU report schemas."""
    for key in ("video_id", "source_id", "clip_id", "id"):
        if clip.get(key):
            return str(clip[key])
    return "unknown-clip"


def timed(clip: dict[str, Any], name: str, fn):
    started = time.perf_counter()
    try:
        value = fn()
        clip["stages"][name] = {"status": "PASS", "elapsed_seconds": round(time.perf_counter() - started, 3), "result": stage_report_value(name, value)}
        return value
    except Blocked as exc:
        clip["stages"][name] = {"status": "BLOCKED", "elapsed_seconds": round(time.perf_counter() - started, 3), "reason": str(exc)}
    except Exception as exc:
        clip["stages"][name] = {"status": "WARNING", "elapsed_seconds": round(time.perf_counter() - started, 3), "error": f"{type(exc).__name__}: {exc}"}
    return None


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffmpeg_pcm(video: Path, output: Path) -> tuple[np.ndarray, dict[str, Any]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise Blocked(f"ffmpeg PCM decode failed: {completed.stderr.strip()}")
    waveform, sample_rate = sf.read(str(output), dtype="float32", always_2d=False)
    if sample_rate != 16000 or waveform.ndim != 1 or len(waveform) < 1600:
        raise Blocked("Decoded PCM is not non-empty 16-kHz mono audio")
    return waveform, {"command": command, "sample_rate": sample_rate, "channels": 1, "samples": int(len(waveform)), "sha256": file_hash(output)}


def track_with_confidence(frames: np.ndarray, weights: Path, device: str):
    detector = YOLO(str(weights)).to(device)
    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    confidence: dict[int, list[float]] = {}
    results = detector.track(source=[frame for frame in frames], persist=True, tracker="bytetrack.yaml", classes=[0], conf=0.5, verbose=False, device=device, stream=True)
    for frame_index, result in enumerate(results):
        if result.boxes is None or result.boxes.id is None:
            continue
        for box, track_id, score in zip(result.boxes.xywh.cpu().numpy(), result.boxes.id.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()):
            x, y, w, h = box[0] - box[2] / 2, box[1] - box[3] / 2, box[2], box[3]
            trajectories.setdefault(int(track_id), []).append((frame_index, np.asarray([x, y, w, h])))
            confidence.setdefault(int(track_id), []).append(float(score))
    if not trajectories:
        raise Blocked("YOLOv8-L + ByteTrack produced no person tracks")
    return trajectories, confidence


def dino_tracks(frames: np.ndarray, trajectories: dict[int, list[tuple[int, np.ndarray]]], selected: list[int], sampled_timestamps: np.ndarray, device: str):
    processor = AutoImageProcessor.from_pretrained(DINO_ID)
    model = Dinov2Model.from_pretrained(DINO_ID).to(device).eval()
    crops, owners, source_times = [], [], []
    height, width = frames.shape[1:3]
    for pid in selected:
        for sampled_index, box in trajectories[pid]:
            x, y, w, h = map(int, box)
            x, y = max(0, min(x, width - 1)), max(0, min(y, height - 1))
            w, h = max(1, min(w, width - x)), max(1, min(h, height - y))
            crop = frames[sampled_index, y:y + h, x:x + w]
            if crop.size:
                crops.append(crop)
                owners.append(pid)
                source_times.append(float(sampled_timestamps[sampled_index]))
    if not crops:
        raise Blocked("No valid DINO crops")
    values = []
    with torch.no_grad():
        for start in range(0, len(crops), 32):
            inputs = processor(crops[start:start + 32], return_tensors="pt").to(device)
            values.append(model(**inputs).last_hidden_state[:, 0, :].cpu().numpy())
    values = np.concatenate(values, axis=0).astype(np.float32)
    per_track_features, per_track_times = {}, {}
    for pid in selected:
        indices = [i for i, owner in enumerate(owners) if owner == pid]
        per_track_features[pid] = values[indices]
        per_track_times[pid] = np.asarray([source_times[i] for i in indices], dtype=np.float64)
    return per_track_features, per_track_times, {"checkpoint": DINO_ID, "native_feature_count": int(len(values)), "native_shape": list(values.shape)}


def hubert_grid(waveform: np.ndarray, duration: float, target_times: np.ndarray, device: str):
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_ID)
    model, load_meta = load_hubert_base(HUBERT_ID, device)
    with torch.no_grad():
        inputs = extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        native = model(inputs.input_values.to(device)).last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)
    native_times = (np.arange(len(native), dtype=np.float64) + 0.5) * (duration / len(native))
    pooled, mask = bin_pool(native, native_times, target_times)
    return pooled, mask, {"checkpoint": HUBERT_ID, "native_shape": list(native.shape), "mapping": "midpoint-bin mean pooling on the shared source-video grid", **load_meta}


def global_scene(frames: np.ndarray, device: str):
    if len(frames) < 16:
        raise Blocked("VideoMAEv2 requires at least 16 frames for this v1.0 diagnostic")
    indices = np.linspace(0, len(frames) - 1, 16).astype(int)
    processor = AutoImageProcessor.from_pretrained(GLOBAL_ID)
    config = AutoConfig.from_pretrained(GLOBAL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(GLOBAL_ID, config=config, trust_remote_code=True).to(device).eval()
    inputs = processor([frames[i] for i in indices], return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].permute(0, 2, 1, 3, 4).to(device)
    with torch.no_grad():
        output = model(**inputs)
    if isinstance(output, torch.Tensor):
        tensor = output
    elif hasattr(output, "last_hidden_state"):
        tensor = output.last_hidden_state
    else:
        tensor = output[0]
    if tensor.shape[-1] != D:
        raise Blocked(f"VideoMAEv2 output last dimension {tensor.shape[-1]} is not 768")
    scene = tensor.reshape(tensor.shape[0], -1, D).mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    return scene, {
        "checkpoint": GLOBAL_ID,
        "input_indices_at_6hz": indices.tolist(),
        "raw_output_shape": list(tensor.shape),
        "pooling": "mean over non-feature output axes to a single 768-D scene calibration vector; no temporal semantics asserted",
        "status": "CUSTOM_RECONSTRUCTION_DEVIATION",
    }


def write_pilot_h5(path: Path, clip: dict[str, Any], local: np.ndarray, local_mask: np.ndarray, agent_mask: np.ndarray, audio: np.ndarray, audio_mask: np.ndarray, scene: np.ndarray, sync: np.ndarray, grid: dict[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("global_v_seq", data=scene[None, :])
        handle.create_dataset("env_feat", data=scene[None, :])
        handle.create_dataset("local_features", data=local)
        handle.create_dataset("local_time_mask", data=local_mask)
        handle.create_dataset("agent_mask", data=agent_mask)
        handle.create_dataset("audio_seq", data=audio)
        handle.create_dataset("audio_time_mask", data=audio_mask)
        handle.create_dataset("physics_sync", data=sync)
        handle.create_dataset("reference_timestamps", data=grid["timestamps"])
        handle.create_dataset("reference_source_frames", data=grid["source_frames"])
        handle.create_dataset("reference_valid", data=grid["valid"])
        handle.attrs["label"] = clip["canonical_label"]
        handle.attrs["protocol"] = "VIBE-GECV-RECON-v1.0"
        handle.attrs["text_ablation"] = True
        handle.attrs["global_representation"] = "VideoMAEv2 pooled scene calibration only"
        handle.attrs["k_selection"] = "CUSTOM: duration descending, mean confidence descending, numeric track ID ascending"


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 6 VIBE-GECV-RECON-v1.0 three-video validation", "",
        "Classification: explicit reconstruction baseline; not exact VIBE reproduction.", "",
        "## Frozen reconstruction contract", "",
        "- Local: [8,32,768] with local_time_mask [8,32] and agent_mask [8].",
        "- Audio: deterministic FFmpeg 16-kHz mono PCM, HuBERT pooled to [32,768] with audio_time_mask [32]; audio failure is invalid, never zero-filled.",
        "- Global: OpenGVLab/VideoMAEv2-Base mean-pooled to one [768] scene-calibration vector; no token-to-T reshape.",
        "- Text: OFF, no anchors, and L_sat=0.",
        "- Synchrony: centroid directional cosine on common consecutive sampled-frame intervals only.",
        "- K=8: custom duration, confidence, ID ordering; not an author policy.", "",
    ]
    for clip in report["clips"]:
        lines += [f"## {clip_identifier(clip)} — {clip.get('class_name', 'unknown class')}", "", "| Stage | Status | Seconds | Detail |", "| --- | --- | ---: | --- |"]
        for name, entry in clip["stages"].items():
            detail = entry.get("reason") or entry.get("error") or "completed"
            lines.append(f"| {name} | {entry['status']} | {entry['elapsed_seconds']} | {str(detail).replace('|', '\\|')} |")
        lines.append("")
    lines += ["## Guards", "", "- Text input is disabled and L_sat is disabled; no captions or RoBERTa anchors were created.", "- Global scene uses OpenGVLab/VideoMAEv2-Base pooled to one 768-D calibration vector; no 1568-to-32 reshape occurs.", "- Pilot HDF5 files are limited to three validation clips and are not production corpus outputs.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-root", type=Path, default=Path.home() / "VIBE_TRANSFER_FROM_FRIEND")
    parser.add_argument("--work-dir", type=Path, default=REPO / "data" / "phase6_reconstruction_pilot")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit("BLOCKED: CUDA unavailable; run from the researcher-owned Lightning terminal.")
    if importlib.util.find_spec("lap") is None:
        raise SystemExit("BLOCKED: lap is required for ByteTrack; install it manually before the L4 validation.")
    yolo = next((p for p in (REPO / "yolov8l.pt", args.transfer_root / "code_reference" / "yolov8l.pt") if p.exists()), None)
    if yolo is None:
        raise SystemExit("BLOCKED: yolov8l.pt unavailable")
    roots = [Path.home() / "VIBE_research_paper" / "datasets" / "GECV" / "GECV-GroupVid"]
    report = {"protocol": "VIBE-GECV-RECON-v1.0", "started_utc": now(), "clips": []}
    h5_paths = []
    for video_id, class_name, archive in PILOT:
        clip = {"video_id": video_id, "class_name": class_name, "canonical_label": CANONICAL_LABELS[class_name], "stages": {}}
        report["clips"].append(clip)
        located = timed(clip, "locate", lambda v=video_id, a=archive: locate_and_describe(v, a, roots, args.transfer_root, args.work_dir / "media"))
        if not located:
            continue
        video = Path(located["path"])
        timed(clip, "ffprobe", lambda: run_ffprobe(video))
        decoded = timed(clip, "decode_6hz", lambda: decode_and_sample(video))
        if not decoded:
            continue
        frames, sampling = decoded
        grid = timed(clip, "shared_t32_grid", lambda: {k: v for k, v in shared_time_grid(sampling["decoded_frame_count"], sampling["source_fps"]).items()})
        tracked = timed(clip, "track", lambda: track_with_confidence(frames, yolo, args.device))
        if not tracked or not grid:
            continue
        tracks = timed(clip, "trajectory_smoothing", lambda: smooth(tracked[0]))
        if not tracks:
            continue
        selection = timed(clip, "k8_custom_selection", lambda: select_k8_duration_confidence_id(tracks, tracked[1]))
        if not selection:
            continue
        selected = selection["selected_track_ids"]
        sampled_timestamps = np.arange(len(frames), dtype=np.float64) * sampling["step"] / sampling["source_fps"]
        dino = timed(clip, "dino_native_and_grid", lambda: dino_tracks(frames, tracks, selected, sampled_timestamps, args.device))
        pcm = timed(clip, "ffmpeg_pcm", lambda: ffmpeg_pcm(video, args.work_dir / "pcm" / f"{video_id}.wav"))
        scene = timed(clip, "videomaev2_scene", lambda: global_scene(frames, args.device))
        if not dino or not pcm or not scene:
            continue
        local, local_mask, agent_mask = map_local_tracks(dino[0], dino[1], selected, grid["timestamps"])
        audio, audio_mask, audio_meta = timed(clip, "hubert_native_and_grid", lambda: hubert_grid(pcm[0], float(grid["duration_seconds"]), grid["timestamps"], args.device)) or (None, None, None)
        sync = timed(clip, "shared_interval_synchrony", lambda: shared_interval_synchrony(tracks, selected, tuple(frames.shape[1:3])))
        if audio is None or sync is None:
            continue
        h5_path = args.work_dir / "h5" / f"{video_id}_recon_v1.h5"
        timed(clip, "write_pilot_h5", lambda: write_pilot_h5(h5_path, clip, local, local_mask, agent_mask, audio, audio_mask, scene[0], sync[0], grid) or {"path": h5_path})
        clip["contract"] = {"local_features": list(local.shape), "local_time_mask": list(local_mask.shape), "agent_mask": list(agent_mask.shape), "audio_features": list(audio.shape), "audio_time_mask": list(audio_mask.shape), "global_scene": list(scene[0].shape), "gamma": sync[2], "selected_track_ids": selected, "text": "disabled", "l_sat": "disabled"}
        h5_paths.append(h5_path)
    if len(h5_paths) == 3:
        batch = vibe_collate_fn([VIBEDataset(h5_paths)[i] for i in range(3)])
        model = VIBE_Transformer(input_dim=D, latent_dim=512, num_classes=3).to(args.device).train()
        criterion = VIBE_Loss(lambda_sat=0.5, lambda_ortho=0.1, enable_sat=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        moved = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        output = model.forward_with_gamma(moved["local_p"], moved["global_v"], moved["env_feat"], moved["audio_seq"], moved["gamma"], agent_mask=moved["agent_mask"], local_time_mask=moved["local_time_mask"], audio_time_mask=moved["audio_time_mask"], global_time_mask=moved["global_time_mask"], reconstruction_mode=True)
        total, hce, kl, sat, ortho = criterion(output, moved["labels"], moved)
        optimizer.zero_grad(); total.backward(); optimizer.step()
        report["model_validation"] = {"status": "PASS", "logits": list(output["logits"].shape), "losses": {"total": float(total.detach()), "hce": float(hce.detach()), "kl": float(kl.detach()), "sat": float(sat.detach()), "ortho": float(ortho.detach())}, "optimizer_step": "PASS", "text_ablation": True}
    else:
        report["model_validation"] = {"status": "BLOCKED", "reason": "All three clip contracts are required for forward/loss/backward validation"}
    report["finished_utc"] = now()
    manifest = REPO / "data" / "manifests" / "phase6_three_video_validation.json"
    document = REPO / "docs" / "phase6_three_video_validation.md"
    manifest.parent.mkdir(parents=True, exist_ok=True); document.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2, default=serialise) + "\n")
    document.write_text(markdown(report))
    print(json.dumps(report, indent=2, default=serialise))
    return 0 if report["model_validation"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
