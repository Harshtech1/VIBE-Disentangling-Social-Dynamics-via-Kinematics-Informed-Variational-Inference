#!/usr/bin/env python3
"""Hard-locked three-video GECV diagnostic; never writes canonical VIBE HDF5.

Run this only from the researcher-owned Lightning terminal with a verified GPU:

    python scripts/phase5c_gecv_3video_diagnostic.py

It processes only videoH001, videoN001, and videoS001. Results are diagnostic
JSON/Markdown, native feature summaries, and temporary extracted MP4s only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from transformers import AutoImageProcessor, Dinov2Model, HubertModel, Wav2Vec2FeatureExtractor
from ultralytics import YOLO


REPO = Path(__file__).resolve().parents[1]
CANONICAL_LABELS = {"Positive": 0, "Neutral": 1, "Negative": 2}
PILOT = (
    ("videoH001", "Positive", "happy.tar.xz"),
    ("videoN001", "Neutral", "neutral.tar.xz"),
    ("videoS001", "Negative", "negative.tar.xz"),
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def command_version(command: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    completed = subprocess.run([executable, "-version"], text=True, capture_output=True, check=False)
    return completed.stdout.splitlines()[0] if completed.returncode == 0 and completed.stdout else f"{executable} (version unavailable)"


def local_model_status(repo_id: str) -> str:
    """Read-only cache hint; model loading/downloading remains an execution-stage concern."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return "local-cache candidate" if (hub / f"models--{repo_id.replace('/', '--')}").exists() else "not found in default local cache; stage may require authorized model download"


def preflight(weights: Path | None, device: str) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    gpu = None
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        gpu = {"name": properties.name, "vram_gb": round(properties.total_memory / 1024**3, 2)}
    return {
        "cuda": {"requested_device": device, "available": cuda, "device_count": torch.cuda.device_count(), "gpu": gpu},
        "lap": {"available": importlib.util.find_spec("lap") is not None, "requirement": "lap is required by Ultralytics ByteTrack; this script never invokes Ultralytics automatic installation"},
        "ffmpeg": command_version("ffmpeg"),
        "ffprobe": command_version("ffprobe"),
        "models": {
            "yolov8l": str(weights) if weights is not None else "BLOCKED: yolov8l.pt not found",
            "dinov2": {"id": "facebook/dinov2-base", "availability": local_model_status("facebook/dinov2-base")},
            "hubert": {"id": "facebook/hubert-base-ls960", "availability": local_model_status("facebook/hubert-base-ls960")},
            "videomae": {"id": "MCG-NJU/videomae-base", "status": "BLOCKED scientifically; not called VideoMAE-v2"},
            "roberta": {"id": "roberta-base", "status": "BLOCKED for GECV without legitimate sample-specific text"},
        },
    }


def print_preflight(summary: dict[str, Any]) -> None:
    cuda = summary["cuda"]
    print("===== Phase 5C preflight =====")
    print(f"CUDA: available={cuda['available']} devices={cuda['device_count']} gpu={cuda['gpu']}")
    print(f"lap: {'PASS' if summary['lap']['available'] else 'BLOCKED (install lap before tracking; no automatic install will be attempted)'}")
    print(f"ffmpeg: {summary['ffmpeg'] or 'BLOCKED'}")
    print(f"ffprobe: {summary['ffprobe'] or 'BLOCKED'}")
    print(f"YOLOv8-L weights: {summary['models']['yolov8l']}")
    print(f"DINOv2: {summary['models']['dinov2']['availability']}")
    print(f"HuBERT: {summary['models']['hubert']['availability']}")


def stage(record: dict[str, Any], name: str, fn) -> Any:
    started = time.perf_counter()
    entry: dict[str, Any] = {"started_utc": now()}
    try:
        result = fn()
        entry.update({"status": "PASS", "result": result})
        return result
    except Blocked as exc:
        entry.update({"status": "BLOCKED", "reason": str(exc)})
        return None
    except Exception as exc:  # Diagnostic must continue to the next clip.
        entry.update({"status": "WARNING", "error": f"{type(exc).__name__}: {exc}"})
        return None
    finally:
        entry["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        record["stages"][name] = entry


class Blocked(RuntimeError):
    pass


def run_ffprobe(video: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,codec_long_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,sample_rate,channels",
        "-of", "json", str(video),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def extract_member(archive: Path, video_id: str, work_dir: Path) -> Path:
    """Extract exactly one MP4 member from a transferred archive."""
    if not archive.exists():
        raise Blocked(f"Archive not found: {archive}")
    work_dir.mkdir(parents=True, exist_ok=True)
    destination = work_dir / f"{video_id}.mp4"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    if destination.exists():
        destination.unlink()
    with tarfile.open(archive, "r:xz") as tf:
        members = [m for m in tf.getmembers() if Path(m.name).name == destination.name]
        if len(members) != 1:
            raise Blocked(f"Expected exactly one {destination.name} in {archive}; found {len(members)}")
        source = tf.extractfile(members[0])
        if source is None:
            raise Blocked(f"Cannot read archive member {members[0].name}")
        with destination.open("wb") as out:
            while chunk := source.read(1024 * 1024):
                out.write(chunk)
    return destination


def locate_video(video_id: str, archive_name: str, roots: list[Path], transfer_root: Path, work_dir: Path) -> Path:
    filename = f"{video_id}.mp4"
    cached = work_dir / filename
    if cached.is_file() and cached.stat().st_size > 0:
        return cached
    for root in roots:
        if root.exists():
            matches = list(root.rglob(filename))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise Blocked(f"Ambiguous source for {filename}: {[str(p) for p in matches]}")
    return extract_member(transfer_root / "raw" / archive_name, video_id, work_dir)


def locate_and_describe(video_id: str, archive_name: str, roots: list[Path], transfer_root: Path, work_dir: Path) -> dict[str, Any]:
    video = locate_video(video_id, archive_name, roots, transfer_root, work_dir)
    return {"path": video, "bytes": video.stat().st_size}


def decode_and_sample(video: Path, target_fps: int = 6) -> tuple[np.ndarray, dict[str, Any]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise Blocked(f"OpenCV could not open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fps = 30.0 if fps <= 0 or math.isnan(fps) else fps
    step = max(1, int(round(fps / target_fps)))
    decoded = 0
    sampled: list[np.ndarray] = []
    first_shape = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if first_shape is None:
            first_shape = list(frame.shape)
        if decoded % step == 0 and frame.shape[0] > 0 and frame.shape[1] > 0:
            sampled.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        decoded += 1
    cap.release()
    if not sampled:
        raise Blocked("No decodable sampled frames")
    return np.asarray(sampled), {
        "policy": "step=max(1, round(source_fps / 6)); retain frames whose zero-based index modulo step is zero",
        "source_fps": fps,
        "target_fps": target_fps,
        "step": step,
        "decoded_frame_count": decoded,
        "sampled_frame_count": len(sampled),
        "first_bgr_shape": first_shape,
    }


def track(frames: np.ndarray, weights: Path, device: str) -> dict[int, list[tuple[int, np.ndarray]]]:
    detector = YOLO(str(weights)).to(device)
    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    results = detector.track(
        source=[frame for frame in frames], persist=True, tracker="bytetrack.yaml",
        classes=[0], conf=0.5, verbose=False, device=device, stream=True,
    )
    for frame_index, result in enumerate(results):
        if result.boxes is None or result.boxes.id is None:
            continue
        boxes = result.boxes.xywh.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        for box, track_id in zip(boxes, ids):
            x, y, w, h = box[0] - box[2] / 2, box[1] - box[3] / 2, box[2], box[3]
            trajectories.setdefault(int(track_id), []).append((frame_index, np.asarray([x, y, w, h])))
    if not trajectories:
        raise Blocked("YOLOv8-L + ByteTrack produced no person tracks")
    return trajectories


def smooth(trajectories: dict[int, list[tuple[int, np.ndarray]]]) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Exact repository smoothing: smooth only tracks with at least five points."""
    result: dict[int, list[tuple[int, np.ndarray]]] = {}
    for pid, track_points in trajectories.items():
        if len(track_points) < 5:
            result[pid] = track_points
            continue
        frame_ids = [point[0] for point in track_points]
        boxes = np.asarray([point[1] for point in track_points])
        filtered = np.zeros_like(boxes)
        for dim in range(4):
            filtered[:, dim] = gaussian_filter1d(boxes[:, dim], sigma=5 / 3)
        result[pid] = [(int(frame), filtered[idx]) for idx, frame in enumerate(frame_ids)]
    return result


def select_k8_custom_reconstruction(trajectories: dict[int, list[tuple[int, np.ndarray]]]) -> dict[str, Any]:
    """Custom reconstruction policy, not an original VIBE selection rule.

    The paper/repository specifies a K=8 cap but no ranking criterion. This
    policy is deterministic for sensitivity analysis: descending observed track
    length, then ascending numeric ByteTrack ID.
    """
    observed = sorted(trajectories)
    lengths = {str(pid): len(trajectories[pid]) for pid in observed}
    ranked = sorted(observed, key=lambda pid: (-len(trajectories[pid]), pid))
    selected = ranked[:8]
    return {
        "status": "CUSTOM_RECONSTRUCTION_DEVIATION",
        "original_policy_status": "No explicit paper/repository K=8 retention ranking was found.",
        "ranking_rule": "descending observed track length; ties broken by ascending numeric track ID",
        "observed_track_ids": observed,
        "observed_track_lengths": lengths,
        "ranked_track_ids": ranked,
        "selected_track_ids": selected,
        "selected_count": len(selected),
        "representation": "K=8 explicit: selected tracks occupy slots 0..selected_count-1; remaining slots are zero-padded downstream.",
        "sensitivity_note": "Alternative retention rules (confidence, area, centrality, or tracker order) can change local features and synchrony, so results must be sensitivity-checked.",
    }


def dino_person_features(frames: np.ndarray, trajectories: dict[int, list[tuple[int, np.ndarray]]], selected: list[int], device: str) -> dict[str, Any]:
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
    crops: list[np.ndarray] = []
    lengths: dict[str, int] = {}
    height, width = frames.shape[1:3]
    for pid in selected:
        count = 0
        for frame_index, box in trajectories[pid]:
            x, y, w, h = map(int, box)
            x, y = max(0, min(x, width - 1)), max(0, min(y, height - 1))
            w, h = max(1, min(w, width - x)), max(1, min(h, height - y))
            crop = frames[frame_index, y:y + h, x:x + w]
            if crop.size:
                crops.append(crop)
                count += 1
        lengths[str(pid)] = count
    if not crops:
        raise Blocked("No valid person crops after trajectory clipping")
    features: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(crops), 32):
            inputs = processor(crops[start:start + 32], return_tensors="pt").to(device)
            features.append(model(**inputs).last_hidden_state[:, 0, :].cpu())
    out = torch.cat(features, dim=0)
    return {"checkpoint": "facebook/dinov2-base", "feature_shape": list(out.shape), "per_track_lengths": lengths, "dtype": str(out.numpy().dtype)}


def hubert(video: Path, device: str) -> dict[str, Any]:
    waveform, sample_rate = librosa.load(str(video), sr=16000, mono=True)
    if len(waveform) < 1600:
        raise Blocked("Audio is missing or shorter than 0.1 seconds after 16-kHz mono decode")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    with torch.no_grad():
        inputs = extractor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        out = model(inputs.input_values.to(device)).last_hidden_state.squeeze(0).cpu().numpy()
    return {"checkpoint": "facebook/hubert-base-ls960", "decoded_sample_rate": sample_rate, "mono_samples": int(len(waveform)), "native_feature_shape": list(out.shape), "dtype": str(out.dtype), "t32_mapping": "BLOCKED: native HuBERT sequence is reported without truncation, padding, or temporal remapping"}


def physics_sync(trajectories: dict[int, list[tuple[int, np.ndarray]]], selected: list[int], frame_shape: tuple[int, int]) -> dict[str, Any]:
    height, width = frame_shape
    velocity: list[np.ndarray] = []
    for pid in selected:
        coords = np.asarray([box for _, box in trajectories[pid]])
        cx = (coords[:, 0] + coords[:, 2] / 2) / width
        cy = (coords[:, 1] + coords[:, 3] / 2) / height
        v = np.stack([np.diff(cx), np.diff(cy)], axis=1) if len(cx) > 1 else np.zeros((0, 2))
        velocity.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6))
    matrix = np.zeros((8, 8), dtype=np.float32)
    for i in range(len(velocity)):
        for j in range(i + 1, len(velocity)):
            n = min(len(velocity[i]), len(velocity[j]))
            if n:
                matrix[i, j] = matrix[j, i] = np.mean(np.sum(velocity[i][:n] * velocity[j][:n], axis=1))
    k = len(selected)
    gamma = float(matrix[:k, :k][~np.eye(k, dtype=bool)].mean()) if k > 1 else 0.0
    return {"shape": [8, 8], "selected_track_ids": selected, "gamma": gamma, "assumptions": "Only the custom K=8-selected tracks are used. Selected tracks occupy leading matrix rows/columns; zero-padded rows/columns are excluded; diagonal is excluded; pairwise sequences are aligned by truncated observed-order length; no temporal mask is propagated."}


def text_status() -> dict[str, Any]:
    candidates = [
        REPO / "GECV_processed" / "metadata" / "train_index.json",
        Path.home() / "VIBE_TRANSFER_FROM_FRIEND" / "GECV_processed" / "metadata" / "train_index.json",
    ]
    found = [path for path in candidates if path.exists()]
    return {"expected_field": "Description (final extractor) / description (raw metadata)", "backbone": "roberta-base", "candidate_metadata": [str(path) for path in found], "status": "BLOCKED", "reason": "No verified non-empty, clip-aligned GECV description source is available; RoBERTa is not run and text is not fabricated."}


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Phase 5C GECV three-video diagnostic", "", f"Run UTC: {report['started_utc']}", "", "No final canonical VIBE HDF5 files were written.", ""]
    for clip in report["clips"]:
        lines += [f"## {clip['video_id']} — {clip['class_name']}", "", f"Canonical label: `{clip['canonical_label']}`", "", "| Stage | Status | Seconds | Detail |", "| --- | --- | ---: | --- |"]
        for name, item in clip["stages"].items():
            detail = item.get("reason") or item.get("error") or "completed"
            escaped_detail = str(detail).replace("|", "\\|")
            lines.append(f"| {name} | {item['status']} | {item['elapsed_seconds']} | {escaped_detail} |")
        selection = clip["stages"].get("k8_selection", {}).get("result")
        if selection:
            lines += ["", "### K=8 selection (custom reconstruction deviation)", "", f"- Observed IDs: `{selection['observed_track_ids']}`", f"- Observed lengths: `{selection['observed_track_lengths']}`", f"- Ranked IDs: `{selection['ranked_track_ids']}`", f"- Selected IDs: `{selection['selected_track_ids']}`", f"- Rule: {selection['ranking_rule']}", f"- Sensitivity: {selection['sensitivity_note']}"]
        lines.append("")
    lines += ["## Scientific guards", "", "- VideoMAE path is recorded as `MCG-NJU/videomae-base`; it is not called VideoMAE-v2 and no T=32 global mapping is made.", "- K=8 track retention uses a custom length-then-ID reconstruction policy, not an original VIBE claim; alternative rules require sensitivity analysis.", "- Native HuBERT time is reported without a T=32 conversion.", "- No GECV text is generated and no final HDF5 is written.", ""]
    return "\n".join(lines)


def write_reports(report: dict[str, Any]) -> None:
    manifest = REPO / "data" / "manifests" / "phase5c_3video_diagnostic.json"
    document = REPO / "docs" / "phase5c_3video_diagnostic.md"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    document.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2, default=json_value) + "\n")
    document.write_text(markdown(report))
    print(json.dumps(report, indent=2, default=json_value))
    print(f"\nWrote diagnostic JSON: {manifest}")
    print(f"Wrote diagnostic Markdown: {document}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-root", type=Path, default=Path.home() / "VIBE_TRANSFER_FROM_FRIEND")
    parser.add_argument("--raw-root", type=Path, action="append", default=[], help="Optional extracted-video root; may be repeated.")
    parser.add_argument("--work-dir", type=Path, default=REPO / "data" / "phase5c_pilot_media", help="Persistent, diagnostic-only location for exactly the three extracted MP4s.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-weights", type=Path, default=None)
    args = parser.parse_args()
    weights = args.yolo_weights or next((p for p in [REPO / "yolov8l.pt", args.transfer_root / "code_reference" / "yolov8l.pt"] if p.exists()), None)
    roots = args.raw_root + [Path.home() / "VIBE_research_paper" / "datasets" / "GECV" / "GECV-GroupVid"]
    summary = preflight(weights, args.device)
    print_preflight(summary)
    report: dict[str, Any] = {"started_utc": now(), "scope": [item[0] for item in PILOT], "device": args.device, "preflight": summary, "clips": []}
    if args.device.startswith("cuda") and not summary["cuda"]["available"]:
        report["overall_status"] = "BLOCKED: CUDA is unavailable. Run only in the researcher Lightning terminal after GPU verification."
        report["finished_utc"] = now()
        write_reports(report)
        return 2
    if not summary["lap"]["available"]:
        report["overall_status"] = "BLOCKED: Python package 'lap' is required for Ultralytics ByteTrack. Install it manually before rerunning; no automatic installation was attempted."
        report["finished_utc"] = now()
        write_reports(report)
        return 2
    if weights is None:
        report["overall_status"] = "BLOCKED: yolov8l.pt was not found in the repository or transfer package."
        report["finished_utc"] = now()
        write_reports(report)
        return 2
    if summary["ffmpeg"] is None or summary["ffprobe"] is None:
        report["overall_status"] = "BLOCKED: ffmpeg and ffprobe are both required for this diagnostic."
        report["finished_utc"] = now()
        write_reports(report)
        return 2
    for video_id, class_name, archive_name in PILOT:
        clip: dict[str, Any] = {"video_id": video_id, "class_name": class_name, "canonical_label": CANONICAL_LABELS[class_name], "stages": {}}
        report["clips"].append(clip)
        video = stage(clip, "locate", lambda v=video_id, a=archive_name: locate_and_describe(v, a, roots, args.transfer_root, args.work_dir))
        if not video:
            continue
        path = Path(video["path"])
        stage(clip, "ffprobe", lambda p=path: run_ffprobe(p))
        decoded = stage(clip, "decode_and_6hz_sample", lambda p=path: decode_and_sample(p))
        if decoded:
            frames, sampling = decoded
            clip["stages"]["decode_and_6hz_sample"]["result"] = sampling
            raw_tracks = stage(clip, "yolov8l_bytetrack", lambda: track(frames, weights, args.device))
            if raw_tracks:
                clip["stages"]["yolov8l_bytetrack"]["result"] = {"track_ids": sorted(raw_tracks), "track_lengths": {str(k): len(v) for k, v in raw_tracks.items()}, "policy": "YOLOv8-L, ByteTrack, person class 0, conf=0.5"}
                smoothed = stage(clip, "trajectory_smoothing", lambda: smooth(raw_tracks))
                if smoothed:
                    clip["stages"]["trajectory_smoothing"]["result"] = {
                        "policy": "Gaussian smoothing only for tracks of length >=5; sigma=5/3 on xywh coordinates",
                        "track_lengths": {str(k): len(v) for k, v in smoothed.items()},
                    }
                selection = stage(clip, "k8_selection", lambda: select_k8_custom_reconstruction(smoothed))
                if selection:
                    clip["stages"]["k8_selection"]["status"] = "WARNING"
                    clip["stages"]["k8_selection"]["reason"] = "CUSTOM RECONSTRUCTION DEVIATION: the paper/repository does not define this K=8 retention ranking."
                    selected = selection["selected_track_ids"]
                    stage(clip, "dinov2_person_features", lambda: dino_person_features(frames, smoothed, selected, args.device))
                    stage(clip, "synchrony", lambda: physics_sync(smoothed, selected, tuple(frames.shape[1:3])))
        stage(clip, "hubert_audio", lambda p=path: hubert(p, args.device))
        clip["stages"]["videomae_contract"] = {"status": "BLOCKED", "reason": "Current code names MCG-NJU/videomae-base, samples 16 frames, and yields a token sequence; it is not asserted as VideoMAE-v2 and no scientific T=32 mapping is applied.", "elapsed_seconds": 0.0}
        clip["stages"]["text_contract"] = {"status": "BLOCKED", "result": text_status(), "elapsed_seconds": 0.0}
        clip["stages"]["canonical_label"] = {"status": "PASS", "result": {"mapping": f"{class_name} -> {CANONICAL_LABELS[class_name]}"}, "elapsed_seconds": 0.0}
    report["finished_utc"] = now()
    report["overall_status"] = "COMPLETE: diagnostic stages are reported individually; global VideoMAE/T=32 and text remain scientific gates."
    write_reports(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
