#!/usr/bin/env python3
"""Read-only schema and H116 forensic audit for VIBE-GECV-RECON-v1.0.

This script never writes HDF5 files or touches extraction outputs.  Its only
write is the requested JSON report.  The optional H116 ByteTrack examination is
explicitly CPU-only and uses a temporary copy of that one archive member.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[1]
SAMPLES = ("videoH001", "videoN001", "videoS001", "videoH115", "videoH117")
EXPECTED = {
    "local_features": ((8, 32, 768), "float32"),
    "local_time_mask": ((8, 32), "bool"),
    "agent_mask": ((8,), "bool"),
    "physics_sync": ((8, 8), "float32"),
    "audio_features": ((32, 768), "float32"),
    "audio_time_mask": ((32,), "bool"),
    "global_scene": ((768,), "float32"),
}


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def finite_info(array: np.ndarray) -> dict[str, Any] | None:
    if not np.issubdtype(array.dtype, np.number):
        return None
    return {
        "nan": int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.inexact) else 0,
        "posinf": int(np.isposinf(array).sum()) if np.issubdtype(array.dtype, np.inexact) else 0,
        "neginf": int(np.isneginf(array).sum()) if np.issubdtype(array.dtype, np.inexact) else 0,
        "all_finite": bool(np.isfinite(array).all()),
    }


def inspect_h5(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"source_id": path.name.removesuffix("_recon_v1.h5"), "path": str(path)}
    if not path.is_file():
        result.update(status="MISSING")
        return result
    try:
        with h5py.File(path, "r") as handle:
            datasets: dict[str, Any] = {}
            for key in sorted(handle.keys()):
                item = handle[key]
                if not isinstance(item, h5py.Dataset):
                    datasets[key] = {"kind": "non_dataset"}
                    continue
                value = item[()]
                datasets[key] = {
                    "shape": list(item.shape),
                    "dtype": str(item.dtype),
                    "nonfinite": finite_info(np.asarray(value)),
                }
            shapes = {
                key: {
                    "present": key in handle,
                    "expected_shape": list(shape),
                    "expected_dtype": dtype,
                    "actual_shape": list(handle[key].shape) if key in handle else None,
                    "actual_dtype": str(handle[key].dtype) if key in handle else None,
                    "matches": bool(key in handle and tuple(handle[key].shape) == shape and str(handle[key].dtype) == dtype),
                }
                for key, (shape, dtype) in EXPECTED.items()
            }
            references = {key: datasets[key] for key in sorted(datasets) if key.startswith("reference_")}
            result.update(
                status="READABLE",
                dataset_keys=sorted(datasets),
                datasets=datasets,
                extraction_schema_shape_checks=shapes,
                attributes={key: json_default(value) if isinstance(value, (np.generic, np.ndarray, Path)) else value for key, value in handle.attrs.items()},
                reference_metadata={"keys": sorted(references), "datasets": references, "reference_valid_present": "reference_valid" in handle},
                compatibility_proposals={
                    "audio_features_to_audio_seq": {
                        "proposal_only": True,
                        "structural_match": bool("audio_features" in handle and tuple(handle["audio_features"].shape) == (32, 768) and str(handle["audio_features"].dtype) == "float32"),
                        "note": "Same fixed tensor shape/dtype; no dataset is renamed and semantic equivalence is not asserted here.",
                    },
                    "global_scene_to_global_v_seq": {
                        "proposal_only": True,
                        "structural_match_after_leading_axis": bool("global_scene" in handle and tuple(handle["global_scene"].shape) == (768,) and str(handle["global_scene"].dtype) == "float32"),
                        "note": "Would require an adapter to add a leading axis [768] -> [1,768]; it is a single scene calibration vector, not an asserted temporal sequence.",
                    },
                    "env_feat": {
                        "present": "env_feat" in handle,
                        "identified_alias": None,
                        "note": "No distinct env_feat dataset or source-confirmed alias is present. global_scene must not silently be treated as env_feat.",
                    },
                },
            )
    except (OSError, ValueError, KeyError) as exc:
        result.update(status="UNREADABLE", error=f"{type(exc).__name__}: {exc}")
    return result


def ffprobe(video: Path) -> dict[str, Any]:
    if shutil.which("ffprobe") is None:
        return {"available": False}
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames,sample_rate,channels", "-of", "json", str(video)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        return {"available": True, "error": completed.stderr.strip()}
    return {"available": True, "result": json.loads(completed.stdout)}


def extract_h116_to_temp(archive: Path, directory: Path) -> Path:
    destination = directory / "videoH116.mp4"
    with tarfile.open(archive, "r:xz") as handle:
        matches = [member for member in handle.getmembers() if Path(member.name).name == destination.name]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one videoH116.mp4 archive member; found {len(matches)}")
        source = handle.extractfile(matches[0])
        if source is None:
            raise RuntimeError("Cannot read videoH116 archive member")
        with destination.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
    return destination


def decode_samples(video: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open videoH116")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = 30.0 if not math.isfinite(fps) or fps <= 0 else fps
    step = max(1, round(fps / 6))
    frames: list[np.ndarray] = []
    decoded = 0
    first_shape = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if first_shape is None:
            first_shape = list(frame.shape)
        if decoded % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        decoded += 1
    capture.release()
    return frames, {"opened": True, "fps": fps, "step": step, "decoded_frames": decoded, "sampled_frames": len(frames), "first_bgr_shape": first_shape}


def cpu_bytetrack(frames: list[np.ndarray], weights: Path) -> dict[str, Any]:
    """CPU-only single-clip forensic check; does not save prediction files."""
    if not weights.is_file():
        return {"status": "BLOCKED", "reason": f"YOLO weights not found: {weights}"}
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights)).to("cpu")
        results = model.track(source=frames, persist=True, tracker="bytetrack.yaml", classes=[0], conf=0.5, device="cpu", stream=True, save=False, verbose=False)
        detections = 0
        person_frames = 0
        tracks: dict[int, int] = {}
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            detections += len(result.boxes)
            person_frames += 1
            if result.boxes.id is not None:
                for identifier in result.boxes.id.cpu().numpy().astype(int):
                    tracks[int(identifier)] = tracks.get(int(identifier), 0) + 1
        return {
            "status": "PASS",
            "device": "cpu",
            "sampled_frames_examined": len(frames),
            "person_detection_count": detections,
            "frames_with_person_detection": person_frames,
            "track_lengths": {str(key): value for key, value in sorted(tracks.items())},
            "conclusion": "no_person_detections" if detections == 0 else ("detections_without_track_ids" if not tracks else "person_tracks_observed"),
        }
    except Exception as exc:
        return {"status": "WARNING", "device": "cpu", "error": f"{type(exc).__name__}: {exc}"}


def h116_forensic(transfer_root: Path, weights: Path, existing_video: Path | None, run_tracking: bool) -> dict[str, Any]:
    archive = transfer_root / "raw" / "happy.tar.xz"
    output: dict[str, Any] = {"source_id": "videoH116", "archive": str(archive), "archive_exists": archive.is_file(), "weights": str(weights), "gpu_used": False}
    if not archive.is_file():
        output.update(status="BLOCKED", reason="Positive archive missing")
        return output
    if existing_video is not None:
        if not existing_video.is_file():
            output.update(status="BLOCKED", reason=f"Requested H116 path is missing: {existing_video}")
            return output
        output["raw_member"] = {"exists": True, "bytes": existing_video.stat().st_size, "ffprobe": ffprobe(existing_video), "location": str(existing_video)}
        try:
            frames, decode = decode_samples(existing_video)
            output["opencv_decode"] = decode
            output["cpu_yolov8l_bytetrack"] = cpu_bytetrack(frames, weights) if run_tracking else {"status": "SKIPPED"}
            output["status"] = "COMPLETE"
        except Exception as exc:
            output.update(status="WARNING", error=f"{type(exc).__name__}: {exc}")
        return output
    with tempfile.TemporaryDirectory(prefix="gecv_h116_readonly_") as temporary:
        try:
            video = extract_h116_to_temp(archive, Path(temporary))
            output["raw_member"] = {"exists": video.is_file(), "bytes": video.stat().st_size, "ffprobe": ffprobe(video)}
            frames, decode = decode_samples(video)
            output["opencv_decode"] = decode
            output["cpu_yolov8l_bytetrack"] = cpu_bytetrack(frames, weights) if run_tracking else {"status": "SKIPPED"}
            output["status"] = "COMPLETE"
        except Exception as exc:
            output.update(status="WARNING", error=f"{type(exc).__name__}: {exc}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "reports" / "gecv_schema_audit.json")
    parser.add_argument("--h5-root", type=Path, default=REPO / "data" / "gecv_reconstruction_v1")
    parser.add_argument("--transfer-root", type=Path, default=Path.home() / "VIBE_TRANSFER_FROM_FRIEND")
    parser.add_argument("--h116-video", type=Path, help="Existing read-only temporary H116 MP4; avoids another archive scan.")
    parser.add_argument("--skip-h116-tracking", action="store_true", help="Decode H116 but do not run the CPU-only ByteTrack forensic check.")
    args = parser.parse_args()
    candidates = (REPO / "yolov8l.pt", args.transfer_root / "code_reference" / "yolov8l.pt")
    weights = next((path for path in candidates if path.is_file()), candidates[0])
    samples = [inspect_h5(args.h5_root / f"{identifier}_recon_v1.h5") for identifier in SAMPLES]
    h116 = h116_forensic(args.transfer_root, weights, args.h116_video, not args.skip_h116_tracking)
    report = {
        "scope": "read-only completed-HDF5 schema audit and one-clip CPU-only H116 forensic check",
        "h5_root": str(args.h5_root),
        "samples": samples,
        "loader_expectations": {
            "file": "dataloading/vibe_dataset.py",
            "required_current_names": ["global_v_seq", "env_feat", "audio_seq", "physics_sync"],
            "reconstruction_names_currently_supported": ["local_features", "local_time_mask", "agent_mask", "audio_time_mask", "global_time_mask"],
            "reference_valid": {"used_by_training_loader": False, "note": "Not referenced by VIBEDataset; it is a Phase 6/strict-validator provenance field."},
            "finding": "The current loader cannot load completed production HDF5 files without a future, explicitly approved compatibility adapter because global_v_seq, env_feat, and audio_seq are absent.",
        },
        "h116_forensic": h116,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=json_default) + "\n")
    print("GECV schema audit (read-only)")
    for item in samples:
        print(f"- {item['source_id']}: {item['status']}; keys={len(item.get('dataset_keys', []))}")
    print(f"- H116 raw/decode/CPU-ByteTrack: {h116.get('status')}; {h116.get('cpu_yolov8l_bytetrack', {}).get('conclusion', h116.get('cpu_yolov8l_bytetrack', {}).get('status'))}")
    print(f"- Loader finding: {report['loader_expectations']['finding']}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
