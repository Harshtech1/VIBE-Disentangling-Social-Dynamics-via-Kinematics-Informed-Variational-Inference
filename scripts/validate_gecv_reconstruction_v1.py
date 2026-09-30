#!/usr/bin/env python3
"""Read-only validator for completed VIBE-GECV-RECON-v1.0 primitives.

Safe during extraction: only manifest PASS/SKIP entries are opened, HDF5 files
are opened read-only, and transient/incomplete files are reported as warnings.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[1]
K, T, D = 8, 32, 768
LABELS = {"positive": 0, "neutral": 1, "negative": 2}
STRICT_SCHEMA = {
    "agent_mask": ((K,), np.bool_),
    "audio_seq": ((T, D), np.float32),
    "audio_time_mask": ((T,), np.bool_),
    "env_feat": ((1, D), np.float32),
    "global_v_seq": ((1, D), np.float32),
    "local_features": ((K, T, D), np.float32),
    "local_time_mask": ((K, T), np.bool_),
    "physics_sync": ((K, K), np.float32),
}
# The active production extractor currently emits these aliases.  They are
# checked separately so the report never silently equates them with STRICT_SCHEMA.
EXTRACTOR_ALIAS_SCHEMA = {
    "agent_mask": ((K,), np.bool_),
    "audio_features": ((T, D), np.float32),
    "audio_time_mask": ((T,), np.bool_),
    "global_scene": ((D,), np.float32),
    "local_features": ((K, T, D), np.float32),
    "local_time_mask": ((K, T), np.bool_),
    "physics_sync": ((K, K), np.float32),
}


def expected_ids() -> dict[str, str]:
    result = {f"videoH{i:03d}": "positive" for i in range(1, 202)}
    result.update({f"videoN{i:03d}": "neutral" for i in range(1, 109) if i != 10})
    result.update({f"videoS{i:03d}": "negative" for i in range(1, 101)})
    return result


def read_manifest(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("records", {}), dict):
            return {}, ["manifest has no records mapping"]
        return data, []
    except FileNotFoundError:
        return {}, ["manifest is not present yet"]
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"manifest is temporarily unreadable: {type(exc).__name__}: {exc}"]


def dtype_matches(actual: np.dtype, expected: type[np.generic]) -> bool:
    return np.dtype(actual) == np.dtype(expected)


def shape_dtype_errors(handle: h5py.File, schema: dict[str, tuple[tuple[int, ...], type[np.generic]]]) -> list[str]:
    errors: list[str] = []
    for name, (shape, dtype) in schema.items():
        if name not in handle:
            errors.append(f"missing dataset {name}")
            continue
        dataset = handle[name]
        if tuple(dataset.shape) != shape:
            errors.append(f"{name} shape {tuple(dataset.shape)} != {shape}")
        if not dtype_matches(dataset.dtype, dtype):
            errors.append(f"{name} dtype {dataset.dtype} != {np.dtype(dtype)}")
    return errors


def finite_errors(handle: h5py.File) -> list[str]:
    errors: list[str] = []
    for name, dataset in handle.items():
        if np.issubdtype(dataset.dtype, np.number) and not np.issubdtype(dataset.dtype, np.integer):
            values = dataset[()]
            if not np.isfinite(values).all():
                errors.append(f"{name} contains NaN or Inf")
    return errors


def validate_masks(handle: h5py.File, local_key: str, audio_key: str) -> list[str]:
    errors: list[str] = []
    agent = handle["agent_mask"][()]
    local_mask = handle["local_time_mask"][()]
    audio_mask = handle["audio_time_mask"][()]
    local = handle[local_key][()]
    audio = handle[audio_key][()]
    if agent.dtype != np.bool_ or local_mask.dtype != np.bool_ or audio_mask.dtype != np.bool_:
        errors.append("one or more masks are not boolean")
        return errors
    if local_mask.shape != (K, T) or agent.shape != (K,) or audio_mask.shape != (T,):
        errors.append("impossible agent/local/audio mask shape")
        return errors
    if local_mask[~agent].any():
        errors.append("padded local agent has true temporal mask entries")
    if not np.all(local[~local_mask] == 0):
        errors.append("masked local temporal positions are non-zero")
    if not np.all(local[~agent] == 0):
        errors.append("padded local agent features are non-zero")
    if not np.all(audio[~audio_mask] == 0):
        errors.append("masked audio temporal positions are non-zero")
    return errors


def validate_reference_metadata(handle: h5py.File, require_reference_valid: bool) -> list[str]:
    """Validate provenance coordinates for the schema variant being inspected.

    The earlier strict Phase-6 schema includes an explicit ``reference_valid``
    mask. The production reconstruction extractor records the source frames
    and timestamps but has no ``reference_valid`` field because its fixed grid
    is fully defined by the masks used for model inputs. Treating that absent
    field as corruption made this *validator* reject otherwise valid production
    files; it does not indicate an HDF5/data change.
    """
    errors: list[str] = []
    expected = {
        "reference_source_frames": ((T,), np.integer),
        "reference_timestamps": ((T,), np.floating),
    }
    if require_reference_valid:
        expected["reference_valid"] = ((T,), np.bool_)
    for name, (shape, kind) in expected.items():
        if name not in handle:
            errors.append(f"missing reference metadata {name}")
            continue
        dataset = handle[name]
        if tuple(dataset.shape) != shape or not np.issubdtype(dataset.dtype, kind):
            errors.append(f"invalid reference metadata {name}: shape={tuple(dataset.shape)}, dtype={dataset.dtype}")
    return errors


def validate_h5(path: Path, expected_class: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        # ``strict_errors`` means errors against the accepted schema variant,
        # not merely differences from the older Phase-6 representation.
        "strict_errors": [],
        "strict_schema_deviation": [],
        "warnings": [],
        "physics_sync": None,
    }
    try:
        with h5py.File(path, "r") as handle:
            strict_errors = shape_dtype_errors(handle, STRICT_SCHEMA)
            alias_errors = shape_dtype_errors(handle, EXTRACTOR_ALIAS_SCHEMA)
            strict = not strict_errors
            alias = not alias_errors
            result["schema_variant"] = "strict_phase6" if strict else ("production_extractor_alias" if alias else "invalid")
            if alias and not strict:
                # Preserve the legacy difference for forensic review, but do
                # not call a production record invalid merely because it does
                # not use fields from the older generic interface.
                result["strict_schema_deviation"].extend(strict_errors)
                result["warnings"].append("strict schema deviation: production aliases audio_features/global_scene replace audio_seq/env_feat/global_v_seq")
            active_schema = STRICT_SCHEMA if strict else EXTRACTOR_ALIAS_SCHEMA if alias else None
            if active_schema is None:
                result["strict_errors"].append("record matches neither the strict Phase-6 nor production extractor schema")
                result["strict_errors"].extend(f"strict: {error}" for error in strict_errors)
                result["strict_errors"].extend(f"production alias: {error}" for error in alias_errors)
                return result
            local_key = "local_features"
            audio_key = "audio_seq" if strict else "audio_features"
            result["strict_errors"].extend(finite_errors(handle))
            result["strict_errors"].extend(validate_masks(handle, local_key, audio_key))
            result["strict_errors"].extend(validate_reference_metadata(handle, require_reference_valid=strict))
            sync = handle["physics_sync"][()]
            result["physics_sync"] = {"min": float(sync.min()), "max": float(sync.max()), "mean": float(sync.mean()), "diagonal": np.diag(sync).tolist()}
            if not np.allclose(np.diag(sync), 0.0):
                result["warnings"].append("physics_sync diagonal is non-zero; extraction contract must define intended diagonal semantics")
            label = handle.attrs.get("label")
            expected_label = LABELS[expected_class]
            if label is None or int(label) != expected_label:
                result["strict_errors"].append(f"label {label!r} != canonical {expected_label}")
            metadata_text = handle.attrs.get("metadata_json")
            if metadata_text is not None:
                try:
                    metadata = json.loads(metadata_text)
                    result["metadata_present"] = True
                    if metadata.get("selection", {}).get("selected_count", 0) > K:
                        result["strict_errors"].append("metadata selected_count exceeds K=8")
                except (TypeError, json.JSONDecodeError):
                    result["warnings"].append("metadata_json is not valid JSON")
            else:
                result["warnings"].append("metadata_json attribute absent")
    except (OSError, KeyError, ValueError) as exc:
        result["strict_errors"].append(f"unreadable or incomplete HDF5: {type(exc).__name__}: {exc}")
    return result


def parse_output_id(path: Path) -> str | None:
    suffix = "_recon_v1.h5"
    return path.name[:-len(suffix)] if path.name.endswith(suffix) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "data/manifests/gecv_reconstruction_extraction_manifest.json")
    parser.add_argument("--output", type=Path, default=REPO / "data/gecv_reconstruction_v1")
    parser.add_argument("--report", type=Path, default=REPO / "reports/gecv_reconstruction_v1_validation.json")
    parser.add_argument("--max-files", type=int, default=None, help="Read at most N completed files; 0 validates only manifest/output inventory.")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    expected = expected_ids()
    manifest, manifest_warnings = read_manifest(args.manifest)
    records = manifest.get("records", {})
    source_id_values = [str(record.get("source_id", key)) for key, record in records.items() if isinstance(record, dict)]
    duplicate_ids = sorted(identifier for identifier, count in Counter(source_id_values).items() if count > 1)
    manifest_errors: list[str] = []
    for key, record in records.items():
        if not isinstance(record, dict):
            manifest_errors.append(f"{key}: record is not an object")
            continue
        if record.get("source_id", key) != key:
            manifest_errors.append(f"{key}: source_id mismatch")
        if record.get("status") not in {"PENDING", "RUNNING", "PASS", "FAIL", "SKIP"}:
            manifest_errors.append(f"{key}: invalid or missing status")
        if record.get("status") == "FAIL" and not record.get("failure"):
            manifest_errors.append(f"{key}: FAIL has no failure reason")
        if record.get("class") in LABELS and record.get("label") != LABELS[record["class"]]:
            manifest_errors.append(f"{key}: class/label conflict")

    all_outputs = sorted(args.output.glob("*.h5")) if args.output.exists() else []
    outputs_by_id = {identifier: path for path in all_outputs if (identifier := parse_output_id(path)) is not None}
    unexpected_outputs = sorted(path.name for path in all_outputs if parse_output_id(path) not in expected)
    completed = [(source_id, record) for source_id, record in records.items() if isinstance(record, dict) and record.get("status") in {"PASS", "SKIP"}]
    if args.max_files is not None:
        completed = completed[:max(0, args.max_files)]
    file_results: dict[str, Any] = {}
    for source_id, record in completed:
        path = outputs_by_id.get(source_id)
        if path is None:
            file_results[source_id] = {"path": record.get("output"), "strict_errors": ["completed manifest entry has no final HDF5"], "warnings": []}
            continue
        file_results[source_id] = validate_h5(path, expected.get(source_id, record.get("class", "positive")))

    failed = sorted(source_id for source_id, record in records.items() if isinstance(record, dict) and record.get("status") == "FAIL")
    completed_ids = {source_id for source_id, record in records.items() if isinstance(record, dict) and record.get("status") in {"PASS", "SKIP"} and source_id in outputs_by_id}
    missing = sorted(set(expected) - completed_ids - set(failed))
    strict_valid = [source_id for source_id, item in file_results.items() if not item["strict_errors"]]
    invalid = {source_id: item["strict_errors"] for source_id, item in file_results.items() if item["strict_errors"]}
    warning_count = len(manifest_warnings) + len(manifest_errors) + sum(len(item["warnings"]) for item in file_results.values())
    class_counts = {class_name: {"expected": sum(value == class_name for value in expected.values()), "completed_outputs": sum(expected.get(source_id) == class_name for source_id in completed_ids), "failed": sum(expected.get(source_id) == class_name for source_id in failed)} for class_name in LABELS}
    report = {
        "protocol": "VIBE-GECV-RECON-v1.0",
        "read_only": True,
        "manifest": str(args.manifest), "output": str(args.output),
        "expected_source_videos": len(expected), "successful_hdf5_outputs": len(completed_ids),
        "failed": len(failed), "missing": len(missing), "unexpected_outputs": unexpected_outputs,
        "h116": {"expected": True, "manifest_status": records.get("videoH116", {}).get("status", "MISSING"), "output_present": "videoH116" in outputs_by_id},
        "class_counts": class_counts,
        "manifest_warnings": manifest_warnings, "manifest_errors": manifest_errors, "duplicate_source_ids": duplicate_ids,
        "summary": {"valid_hdf5_files": len(strict_valid), "invalid_hdf5_files": len(invalid), "files_checked": len(file_results), "missing_files": len(missing), "failed_files": len(failed), "total_validation_errors": len(manifest_errors) + sum(len(reasons) for reasons in invalid.values()), "total_warnings": warning_count},
        "invalid_files": invalid, "file_results": file_results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.json_only:
        print(json.dumps(report, separators=(",", ":")))
    else:
        print(json.dumps(report["summary"], indent=2))
        print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
