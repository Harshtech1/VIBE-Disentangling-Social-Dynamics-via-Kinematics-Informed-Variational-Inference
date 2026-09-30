#!/usr/bin/env python3
"""Read-only integrity audit for VIBE-GECV-RECON-v1.0 data and frozen split."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


SCRIPT_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/gecv_reconstruction_v1"
SPLIT = ROOT / "data/manifests/gecv_reconstruction_v1_split_seed20260928.json"
EXTRACTION = ROOT / "data/manifests/gecv_reconstruction_extraction_manifest.json"
OUT = ROOT / "reports/dataset_audit"
RUNB = ROOT / "checkpoints/gecv_reconstruction_v1_runB/model_best.pth"
EXPECTED_RUNB_HASH = "3955147be0a5dea33ea0d2734c648508f50e2da72af9899bbb1e02efa4d8551a"
EXPECTED_SPLIT_HASH = "75691440c5982cc9efa0311c9b8ba68139f318ba82858e42e3cacbd4271ec823"
EXPECTED_CORPUS_HASH = "d84c8fed0f72de4b9a485e3502d41c107a2b9c0c89ff000f5b880177448854b9"
CLASS_NAMES = {0: "Positive", 1: "Neutral", 2: "Negative"}
EXPECTED_SHAPES = {
    "local_features": [8, 32, 768], "local_time_mask": [8, 32], "agent_mask": [8],
    "audio_features": [32, 768], "audio_time_mask": [32], "global_scene": [768],
    "physics_sync": [8, 8], "synchrony_pair_mask": [8, 8], "gamma": [],
    "reference_source_frames": [32], "reference_timestamps": [32], "selected_agent_ids": "variable length [1, 8] matching valid-agent count",
}
EXPECTED_DTYPES = {
    "local_features": "float32", "local_time_mask": "bool", "agent_mask": "bool",
    "audio_features": "float32", "audio_time_mask": "bool", "global_scene": "float32",
    "physics_sync": "float32", "synchrony_pair_mask": "bool", "gamma": "float32",
    "reference_source_frames": "int64", "reference_timestamps": "float64", "selected_agent_ids": "int32",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def listify(value) -> list:
    return list(value) if value is not None else []


def feature_digest(file: h5py.File) -> str:
    """Hash feature content and metadata needed to identify exact duplicate records."""
    digest = hashlib.sha256()
    for key in sorted(file.keys()):
        dataset = file[key]
        digest.update(key.encode())
        digest.update(str(dataset.shape).encode())
        digest.update(str(dataset.dtype).encode())
        value = dataset[()]
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(str(file.attrs.get("label", "")).encode())
    return digest.hexdigest()


def update_tensor_stats(stats: dict, key: str, value: np.ndarray, dataset: h5py.Dataset) -> None:
    row = stats.setdefault(key, {
        "expected_shape": EXPECTED_SHAPES.get(key), "expected_dtype": EXPECTED_DTYPES.get(key),
        "records": 0, "observed_shapes": Counter(), "observed_dtypes": Counter(),
        "nan_count": 0, "inf_count": 0, "empty_tensor_records": 0,
    })
    row["records"] += 1
    row["observed_shapes"][str(tuple(dataset.shape))] += 1
    row["observed_dtypes"][str(dataset.dtype)] += 1
    if value.size == 0:
        row["empty_tensor_records"] += 1
    if np.issubdtype(value.dtype, np.floating):
        row["nan_count"] += int(np.isnan(value).sum())
        row["inf_count"] += int(np.isinf(value).sum())


def check_file(path: Path, expected_label: int | None, tensor_stats: dict) -> tuple[list[str], dict, str]:
    errors, checks = [], {}
    with h5py.File(path, "r") as file:
        keys = set(file.keys())
        missing = sorted(set(EXPECTED_SHAPES) - keys)
        extra = sorted(keys - set(EXPECTED_SHAPES))
        if missing:
            errors.append(f"missing datasets: {missing}")
        if extra:
            errors.append(f"unexpected datasets: {extra}")
        values = {}
        for key in sorted(keys & set(EXPECTED_SHAPES)):
            dataset = file[key]
            value = np.asarray(dataset[()])
            values[key] = value
            update_tensor_stats(tensor_stats, key, value, dataset)
            if key == "selected_agent_ids":
                if len(dataset.shape) != 1 or not 1 <= dataset.shape[0] <= 8:
                    errors.append(f"{key} shape {list(dataset.shape)}")
            elif list(dataset.shape) != EXPECTED_SHAPES[key]:
                errors.append(f"{key} shape {list(dataset.shape)}")
            if str(dataset.dtype) != EXPECTED_DTYPES[key]:
                errors.append(f"{key} dtype {dataset.dtype}")
            if value.size == 0:
                errors.append(f"{key} empty")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                errors.append(f"{key} non-finite")
        label = int(file.attrs.get("label", -99))
        if label not in CLASS_NAMES:
            errors.append(f"invalid label {label}")
        if expected_label is not None and label != expected_label:
            errors.append(f"label conflicts with extraction manifest: {label} != {expected_label}")
        if file.attrs.get("protocol", "") != "VIBE-GECV-RECON-v1.0":
            errors.append("protocol attribute mismatch")
        if file.attrs.get("text_status", "") != "OFF":
            errors.append("text_status is not OFF")
        if file.attrs.get("l_sat_status", "") != "DISABLED_ZERO":
            errors.append("l_sat_status is not DISABLED_ZERO")

        if {"agent_mask", "local_time_mask", "local_features"} <= values.keys():
            agents, local_mask, local = values["agent_mask"].astype(bool), values["local_time_mask"].astype(bool), values["local_features"]
            checks["valid_agent_count"] = int(agents.sum())
            checks["valid_local_time_count"] = int(local_mask.sum())
            if np.any(local_mask[~agents]):
                errors.append("inactive agent has valid local time")
            if np.any(np.abs(local[~local_mask]) > 0):
                errors.append("masked local feature positions are nonzero")
            if np.any(agents & ~local_mask.any(axis=1)):
                errors.append("active agent has no valid local time")
            if "selected_agent_ids" in values and values["selected_agent_ids"].shape[0] != int(agents.sum()):
                errors.append("selected_agent_ids length does not match valid-agent count")
        if {"audio_time_mask", "audio_features"} <= values.keys():
            audio_mask, audio = values["audio_time_mask"].astype(bool), values["audio_features"]
            checks["valid_audio_time_count"] = int(audio_mask.sum())
            if not audio_mask.any():
                errors.append("no valid audio time")
            if np.any(np.abs(audio[~audio_mask]) > 0):
                errors.append("masked audio feature positions are nonzero")
        if {"agent_mask", "synchrony_pair_mask", "physics_sync"} <= values.keys():
            agents, pair = values["agent_mask"].astype(bool), values["synchrony_pair_mask"].astype(bool)
            checks["valid_synchrony_pair_count"] = int(pair.sum())
            if np.any(np.diag(pair)):
                errors.append("synchrony pair-mask diagonal is valid")
            if not np.array_equal(pair, pair.T):
                errors.append("synchrony pair-mask is not symmetric")
            active_pairs = agents[:, None] & agents[None, :]
            if np.any(pair & ~active_pairs):
                errors.append("synchrony pair-mask includes inactive agent")
        if "reference_timestamps" in values and np.any(np.diff(values["reference_timestamps"]) < 0):
            errors.append("reference timestamps are not monotonic")
        if "metadata_json" in file.attrs:
            try:
                metadata = json.loads(file.attrs["metadata_json"])
                metadata_label = metadata.get("label")
                checks["metadata_label"] = metadata_label
                if metadata_label is not None and int(metadata_label) != label:
                    errors.append("metadata label conflicts with HDF5 label")
            except (json.JSONDecodeError, TypeError):
                errors.append("malformed metadata_json")
        digest = feature_digest(file)
    return errors, checks, digest


def markdown(report: dict) -> str:
    checks = report["checks"]
    lines = ["# GECV Reconstruction Dataset + Split Integrity Audit", "",
             f"Timestamp: `{report['provenance']['timestamp_utc']}`", "", "## Result", "",
             f"Overall status: **{report['overall_status']}**", "", "## Compact Check Summary", "",
             "| Check | Status |", "|---|---|"]
    lines += [f"| {name.replace('_', ' ')} | {value['status']} |" for name, value in checks.items()]
    lines += ["", "## Sample Accounting", "",
              f"- Extraction-manifest raw records: {report['sample_accounting']['raw_extraction_records']}",
              f"- Physical HDF5 records: {report['sample_accounting']['physical_hdf5_records']}",
              f"- Usable split-assigned records: {report['sample_accounting']['usable_split_assigned_records']}",
              f"- Expected usable records: {report['sample_accounting']['expected_usable_records']}",
              f"- Split counts: train={report['sample_accounting']['split_counts']['train']}, validation={report['sample_accounting']['split_counts']['validation']}, test={report['sample_accounting']['split_counts']['test']}",
              "", "## Class Distribution", "", "| Split | Positive | Neutral | Negative | Total |", "|---|---:|---:|---:|---:|"]
    for split, counts in report["class_distribution"]["observed_counts"].items():
        lines.append(f"| {split} | {counts['Positive']} | {counts['Neutral']} | {counts['Negative']} | {sum(counts.values())} |")
    lines += ["", "## Leakage + Metadata Scope", "",
              "Sample/source-video intersections are calculated from frozen split IDs. Participant/session checks are **NOT_AVAILABLE** because the stored manifest/HDF5 metadata do not expose participant or session identifiers; none were inferred.",
              "", "## Feature Integrity", "",
              f"All {report['feature_integrity']['audited_records']} split-assigned HDF5 records were inspected read-only. Malformed records: {len(report['feature_integrity']['malformed_records'])}.",
              "", "## Provenance", "",
              f"- Split SHA-256: `{report['provenance']['split_manifest_sha256']}`",
              f"- Source corpus hash: `{report['provenance']['source_corpus_hash']}`",
              f"- Protected Run B checkpoint status: **{report['checks']['run_b_protected_checkpoint']['status']}**",
              "", "This audit does not train, repair, rewrite, or otherwise modify dataset, split, model, or checkpoint artifacts."]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text())
    extraction = json.loads(EXTRACTION.read_text())
    assignments = split["splits"]
    assigned_by_split = {name: set(ids) for name, ids in assignments.items()}
    assigned_all = set().union(*assigned_by_split.values())
    physical = {path.name.removesuffix("_recon_v1.h5"): path for path in DATA.glob("*_recon_v1.h5")}
    extraction_records = extraction["records"]
    output_record_ids = {sid for sid, entry in extraction_records.items() if entry.get("output") and sid in physical}
    expected_counts = {"train": {"Positive": 139, "Neutral": 74, "Negative": 69}, "validation": {"Positive": 30, "Neutral": 16, "Negative": 15}, "test": {"Positive": 30, "Neutral": 16, "Negative": 15}}

    intersections = {f"{a}_{b}": sorted(assigned_by_split[a] & assigned_by_split[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}
    assigned_counter = Counter(item for members in assignments.values() for item in members)
    tensor_stats, malformed, feature_hashes, observed_labels, metadata_label_present, label_failures = {}, [], defaultdict(list), defaultdict(Counter), False, []
    per_split_counts = {name: Counter() for name in assignments}
    valid_agents, valid_times = [], []
    for split_name, ids in assignments.items():
        for sample_id in ids:
            path = physical.get(sample_id)
            if path is None:
                malformed.append({"sample_id": sample_id, "errors": ["assigned file missing"]})
                continue
            expected_label = extraction_records.get(sample_id, {}).get("label")
            errors, checks, digest = check_file(path, expected_label, tensor_stats)
            feature_hashes[digest].append(sample_id)
            label = int(expected_label) if not errors or "label conflicts" not in " ".join(errors) else -1
            # Read the authoritative stored label independently of external manifest.
            with h5py.File(path, "r") as file:
                label = int(file.attrs["label"])
            observed_labels[split_name][label] += 1
            if checks.get("metadata_label") is not None:
                metadata_label_present = True
            label_failures.extend({"sample_id": sample_id, "error": error} for error in errors if "label" in error)
            valid_agents.append(checks.get("valid_agent_count", 0))
            valid_times.append(checks.get("valid_local_time_count", 0))
            if errors:
                malformed.append({"sample_id": sample_id, "errors": errors})
    duplicate_records = {digest: ids for digest, ids in feature_hashes.items() if len(ids) > 1}
    observed_counts = {split_name: {CLASS_NAMES[k]: int(counts[k]) for k in range(3)} for split_name, counts in observed_labels.items()}
    expected_present = {sid for sid in extraction_records if sid not in split.get("failed_ids", []) and sid not in split.get("smoke_only_ids", [])}
    unexpected_physical = sorted(set(physical) - output_record_ids)
    unassigned_physical = sorted(set(physical) - assigned_all)

    split_hash, runb_hash = sha256(SPLIT), sha256(RUNB)
    checks = {
        "sample_accounting": {"status": status(len(assigned_all) == 404 and sum(map(len, assignments.values())) == 404), "detail": "404 unique usable split assignments"},
        "split_counts": {"status": status({name: len(ids) for name, ids in assignments.items()} == {"train": 282, "validation": 61, "test": 61}), "detail": {name: len(ids) for name, ids in assignments.items()}},
        "no_missing_split_assignments": {"status": status(assigned_all == expected_present), "missing": sorted(expected_present - assigned_all), "unexpected": sorted(assigned_all - expected_present)},
        "no_unexpected_physical_hdf5_samples": {"status": status(not unexpected_physical and unassigned_physical == sorted(split.get("smoke_only_ids", []))), "unexpected": unexpected_physical, "permitted_smoke_only_unassigned": unassigned_physical},
        "no_duplicate_sample_ids": {"status": status(not [key for key, count in assigned_counter.items() if count > 1]), "duplicates": sorted(key for key, count in assigned_counter.items() if count > 1)},
        "class_distribution": {"status": status(observed_counts == expected_counts), "observed": observed_counts, "expected": expected_counts},
        "sample_id_split_intersections": {"status": status(not any(intersections.values())), "intersections": intersections},
        "source_video_split_intersections": {"status": status(not any(intersections.values())), "method": "sample ID equals source video ID in this one-record-per-video corpus", "intersections": intersections},
        "participant_session_overlap": {"status": "NOT_AVAILABLE", "reason": "No participant/session identifiers in frozen split manifest or HDF5 metadata."},
        "exact_duplicate_feature_records": {"status": status(not duplicate_records), "method": "SHA-256 over every stored dataset's key, shape, dtype, bytes, and label", "duplicates": duplicate_records},
        "feature_schema_and_finiteness": {"status": status(not malformed), "malformed_record_count": len(malformed)},
        "label_integrity": {"status": status(all(label in CLASS_NAMES for counts in observed_labels.values() for label in counts) and not label_failures), "mapping": {v: k for k, v in CLASS_NAMES.items()}, "metadata_label_presence": "PASS" if metadata_label_present else "NOT_AVAILABLE", "failures": label_failures},
        "run_b_protected_checkpoint": {"status": status(runb_hash == EXPECTED_RUNB_HASH), "sha256": runb_hash, "expected_sha256": EXPECTED_RUNB_HASH},
        "provenance_hashes": {"status": status(split_hash == EXPECTED_SPLIT_HASH and split.get("source_corpus_hash") == EXPECTED_CORPUS_HASH), "split_sha256": split_hash, "source_corpus_hash": split.get("source_corpus_hash")},
    }
    fail = any(check["status"] == "FAIL" for check in checks.values())
    report = {
        "audit": "VIBE-GECV-RECON-v1.0 dataset + frozen split integrity audit",
        "overall_status": "FAIL" if fail else "PASS",
        "checks": checks,
        "sample_accounting": {
            "raw_extraction_records": len(extraction_records), "physical_hdf5_records": len(physical), "physical_hdf5_with_extraction_output": len(output_record_ids),
            "usable_split_assigned_records": len(assigned_all), "expected_usable_records": 404,
            "split_counts": {name: len(ids) for name, ids in assignments.items()}, "smoke_only_excluded": split.get("smoke_only_ids", []), "failed_excluded": split.get("failed_ids", []),
            "unexpected_physical_hdf5_samples": unexpected_physical, "unassigned_physical_hdf5_samples": unassigned_physical,
        },
        "class_distribution": {"observed_counts": observed_counts, "expected_counts": expected_counts,
                               "percentages": {s: {c: count / sum(values.values()) * 100 for c, count in values.items()} for s, values in observed_counts.items()}},
        "leakage": {"sample_id_intersections": intersections, "source_video_intersections": intersections,
                    "participant_session": {"status": "NOT_AVAILABLE", "reason": "No participant/session identifiers available."}, "exact_duplicate_feature_records": duplicate_records},
        "feature_integrity": {
            "audited_records": len(assigned_all), "physical_storage": {"Kmax": 8, "T": 32}, "tensors": {key: {**value, "observed_shapes": dict(value["observed_shapes"]), "observed_dtypes": dict(value["observed_dtypes"])} for key, value in tensor_stats.items()},
            "valid_agent_count": {"min": int(min(valid_agents)), "max": int(max(valid_agents)), "mean": float(np.mean(valid_agents))},
            "valid_local_time_count": {"min": int(min(valid_times)), "max": int(max(valid_times)), "mean": float(np.mean(valid_times))},
            "malformed_records": malformed,
        },
        "label_integrity": {"canonical_mapping": {"Positive": 0, "Neutral": 1, "Negative": 2}, "observed_hdf5_labels": observed_counts,
                            "conflicting_metadata_labels": "NOT_AVAILABLE" if not metadata_label_present else []},
        "provenance": {
            "dataset": str(DATA.relative_to(ROOT)), "split_manifest": str(SPLIT.relative_to(ROOT)), "split_manifest_sha256": split_hash,
            "expected_split_manifest_sha256": EXPECTED_SPLIT_HASH, "source_corpus_hash": split.get("source_corpus_hash"), "expected_source_corpus_hash": EXPECTED_CORPUS_HASH,
            "dataset_schema_version": split.get("reconstruction_protocol"), "script": str(Path(__file__).relative_to(ROOT)), "script_version": SCRIPT_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": command("git", "rev-parse", "HEAD") or "UNAVAILABLE",
            "working_tree_state": "clean" if not command("git", "status", "--porcelain") else "dirty",
        },
    }
    json_path = OUT / "gecv_dataset_integrity_report.json"
    md_path = OUT / "gecv_dataset_integrity_report.md"
    csv_path = OUT / "gecv_dataset_integrity_summary.csv"
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    md_path.write_text(markdown(report))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        for name, check in checks.items():
            writer.writerow({"check": name, "status": check["status"], "detail": json.dumps({k: v for k, v in check.items() if k != "status"}, default=str)})
    print("DATASET INTEGRITY AUDIT COMPLETE")
    for name, check in checks.items():
        print(f"{check['status']:>13}  {name}")
    print(f"Protected Run B checkpoint: {checks['run_b_protected_checkpoint']['status']}")


if __name__ == "__main__":
    main()
