"""Deterministic split management for VIBE-GECV reconstruction records.

This module reads completed HDF5 metadata only.  It never copies, renames, or
modifies a feature file.  The resulting split is a reconstruction split, not an
official GECV paper split.

The frozen seed 20260928 assigns 404 records to train=282, validation=61, and
test=61. IDs identify source records, not participants or sessions; those
identities are unavailable and must not be inferred from filenames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PROTOCOL = "VIBE-GECV-RECON-v1.0"
SPLIT_PROTOCOL = "VIBE-GECV-RECON-v1.0-split"
SEED = 20260928
SMOKE_ONLY_IDS = ("videoH001", "videoN001", "videoS001")
FAILED_IDS = ("videoH116",)
LABELS = {0: "Positive", 1: "Neutral", 2: "Negative"}
EXPECTED_SUCCESSFUL_COUNTS = {0: 200, 1: 107, 2: 100}
EXPECTED_ELIGIBLE_COUNTS = {0: 199, 1: 106, 2: 99}
EXPECTED_SPLIT_COUNTS = {
    "train": {0: 139, 1: 74, 2: 69},
    "validation": {0: 30, 1: 16, 2: 15},
    "test": {0: 30, 1: 16, 2: 15},
}


def source_id_from_path(path: Path) -> str:
    suffix = "_recon_v1.h5"
    if not path.name.endswith(suffix):
        raise ValueError(f"Not a reconstruction HDF5 filename: {path.name}")
    return path.name.removesuffix(suffix)


def expected_label_from_id(source_id: str) -> int:
    if source_id.startswith("videoH"):
        return 0
    if source_id.startswith("videoN"):
        return 1
    if source_id.startswith("videoS"):
        return 2
    raise ValueError(f"Unknown reconstruction source ID: {source_id}")


def inspect_successful_files(data_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(data_root.glob("*_recon_v1.h5")):
        source_id = source_id_from_path(path)
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("protocol") != PROTOCOL:
                raise ValueError(f"Unexpected protocol in {path}: {handle.attrs.get('protocol')!r}")
            if handle.attrs.get("validity") != "PASS":
                raise ValueError(f"Non-PASS reconstruction file included: {path}")
            label = int(handle.attrs["label"])
        if label not in LABELS:
            raise ValueError(f"Invalid canonical label {label} in {path}")
        if label != expected_label_from_id(source_id):
            raise ValueError(f"ID/label conflict for {path}: ID implies {expected_label_from_id(source_id)}, attr is {label}")
        records.append({"source_id": source_id, "label": label, "path": str(path)})
    ids = [record["source_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate reconstruction source IDs")
    counts = {label: sum(record["label"] == label for record in records) for label in LABELS}
    if counts != EXPECTED_SUCCESSFUL_COUNTS:
        raise ValueError(f"Successful class counts differ: {counts} != {EXPECTED_SUCCESSFUL_COUNTS}")
    if set(FAILED_IDS) & set(ids):
        raise ValueError("Failed source H116 must not have a successful reconstruction HDF5")
    return records


def corpus_hash(records: list[dict[str, Any]]) -> str:
    canonical = "\n".join(f"{record['source_id']}\t{record['label']}" for record in sorted(records, key=lambda item: item["source_id"]))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_split_manifest(data_root: Path, seed: int = SEED) -> dict[str, Any]:
    records = inspect_successful_files(data_root)
    by_id = {record["source_id"]: record for record in records}
    missing_smoke = set(SMOKE_ONLY_IDS) - set(by_id)
    if missing_smoke:
        raise ValueError(f"Smoke-only IDs missing from successful corpus: {sorted(missing_smoke)}")
    eligible = [record for record in records if record["source_id"] not in SMOKE_ONLY_IDS]
    eligible_counts = {label: sum(record["label"] == label for record in eligible) for label in LABELS}
    if eligible_counts != EXPECTED_ELIGIBLE_COUNTS:
        raise ValueError(f"Eligible class counts differ: {eligible_counts} != {EXPECTED_ELIGIBLE_COUNTS}")

    split_ids: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    seed_sequences = np.random.SeedSequence(seed).spawn(len(LABELS))
    for label, child_seed in zip(sorted(LABELS), seed_sequences):
        class_ids = sorted(record["source_id"] for record in eligible if record["label"] == label)
        ordered = [class_ids[index] for index in np.random.default_rng(child_seed).permutation(len(class_ids))]
        train_count = EXPECTED_SPLIT_COUNTS["train"][label]
        validation_count = EXPECTED_SPLIT_COUNTS["validation"][label]
        test_count = EXPECTED_SPLIT_COUNTS["test"][label]
        if train_count + validation_count + test_count != len(ordered):
            raise ValueError(f"Configured split counts do not exhaust label {label}")
        split_ids["train"].extend(ordered[:train_count])
        split_ids["validation"].extend(ordered[train_count:train_count + validation_count])
        split_ids["test"].extend(ordered[train_count + validation_count:])

    for split in split_ids:
        split_ids[split] = sorted(split_ids[split])
    validate_split_manifest({"splits": split_ids}, records)
    counts = {
        split: {LABELS[label]: sum(by_id[source_id]["label"] == label for source_id in source_ids) for label in LABELS}
        for split, source_ids in split_ids.items()
    }
    return {
        "protocol": SPLIT_PROTOCOL,
        "reconstruction_protocol": PROTOCOL,
        "seed": seed,
        "source_corpus": str(data_root),
        "source_corpus_hash": corpus_hash(records),
        "official_paper_split": False,
        "note": "This is a deterministic VIBE-GECV-RECON-v1.0 reconstruction split, not an official paper GECV split.",
        "smoke_only_ids": list(SMOKE_ONLY_IDS),
        "failed_ids": list(FAILED_IDS),
        "successful_source_ids": sorted(by_id),
        "splits": split_ids,
        "class_counts": counts,
        "total_counts": {split: len(source_ids) for split, source_ids in split_ids.items()},
    }


def validate_split_manifest(manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
    splits = manifest["splits"]
    required = {"train", "validation", "test"}
    if set(splits) != required:
        raise ValueError(f"Split names must be exactly {sorted(required)}")
    all_ids = [source_id for split in required for source_id in splits[split]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Split overlap or duplicate source ID detected")
    source_labels = {record["source_id"]: record["label"] for record in records}
    eligible_ids = set(source_labels) - set(SMOKE_ONLY_IDS)
    if set(all_ids) != eligible_ids:
        missing = sorted(eligible_ids - set(all_ids))
        unexpected = sorted(set(all_ids) - eligible_ids)
        raise ValueError(f"Split membership mismatch; missing={missing}, unexpected={unexpected}")
    for split, expected_counts in EXPECTED_SPLIT_COUNTS.items():
        actual_counts = {label: sum(source_labels[source_id] == label for source_id in splits[split]) for label in LABELS}
        if actual_counts != expected_counts:
            raise ValueError(f"{split} class counts differ: {actual_counts} != {expected_counts}")


def write_split_manifest(data_root: Path, manifest_path: Path, seed: int = SEED) -> dict[str, Any]:
    manifest = build_split_manifest(data_root, seed)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_split_files(manifest_path: Path, data_root: Path) -> dict[str, list[Path]]:
    """Resolve a frozen manifest into existing HDF5 paths without reshuffling.

    Separate train/validation/test lists make the test set unavailable to the
    trainer's validation-based checkpoint selection logic.
    """
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != SPLIT_PROTOCOL:
        raise ValueError(f"Unexpected split protocol: {manifest.get('protocol')!r}")
    records = inspect_successful_files(data_root)
    if manifest.get("source_corpus_hash") != corpus_hash(records):
        raise ValueError("Reconstruction corpus does not match the frozen split manifest")
    validate_split_manifest(manifest, records)
    by_id = {record["source_id"]: Path(record["path"]) for record in records}
    return {split: [by_id[source_id] for source_id in manifest["splits"][split]] for split in ("train", "validation", "test")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/gecv_reconstruction_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/gecv_reconstruction_v1_split_seed20260928.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    manifest = write_split_manifest(args.data_root, args.manifest, args.seed)
    print(json.dumps({"manifest": str(args.manifest), "total_counts": manifest["total_counts"], "class_counts": manifest["class_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
