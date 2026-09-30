#!/usr/bin/env python3
"""Summarize completed VIBE-GECV-RECON-v1.0 seed-robustness runs.

This is a read-only reporting utility.  It does not start training, modify
checkpoints, or alter the frozen corpus or split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/robustness"
RUN_B_METRICS = ROOT / "reports/runB/gecv_runB_metrics.json"
RUN_B_PROVENANCE = ROOT / "reports/runB/gecv_runB_provenance.json"
SEED29_METRICS = OUT / "gecv_seed20260929_metrics.json"
SPLIT = ROOT / "data/manifests/gecv_reconstruction_v1_split_seed20260928.json"
BASE_CONFIG = ROOT / "config/train_gecv_reconstruction_v1.yaml"
SEED29_CONFIG = ROOT / "config/train_gecv_reconstruction_v1_seed20260929.yaml"
RUN_B_CKPT = ROOT / "checkpoints/gecv_reconstruction_v1_runB/model_best.pth"
SEED29_CKPT = ROOT / "checkpoints/gecv_reconstruction_v1_seed20260929/model_best.pth"
RUN_B_SHA = "3955147be0a5dea33ea0d2734c648508f50e2da72af9899bbb1e02efa4d8551a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, object]:
    def run(*args: str) -> str | None:
        result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("git", "status", "--porcelain")
    return {"head": run("git", "rev-parse", "HEAD"), "working_tree_clean": status == "", "available": status is not None}


def epoch_at(run: dict, epoch: int) -> dict:
    return next(record for record in run["epochs"] if record["epoch"] == epoch)


def row_from_run_b(metrics: dict, provenance: dict) -> dict:
    selection = provenance["checkpoint_selection"]
    best = epoch_at(metrics, selection["best_epoch"])["validation"]
    test = provenance["test_evaluation"]["test_metrics"]
    conf = provenance["configuration"]
    return {
        "seed": 20260928,
        "status": "COMPLETED_REFERENCE_RUN_B",
        "metrics_path": str(RUN_B_METRICS.relative_to(ROOT)),
        "config_path": conf["scientific_config_path"],
        "config_sha256": conf["scientific_config_sha256"],
        "split_manifest_sha256": conf["split_manifest_sha256"],
        "source_corpus_hash": conf["source_corpus_hash"],
        "gpu_name": provenance["gpu"]["name"],
        "best_epoch": selection["best_epoch"],
        "best_val_accuracy": best["acc"],
        "best_val_macro_f1": best["macro_f1"],
        "best_val_weighted_f1": best["weighted_f1"],
        "test_accuracy": test["acc"],
        "test_macro_f1": test["macro_f1"],
        "test_weighted_f1": test["weighted_f1"],
        "test_macro_precision": test["macro_precision"],
        "test_macro_recall": test["macro_recall"],
        "test_confusion_matrix": test["confusion_matrix"],
        "test_per_class": test["per_class"],
        "training_loop_completed_epoch": provenance["training_loop_completed_epoch"],
        "test_checkpoint_epoch": provenance["test_evaluation"]["test_checkpoint_epoch"],
        "checkpoint_path": selection["best_checkpoint_path"],
        "checkpoint_sha256": selection["best_checkpoint_sha256"],
    }


def row_from_seed29(metrics: dict) -> dict:
    best_epoch = metrics["best_epoch"]
    best = epoch_at(metrics, best_epoch)["validation"]
    test = metrics["final_test_metrics"]
    provenance = metrics["provenance"]
    return {
        "seed": 20260929,
        "status": "COMPLETED",
        "metrics_path": str(SEED29_METRICS.relative_to(ROOT)),
        "config_path": provenance["config_path"],
        "config_sha256": provenance["config_sha256"],
        "split_manifest_sha256": provenance["split_sha256"],
        "source_corpus_hash": provenance["source_corpus_hash"],
        "gpu_name": provenance["gpu_name"],
        "best_epoch": best_epoch,
        "best_val_accuracy": best["acc"],
        "best_val_macro_f1": best["macro_f1"],
        "best_val_weighted_f1": best["weighted_f1"],
        "test_accuracy": test["acc"],
        "test_macro_f1": test["macro_f1"],
        "test_weighted_f1": test["weighted_f1"],
        "test_macro_precision": test["macro_precision"],
        "test_macro_recall": test["macro_recall"],
        "test_confusion_matrix": test["confusion_matrix"],
        "test_per_class": test["per_class"],
        "training_loop_completed_epoch": metrics["training_loop_completed_epoch"],
        "test_checkpoint_epoch": metrics["test_checkpoint_epoch"],
        "checkpoint_path": metrics["test_checkpoint_path"],
        "checkpoint_sha256": metrics["test_checkpoint_sha256"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_b = json.loads(RUN_B_METRICS.read_text())
    run_b_provenance = json.loads(RUN_B_PROVENANCE.read_text())
    seed29 = json.loads(SEED29_METRICS.read_text())
    rows = [row_from_run_b(run_b, run_b_provenance), row_from_seed29(seed29)]
    integrity = {
        "run_b_protected_checkpoint": {"expected_sha256": RUN_B_SHA, "actual_sha256": sha256(RUN_B_CKPT), "status": "PASS" if sha256(RUN_B_CKPT) == RUN_B_SHA else "FAIL"},
        "seed20260929_checkpoint": {"expected_sha256": rows[1]["checkpoint_sha256"], "actual_sha256": sha256(SEED29_CKPT), "status": "PASS" if sha256(SEED29_CKPT) == rows[1]["checkpoint_sha256"] else "FAIL"},
        "frozen_split": {"expected_sha256": "75691440c5982cc9efa0311c9b8ba68139f318ba82858e42e3cacbd4271ec823", "actual_sha256": sha256(SPLIT), "status": "PASS" if sha256(SPLIT) == "75691440c5982cc9efa0311c9b8ba68139f318ba82858e42e3cacbd4271ec823" else "FAIL"},
        "baseline_config": {"expected_sha256": "7d3b3978025120eccef4a77a9d9685cde951a971394813f69a4ec2c74e269b39", "actual_sha256": sha256(BASE_CONFIG), "status": "PASS" if sha256(BASE_CONFIG) == "7d3b3978025120eccef4a77a9d9685cde951a971394813f69a4ec2c74e269b39" else "FAIL"},
    }
    document = {
        "protocol": "VIBE-GECV-RECON-v1.0",
        "purpose": "Seed robustness reporting; read-only post-run summary.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed_seed_count": 2,
        "planned_seed_count": 3,
        "pending_seed": 20260930,
        "three_seed_conclusion": "NOT_AVAILABLE: seed 20260930 has not been run; no three-seed aggregate or inferential claim is made.",
        "runs": rows,
        "integrity_checks": integrity,
        "git": git_state(),
    }
    (OUT / "gecv_3seed_summary.json").write_text(json.dumps(document, indent=2) + "\n")

    fields = ["seed", "status", "best_epoch", "best_val_accuracy", "best_val_macro_f1", "best_val_weighted_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1", "test_macro_precision", "test_macro_recall", "training_loop_completed_epoch", "test_checkpoint_epoch", "checkpoint_path", "checkpoint_sha256", "config_sha256", "split_manifest_sha256", "source_corpus_hash", "gpu_name"]
    with (OUT / "gecv_3seed_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in fields} for row in rows])

    lines = [
        "# VIBE-GECV-RECON-v1.0 Seed Robustness Status",
        "",
        "This is a post-run, read-only summary. It does not constitute a completed three-seed benchmark: seeds 20260928 and 20260929 are complete, while seed 20260930 is pending and was not started by this reporting step.",
        "",
        "| Seed | Status | Best epoch | Best validation macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 | Checkpoint SHA-256 |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['status']} | {row['best_epoch']} | {row['best_val_macro_f1']:.10f} | {row['test_accuracy']:.10f} | {row['test_macro_f1']:.10f} | {row['test_weighted_f1']:.10f} | `{row['checkpoint_sha256']}` |")
    lines += [
        "| 20260930 | PENDING — NOT RUN | — | — | — | — | — | — |",
        "",
        "## Protocol and selection",
        "",
        "Both completed runs use the frozen reconstruction corpus and split. Each trained for 30 epochs, selected `model_best.pth` strictly by validation macro-F1, and evaluated the held-out test partition once after training using that selected checkpoint. Test metrics were not used for selection.",
        "",
        "The 20260928 row is the existing protected Run B reference and was not retrained. The 20260929 configuration changes the seed only; its configuration SHA-256 is recorded in the CSV/JSON. No aggregate mean, standard deviation, ranking, significance statement, or three-seed conclusion is reported because only two of the planned three runs are complete.",
        "",
        "## Integrity checks",
        "",
    ]
    for name, result in integrity.items():
        lines.append(f"- `{name}`: **{result['status']}** (`{result['actual_sha256']}`)")
    lines += ["", "## Seed 20260929 held-out test confusion matrix", "", "Class order: Positive, Neutral, Negative.", "", "```text", *[" ".join(map(str, row)) for row in rows[1]["test_confusion_matrix"]], "```", ""]
    (OUT / "gecv_3seed_summary.md").write_text("\n".join(lines))
    print(json.dumps({"seed_20260929": rows[1], "integrity_checks": integrity}, indent=2))


if __name__ == "__main__":
    main()
