#!/usr/bin/env python3
"""Build the authoritative VIBE-GECV-RECON-v1.1 three-seed baseline package.

This post-processing utility is read-only with respect to HDF5 inputs, split
manifest, model source, and checkpoints. It reads completed V1.1 run reports
and writes new aggregate, audit, table, and figure artifacts under
``reports/final_baseline_v1_1``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from dataloading.reconstruction_split import corpus_hash, inspect_successful_files


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/final_baseline_v1_1"
FIGURES = OUT / "figures"
DATA = ROOT / "data/gecv_reconstruction_v1"
SPLIT = ROOT / "data/manifests/gecv_reconstruction_v1_split_seed20260928.json"
SOURCE = ROOT / "model/vibe_model.py"
V1_CONFIG = ROOT / "config/train_gecv_reconstruction_v1.yaml"
RUNS = {
    20260928: {
        "metrics": ROOT / "reports/robustness/gecv_v1_1_seed20260928_metrics.json",
        "config": ROOT / "config/train_gecv_reconstruction_v1_1.yaml",
        "checkpoint": ROOT / "checkpoints/gecv_reconstruction_v1_1_seed20260928/model_best.pth",
    },
    20260929: {
        "metrics": ROOT / "reports/robustness/gecv_v1_1_seed20260929_metrics.json",
        "config": ROOT / "config/train_gecv_reconstruction_v1_1_seed20260929.yaml",
        "checkpoint": ROOT / "checkpoints/gecv_reconstruction_v1_1_seed20260929/model_best.pth",
    },
    20260930: {
        "metrics": ROOT / "reports/robustness/gecv_v1_1_seed20260930_metrics.json",
        "config": ROOT / "config/train_gecv_reconstruction_v1_1_seed20260930.yaml",
        "checkpoint": ROOT / "checkpoints/gecv_reconstruction_v1_1_seed20260930/model_best.pth",
    },
}
EXPECTED = {
    "v1_0_config": "7d3b3978025120eccef4a77a9d9685cde951a971394813f69a4ec2c74e269b39",
    "split": "75691440c5982cc9efa0311c9b8ba68139f318ba82858e42e3cacbd4271ec823",
    "v1_1_model": "134920b263b1e3dbac1e0252b8eaefe59b1f8e052489408ba20db66834bc3b7c",
    "run_b": "3955147be0a5dea33ea0d2734c648508f50e2da72af9899bbb1e02efa4d8551a",
}
RUN_B = ROOT / "checkpoints/gecv_reconstruction_v1_runB/model_best.pth"
METRICS = ("best_val_accuracy", "best_val_macro_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1", "test_macro_precision", "test_macro_recall")
CLASSES = ("Positive", "Neutral", "Negative")
DPI = 300


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def git_state() -> dict[str, object]:
    def run(*args: str) -> str | None:
        result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    status = run("git", "status", "--porcelain")
    return {"available": status is not None, "head": run("git", "rev-parse", "HEAD"), "working_tree_clean": status == ""}


def epoch_record(run: dict, epoch: int) -> dict:
    return next(record for record in run["epochs"] if record["epoch"] == epoch)


def row(seed: int, report: dict) -> dict:
    item = RUNS[seed]
    best_epoch = int(report["best_epoch"])
    validation = epoch_record(report, best_epoch)["validation"]
    test = report["final_test_metrics"]
    return {
        "seed": seed,
        "metrics_path": relative(item["metrics"]),
        "config_path": relative(item["config"]),
        "config_sha256": sha256(item["config"]),
        "checkpoint_path": relative(item["checkpoint"]),
        "checkpoint_sha256": sha256(item["checkpoint"]),
        "best_epoch": best_epoch,
        "best_val_accuracy": float(validation["acc"]),
        "best_val_macro_f1": float(validation["macro_f1"]),
        "test_accuracy": float(test["acc"]),
        "test_macro_f1": float(test["macro_f1"]),
        "test_weighted_f1": float(test["weighted_f1"]),
        "test_macro_precision": float(test["macro_precision"]),
        "test_macro_recall": float(test["macro_recall"]),
        "test_confusion_matrix": test["confusion_matrix"],
        "test_per_class": test["per_class"],
        "training_loop_completed_epoch": int(report["training_loop_completed_epoch"]),
        "test_checkpoint_epoch": int(report["test_checkpoint_epoch"]),
    }


def status(condition: bool, detail: object) -> dict:
    return {"status": "PASS" if condition else "FAIL", "detail": detail}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
        "axes.titlesize": 12, "legend.fontsize": 9, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": DPI, "savefig.dpi": DPI,
    })


def save(fig: plt.Figure, stem: str) -> list[str]:
    if not fig.get_constrained_layout():
        fig.tight_layout()
    paths = [FIGURES / f"{stem}.png", FIGURES / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=DPI, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return [relative(path) for path in paths]


def make_figures(rows: list[dict], reports: dict[int, dict]) -> dict[str, list[str]]:
    setup_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    seeds = np.array([item["seed"] for item in rows])
    figures: dict[str, list[str]] = {}

    # Aggregate held-out test metrics, with each seed shown rather than hidden.
    fields = ("test_accuracy", "test_macro_f1", "test_weighted_f1", "test_macro_precision", "test_macro_recall")
    labels = ("Accuracy", "Macro-F1", "Weighted-F1", "Macro precision", "Macro recall")
    values = np.array([[item[field] for item in rows] for field in fields])
    means, stds = values.mean(axis=1), values.std(axis=1, ddof=1)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(fields))
    ax.bar(x, means * 100, yerr=stds * 100, capsize=4, color="#3d7eb5", edgecolor="#2b4f70", label="Mean ± sample SD")
    for seed_index, seed in enumerate(seeds):
        ax.scatter(x, values[:, seed_index] * 100, s=32, zorder=3, label=f"Seed {seed}", color=("#c03a2b", "#238b45", "#6a3d9a")[seed_index])
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Held-out test score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("V1.1 Three-seed Held-out Test Metrics", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    figures["three_seed_test_metrics"] = save(fig, "three_seed_test_metrics")

    # All recorded validation macro-F1 trajectories; marker identifies each selected checkpoint.
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    palette = ("#1b5eaa", "#c03a2b", "#238b45")
    for color, item in zip(palette, rows):
        report = reports[item["seed"]]
        epochs = [record["epoch"] for record in report["epochs"]]
        macro = [record["validation"]["macro_f1"] * 100 for record in report["epochs"]]
        ax.plot(epochs, macro, marker="o", markersize=2.8, color=color, label=f"Seed {item['seed']}")
        ax.scatter([item["best_epoch"]], [item["best_val_macro_f1"] * 100], s=58, color=color, edgecolor="black", zorder=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation macro-F1 (%)"); ax.set_ylim(0, 100)
    ax.set_title("V1.1 Validation Macro-F1 by Seed", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    figures["validation_macro_f1_by_seed"] = save(fig, "validation_macro_f1_by_seed")

    # Per-seed test confusion matrices are counts from selected checkpoints.
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8), sharey=True, layout="constrained")
    matrices = [np.asarray(item["test_confusion_matrix"]) for item in rows]
    vmax = max(int(matrix.max()) for matrix in matrices)
    for ax, item, matrix in zip(axes, rows, matrices):
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontweight="bold", color="white" if matrix[i,j] > vmax * 0.55 else "#222")
        ax.set_title(f"Seed {item['seed']}", fontweight="bold")
        ax.set_xticks(range(3), CLASSES, rotation=35, ha="right")
        ax.set_xlabel("Predicted")
    axes[0].set_yticks(range(3), CLASSES); axes[0].set_ylabel("True")
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03, label="Count")
    figures["test_confusion_matrices_by_seed"] = save(fig, "test_confusion_matrices_by_seed")

    # Per-class test F1: each individual seed remains visible.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(CLASSES)); width = 0.22
    for index, (item, color) in enumerate(zip(rows, palette)):
        values = [item["test_per_class"][klass]["f1"] * 100 for klass in CLASSES]
        ax.bar(x + (index - 1) * width, values, width, label=f"Seed {item['seed']}", color=color)
    ax.set_xticks(x, CLASSES); ax.set_ylim(0, 100); ax.set_ylabel("Held-out test F1 (%)")
    ax.set_title("V1.1 Held-out Per-class F1", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7); ax.set_axisbelow(True); ax.legend(frameon=False)
    figures["per_class_f1_by_seed"] = save(fig, "per_class_f1_by_seed")

    # Schedule is identical by configuration and recorded identically in all runs.
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for color, seed in zip(palette, seeds):
        records = reports[int(seed)]["epochs"]
        ax.plot([r["epoch"] for r in records], [r["learning_rate"] for r in records], color=color, label=f"Seed {seed}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate"); ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_title("V1.1 Warmup-cosine Learning-rate Schedule", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7); ax.set_axisbelow(True); ax.legend(frameon=False)
    figures["learning_rate_schedule_by_seed"] = save(fig, "learning_rate_schedule_by_seed")
    return figures


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {seed: json.loads(paths["metrics"].read_text()) for seed, paths in RUNS.items()}
    rows = [row(seed, reports[seed]) for seed in sorted(RUNS)]
    configurations = {seed: yaml.safe_load(paths["config"].read_text()) for seed, paths in RUNS.items()}
    strip_for_science = lambda cfg: {key: value for key, value in cfg.items() if key not in {"seed", "log_dir", "checkpoint_dir"}}
    scientific_config_match = all(strip_for_science(configurations[20260928]) == strip_for_science(configurations[seed]) for seed in RUNS)
    split = json.loads(SPLIT.read_text())
    actual_corpus_hash = corpus_hash(inspect_successful_files(DATA))
    checks = {
        "three_reports_complete": status(all(len(reports[seed].get("epochs", [])) == 30 and reports[seed].get("training_loop_completed_epoch") == 30 for seed in RUNS), {seed: len(reports[seed].get("epochs", [])) for seed in RUNS}),
        "checkpoint_selection": status(all(item["test_checkpoint_epoch"] == item["best_epoch"] and item["best_val_macro_f1"] == reports[item["seed"]]["best_validation_metric"] for item in rows), {item["seed"]: {"best_epoch": item["best_epoch"], "test_checkpoint_epoch": item["test_checkpoint_epoch"]} for item in rows}),
        "reported_checkpoint_hashes": status(all(sha256(RUNS[seed]["checkpoint"]) == reports[seed]["test_checkpoint_sha256"] for seed in RUNS), {seed: sha256(RUNS[seed]["checkpoint"]) for seed in RUNS}),
        "scientific_config_equivalence": status(scientific_config_match, "All V1.1 configs match after excluding only seed/log_dir/checkpoint_dir."),
        # Seed 20260928 uses the reusable base YAML, while its CLI supplied
        # isolated runtime output paths. The report captures those effective
        # paths; later seed YAMLs carry their isolated paths directly.
        "config_seeds_and_outputs": status(
            all(configurations[seed]["seed"] == seed for seed in RUNS)
            and all(reports[seed]["config"]["log_dir"].endswith(str(seed)) and reports[seed]["config"]["checkpoint_dir"].endswith(str(seed)) for seed in RUNS),
            {seed: {"source_config_log_dir": configurations[seed]["log_dir"], "source_config_checkpoint_dir": configurations[seed]["checkpoint_dir"], "effective_log_dir": reports[seed]["config"]["log_dir"], "effective_checkpoint_dir": reports[seed]["config"]["checkpoint_dir"]} for seed in RUNS},
        ),
        "frozen_split": status(sha256(SPLIT) == EXPECTED["split"] and split["source_corpus_hash"] == actual_corpus_hash, {"split_sha256": sha256(SPLIT), "source_corpus_hash": actual_corpus_hash}),
        "v1_1_source": status(sha256(SOURCE) == EXPECTED["v1_1_model"], sha256(SOURCE)),
        "v1_0_protection": status(sha256(V1_CONFIG) == EXPECTED["v1_0_config"] and sha256(RUN_B) == EXPECTED["run_b"], {"v1_0_config": sha256(V1_CONFIG), "run_b": sha256(RUN_B)}),
    }
    overall = "PASS" if all(check["status"] == "PASS" for check in checks.values()) else "FAIL"
    aggregate = {field: {"mean": float(np.mean([item[field] for item in rows])), "std_sample": float(np.std([item[field] for item in rows], ddof=1))} for field in METRICS}
    per_class = {klass: {metric: {"mean": float(np.mean([item["test_per_class"][klass][metric] for item in rows])), "std_sample": float(np.std([item["test_per_class"][klass][metric] for item in rows], ddof=1))} for metric in ("precision", "recall", "f1")} for klass in CLASSES}
    figures = make_figures(rows, reports)
    figure_manifest = {
        "protocol": "VIBE-GECV-RECON-v1.1", "source_data": [item["metrics_path"] for item in rows],
        "figures": [{"figure_name": name, "files": files, "metrics_used": "recorded V1.1 per-seed metrics", "description": name.replace("_", " ")} for name, files in figures.items()],
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(figure_manifest, indent=2) + "\n")
    document = {
        "protocol": "VIBE-GECV-RECON-v1.1", "purpose": "Authoritative corrected-protocol three-seed baseline aggregate.",
        "created_utc": datetime.now(timezone.utc).isoformat(), "training_performed_by_this_script": False,
        "statistical_convention": "mean and sample standard deviation (ddof=1) over three independently seeded runs; no significance claim.",
        "runs": rows, "aggregate_test_metrics": aggregate, "aggregate_per_class_test_metrics": per_class,
        "aggregate_confusion_matrix_sum": np.sum([item["test_confusion_matrix"] for item in rows], axis=0).tolist(),
        "integrity_audit": {"overall_status": overall, "checks": checks, "git": git_state()},
        "figures": figure_manifest["figures"],
    }
    (OUT / "gecv_v1_1_three_seed_aggregate.json").write_text(json.dumps(document, indent=2) + "\n")

    fields = ["seed", "best_epoch", "best_val_accuracy", "best_val_macro_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1", "test_macro_precision", "test_macro_recall", "training_loop_completed_epoch", "test_checkpoint_epoch", "checkpoint_path", "checkpoint_sha256", "config_path", "config_sha256", "metrics_path"]
    with (OUT / "gecv_v1_1_three_seed_aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows([{field: item[field] for field in fields} for item in rows])
        writer.writerow({"seed": "MEAN", **{field: aggregate[field]["mean"] for field in METRICS}})
        writer.writerow({"seed": "SAMPLE_STD", **{field: aggregate[field]["std_sample"] for field in METRICS}})

    audit_json = {"protocol": document["protocol"], "created_utc": document["created_utc"], "read_only": True, "overall_status": overall, "checks": checks, "run_checkpoints": [{"seed": item["seed"], "path": item["checkpoint_path"], "sha256": item["checkpoint_sha256"]} for item in rows]}
    (OUT / "gecv_v1_1_three_seed_integrity_audit.json").write_text(json.dumps(audit_json, indent=2) + "\n")

    lines = [
        "# VIBE-GECV-RECON-v1.1 Authoritative Three-seed Baseline",
        "",
        "This is the corrected-protocol baseline. It aggregates three completed, independently seeded V1.1 runs on the same frozen reconstruction corpus and train/validation/test split. Mean and sample standard deviation are descriptive summaries over three seeds; no statistical-significance claim is made.",
        "",
        "## Protocol", "",
        "V1.1 preserves the V1.0 reconstruction settings but corrects the gamma/scene-scale operation to use `gamma [B,1,1]`, `scene_scale [B,1,1]`, and `gamma_prime [B,1,1]`, preventing cross-sample broadcasting. V1.0 results remain historical and are not combined with these metrics.",
        "",
        "## Per-seed results", "",
        "| Seed | Best epoch | Best val. macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 | Macro precision | Macro recall |", "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [f"| {item['seed']} | {item['best_epoch']} | {item['best_val_macro_f1']:.4f} | {item['test_accuracy']:.4f} | {item['test_macro_f1']:.4f} | {item['test_weighted_f1']:.4f} | {item['test_macro_precision']:.4f} | {item['test_macro_recall']:.4f} |" for item in rows]
    lines += [
        f"| Mean | — | {aggregate['best_val_macro_f1']['mean']:.4f} | {aggregate['test_accuracy']['mean']:.4f} | {aggregate['test_macro_f1']['mean']:.4f} | {aggregate['test_weighted_f1']['mean']:.4f} | {aggregate['test_macro_precision']['mean']:.4f} | {aggregate['test_macro_recall']['mean']:.4f} |",
        f"| Sample SD | — | {aggregate['best_val_macro_f1']['std_sample']:.4f} | {aggregate['test_accuracy']['std_sample']:.4f} | {aggregate['test_macro_f1']['std_sample']:.4f} | {aggregate['test_weighted_f1']['std_sample']:.4f} | {aggregate['test_macro_precision']['std_sample']:.4f} | {aggregate['test_macro_recall']['std_sample']:.4f} |",
        "", "## Checkpoint provenance", "",
        "| Seed | Test checkpoint epoch | Checkpoint SHA-256 | Config SHA-256 |", "|---:|---:|---|---|",
    ]
    lines += [f"| {item['seed']} | {item['test_checkpoint_epoch']} | `{item['checkpoint_sha256']}` | `{item['config_sha256']}` |" for item in rows]
    lines += ["", "## Integrity and provenance audit", ""]
    lines += [f"- `{name}`: **{check['status']}** — {check['detail']}" for name, check in checks.items()]
    lines += ["", "## Figures and tables", "", *[f"- `{figure['figure_name']}`: {', '.join(figure['files'])}" for figure in figure_manifest['figures']], "", "## Scope", "", "This is a VIBE-GECV-RECON-v1.1 reconstruction-protocol baseline, not the original paper's exact benchmark or an exact paper-protocol reproduction. The reconstructed corpus/split, text-disabled setting, and other documented reconstruction limitations remain in force.", ""]
    (OUT / "gecv_v1_1_three_seed_aggregate.md").write_text("\n".join(lines))
    audit_lines = ["# V1.1 Three-seed Integrity and Provenance Audit", "", "Read-only audit; no model training, checkpoint change, corpus change, or split change was performed.", "", f"**Overall: {overall}**", ""]
    audit_lines += [f"- **{name}: {check['status']}** — {check['detail']}" for name, check in checks.items()]
    (OUT / "gecv_v1_1_three_seed_integrity_audit.md").write_text("\n".join(audit_lines) + "\n")
    print(json.dumps({"status": overall, "aggregate": {field: aggregate[field] for field in METRICS}, "output": relative(OUT)}, indent=2))


if __name__ == "__main__":
    main()
