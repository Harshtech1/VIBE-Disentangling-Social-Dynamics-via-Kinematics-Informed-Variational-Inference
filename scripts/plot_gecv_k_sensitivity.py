#!/usr/bin/env python3
"""Plot completed VIBE-GECV-RECON-v1.0 effective-K sensitivity results.

This script reads completed reports only. It never trains, edits model/data, or
modifies checkpoints; it writes figures and publication artifacts under
reports/figures/gecv_k_sensitivity/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    3: ROOT / "reports/k_sensitivity/gecv_k3_metrics.json",
    5: ROOT / "reports/k_sensitivity/gecv_k5_metrics.json",
    8: ROOT / "reports/runB/gecv_runB_metrics.json",
}
RUN_B_PROVENANCE = ROOT / "reports/runB/gecv_runB_provenance.json"
SPLIT_PATH = ROOT / "data/manifests/gecv_reconstruction_v1_split_seed20260928.json"
INTEGRATION_PATH = ROOT / "reports/k_sensitivity/gecv_effective_k_cpu_integration.json"
OUT = ROOT / "reports/figures/gecv_k_sensitivity"
DPI = 300
COLORS = ["#1b5eaa", "#e67e22", "#238b45"]


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
        "axes.titlesize": 12, "legend.fontsize": 9, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": DPI, "savefig.dpi": DPI,
    })


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(fig: plt.Figure, name: str) -> list[str]:
    fig.tight_layout()
    png, pdf = OUT / f"{name}.png", OUT / f"{name}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(p.relative_to(ROOT)) for p in (png, pdf)]


def records() -> list[dict]:
    run_b_provenance = json.loads(RUN_B_PROVENANCE.read_text())
    rows = []
    for effective_k, path in REPORTS.items():
        report = json.loads(path.read_text())
        best_epoch = report["best_epoch"]
        best_validation = next(entry["validation"] for entry in report["epochs"] if entry["epoch"] == best_epoch)
        test = report["final_test_metrics"]
        provenance = report.get("provenance", {})
        if effective_k == 8:
            provenance = {
                "config_path": run_b_provenance["configuration"]["scientific_config_path"],
                "config_sha256": run_b_provenance["configuration"]["scientific_config_sha256"],
                "split_manifest_path": run_b_provenance["configuration"]["split_manifest_path"],
                "split_sha256": run_b_provenance["configuration"]["split_manifest_sha256"],
                "source_corpus_hash": run_b_provenance["configuration"]["source_corpus_hash"],
                "gpu_name": run_b_provenance["gpu"]["name"],
            }
        rows.append({
            "effective_k": effective_k,
            "storage_max_k": report.get("storage_max_k", 8),
            "best_epoch": best_epoch,
            "best_val_accuracy": best_validation["acc"],
            "best_val_macro_f1": report["best_validation_metric"],
            "test_accuracy": test["acc"],
            "test_macro_f1": test["macro_f1"],
            "test_weighted_f1": test["weighted_f1"],
            "test_macro_precision": test["macro_precision"],
            "test_macro_recall": test["macro_recall"],
            "checkpoint_path": report["test_checkpoint_path"],
            "checkpoint_sha256": report["test_checkpoint_sha256"],
            "config_path": provenance["config_path"],
            "config_sha256": provenance["config_sha256"],
            "split_manifest_path": provenance["split_manifest_path"],
            "split_sha256": provenance["split_sha256"],
            "source_corpus_hash": provenance["source_corpus_hash"],
            "gpu_name": provenance["gpu_name"],
            "source_json": str(path.relative_to(ROOT)),
        })
    return rows


def metric_plot(rows: list[dict], key: str, title: str, ylabel: str, filename: str, percent: bool) -> list[str]:
    k = [row["effective_k"] for row in rows]
    values = np.array([row[key] for row in rows]) * (100 if percent else 1)
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.plot(k, values, color="#1b5eaa", marker="o", markersize=7, linewidth=1.8)
    for x, y in zip(k, values):
        ax.annotate(f"{y:.2f}%" if percent else f"{y:.4f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(k)
    ax.set_xlabel("Effective K")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    ax.set_axisbelow(True)
    if percent:
        ax.set_ylim(90, 100)
    else:
        ax.set_ylim(0.90, 0.98)
    return save(fig, filename)


def combined_plot(rows: list[dict]) -> list[str]:
    k = [row["effective_k"] for row in rows]
    panels = [
        ("test_accuracy", "Held-out Test Accuracy", "Accuracy (%)", True),
        ("test_macro_f1", "Held-out Test Macro-F1", "Macro-F1", False),
        ("best_val_macro_f1", "Best Validation Macro-F1", "Macro-F1", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), sharex=True)
    for ax, (key, title, ylabel, percent), color in zip(axes, panels, COLORS):
        values = np.array([row[key] for row in rows]) * (100 if percent else 1)
        ax.plot(k, values, color=color, marker="o", markersize=6, linewidth=1.8)
        for x, y in zip(k, values):
            ax.annotate(f"{y:.2f}%" if percent else f"{y:.4f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Effective K")
        ax.set_ylabel(ylabel)
        ax.set_xticks(k)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_ylim((90, 100) if percent else (0.90, 0.98))
    return save(fig, "k_sensitivity_combined")


def write_csv(rows: list[dict]) -> str:
    path = OUT / "k_sensitivity_summary_table.csv"
    columns = ["effective_k", "best_epoch", "best_val_accuracy", "best_val_macro_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1", "test_macro_precision", "test_macro_recall"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in columns} for row in rows])
    return str(path.relative_to(ROOT))


def write_report(rows: list[dict], protected_hash: str, integration: dict) -> str:
    path = OUT / "k_sensitivity_report.md"
    lines = [
        "# VIBE-GECV-RECON-v1.0 Effective-K Sensitivity",
        "",
        "## Objective",
        "",
        "Describe recorded reconstruction-protocol outcomes when the effective number of active stored agents is K=3, 5, or 8.",
        "",
        "## Protocol",
        "",
        "All runs use the frozen source corpus, split manifest, seed, 30 epochs, AdamW optimizer, warmup-cosine schedule, model topology, text-disabled/SAT-disabled loss configuration, and stored physical HDF5 shape. This is a reconstruction-protocol sensitivity experiment, not the original paper's full three-seed benchmark.",
        "",
        "Physical storage remained K=8. For effective K=3 and K=5, only the first deterministically stored slots were active; later slots were masked. Gamma was recomputed from active valid synchrony pairs in the corresponding stored synchrony submatrix. The CPU integration gate passed and recorded inactive-agent logit invariance with a maximum absolute difference of " + f"{integration['inactive_agent_logit_invariance']['max_abs_logit_difference']:.1f}" + ". K=8 is the existing frozen Run B baseline and was not retrained.",
        "",
        "## Recorded Results",
        "",
        "| Effective K | Best epoch | Best val. accuracy | Best val. macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['effective_k']} | {row['best_epoch']} | {row['best_val_accuracy']:.4f} | {row['best_val_macro_f1']:.4f} | {row['test_accuracy']:.4f} | {row['test_macro_f1']:.4f} | {row['test_weighted_f1']:.4f} |")
    lines += [
        "| 10 | NOT RUN | — | — | — | — | — |",
        "",
        "K=10 was not fabricated or approximated; it requires raw-video re-extraction because the completed corpus stores only eight agent slots.",
        "",
        "## Checkpoints",
        "",
        "| Effective K | Checkpoint SHA-256 |",
        "|---:|---|",
    ]
    lines += [f"| {row['effective_k']} | `{row['checkpoint_sha256']}` |" for row in rows]
    lines += [
        "",
        f"The protected Run B K=8 checkpoint was verified after K=3/K=5 processing: `{protected_hash}`.",
        "",
        "## Interpretation Boundary",
        "",
        "These recorded values are reported descriptively. They do not establish statistical significance and are not ranked or used to select a winner. They are not claims about the original paper's exact protocol or its full three-seed benchmark.",
    ]
    path.write_text("\n".join(lines) + "\n")
    return str(path.relative_to(ROOT))


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = records()
    integration = json.loads(INTEGRATION_PATH.read_text())
    protected_path = ROOT / "checkpoints/gecv_reconstruction_v1_runB/model_best.pth"
    protected_hash = sha256(protected_path)
    expected = next(row["checkpoint_sha256"] for row in rows if row["effective_k"] == 8)
    if protected_hash != expected:
        raise RuntimeError("Protected Run B checkpoint hash mismatch")

    generated = {
        "k_vs_test_accuracy": metric_plot(rows, "test_accuracy", "Effective K vs Held-out Test Accuracy", "Accuracy (%)", "k_vs_test_accuracy", True),
        "k_vs_test_macro_f1": metric_plot(rows, "test_macro_f1", "Effective K vs Held-out Test Macro-F1", "Macro-F1", "k_vs_test_macro_f1", False),
        "k_vs_validation_macro_f1": metric_plot(rows, "best_val_macro_f1", "Effective K vs Best Validation Macro-F1", "Macro-F1", "k_vs_validation_macro_f1", False),
        "k_sensitivity_combined": combined_plot(rows),
    }
    csv_path = write_csv(rows)
    report_path = write_report(rows, protected_hash, integration)
    manifest = {
        "protocol": "VIBE-GECV-RECON-v1.0",
        "experiment": "effective-K sensitivity",
        "source_jsons": [row["source_json"] for row in rows],
        "integration_report": str(INTEGRATION_PATH.relative_to(ROOT)),
        "split_manifest": str(SPLIT_PATH.relative_to(ROOT)),
        "split_manifest_sha256": rows[0]["split_sha256"],
        "source_corpus_hash": rows[0]["source_corpus_hash"],
        "runs": rows,
        "k10": "NOT RUN — requires raw-video re-extraction",
        "run_b_protected_checkpoint": {"path": str(protected_path.relative_to(ROOT)), "sha256": protected_hash, "status": "PASS"},
        "generated_figures": generated,
        "generated_artifacts": [csv_path, report_path],
    }
    (OUT / "k_sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("K-SENSITIVITY PLOTTING COMPLETE")
    print("K | Best Val Macro-F1 | Test Accuracy | Test Macro-F1")
    for row in rows:
        print(f"{row['effective_k']} | {row['best_val_macro_f1']:.4f} | {row['test_accuracy'] * 100:.2f}% | {row['test_macro_f1']:.4f}")
    print("Run B protected checkpoint: PASS")
    for path in sorted(OUT.iterdir()):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
