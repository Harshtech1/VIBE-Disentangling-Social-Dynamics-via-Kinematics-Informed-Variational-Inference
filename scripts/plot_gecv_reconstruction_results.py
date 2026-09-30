#!/usr/bin/env python3
"""Create publication-quality plots from recorded VIBE-GECV-RECON-v1.0 Run B metrics.

This script is read-only with respect to experiment inputs and artifacts: it reads
the completed Run B JSON and frozen split manifest, then writes figures and their
documentation beneath reports/figures/gecv_reconstruction_v1/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports/runB/gecv_runB_metrics.json"
SPLIT_PATH = ROOT / "data/manifests/gecv_reconstruction_v1_split_seed20260928.json"
OUT_DIR = ROOT / "reports/figures/gecv_reconstruction_v1"
DPI = 300
COLORS = {"accuracy": "#1b5eaa", "macro": "#c03a2b", "weighted": "#238b45", "precision": "#6a3d9a", "recall": "#e67e22"}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
        "axes.titlesize": 12, "legend.fontsize": 9, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": DPI, "savefig.dpi": DPI,
    })


def save(fig: plt.Figure, stem: str) -> list[str]:
    fig.tight_layout()
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(p.relative_to(ROOT)) for p in (png, pdf)]


def epoch_arrays(metrics: dict) -> tuple[np.ndarray, list[dict]]:
    records = metrics["epochs"]
    return np.array([record["epoch"] for record in records]), records


def line_axes(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    ax.set_axisbelow(True)


def add_best_epoch(ax: plt.Axes, best_epoch: int, value: float | None = None) -> None:
    ax.axvline(best_epoch, color="#4d4d4d", linestyle="--", linewidth=1.0, label=f"Selected best epoch ({best_epoch})")
    if value is not None:
        ax.annotate(f"Best macro-F1 = {value:.4f}", xy=(best_epoch, value * 100),
                    xytext=(best_epoch + 1.0, max(3, value * 100 - 8)),
                    arrowprops={"arrowstyle": "-", "color": "#4d4d4d"}, fontsize=9)


def confusion_figure(matrix: np.ndarray, classes: list[str], normalized: bool) -> plt.Figure:
    display = matrix / matrix.sum(axis=1, keepdims=True) if normalized else matrix
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(display, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Proportion" if normalized else "Count")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = f"{display[row, col]:.2f}" if normalized else str(int(display[row, col]))
            color = "white" if display[row, col] > display.max() * 0.55 else "#222222"
            ax.text(col, row, value, ha="center", va="center", color=color, fontweight="bold")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Held-out Test Confusion Matrix" + (" (Row-normalized)" if normalized else " (Counts)"), loc="left", fontweight="bold")
    return fig


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_PATH.read_text())
    split = json.loads(SPLIT_PATH.read_text())
    epochs, records = epoch_arrays(metrics)
    best_epoch = metrics["best_epoch"]
    best_macro = metrics["best_validation_metric"]
    validation = [record["validation"] for record in records]
    generated: dict[str, list[str]] = {}

    # 1. Validation performance.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, [m["acc"] * 100 for m in validation], marker="o", markersize=3, color=COLORS["accuracy"], label="Validation Accuracy")
    ax.plot(epochs, [m["macro_f1"] * 100 for m in validation], marker="o", markersize=3, color=COLORS["macro"], label="Validation Macro-F1")
    ax.plot(epochs, [m["weighted_f1"] * 100 for m in validation], marker="o", markersize=3, color=COLORS["weighted"], label="Validation Weighted-F1")
    add_best_epoch(ax, best_epoch, best_macro)
    line_axes(ax, "Validation Performance Evolution", "Score (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", frameon=False)
    generated["validation_performance_evolution"] = save(fig, "validation_performance_evolution")

    # 2. Validation macro precision/recall.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, [m["macro_precision"] * 100 for m in validation], marker="o", markersize=3, color=COLORS["precision"], label="Macro Precision")
    ax.plot(epochs, [m["macro_recall"] * 100 for m in validation], marker="o", markersize=3, color=COLORS["recall"], label="Macro Recall")
    add_best_epoch(ax, best_epoch)
    line_axes(ax, "Validation Precision and Recall Evolution", "Score (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", frameon=False)
    generated["validation_precision_recall_evolution"] = save(fig, "validation_precision_recall_evolution")

    # 3. Training loss. Log scale retains visibility of the recorded KL dominance.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for key, label, color in [
        ("total_mean", "Total loss", "#222222"), ("hce_mean", "HCE loss", "#1b5eaa"),
        ("kl_mean", "KL loss", "#c03a2b"), ("ortho_mean", "Orthogonality loss", "#238b45"),
    ]:
        ax.plot(epochs, [r["train_loss"][key] for r in records], marker="o", markersize=2.5, color=color, label=label)
    line_axes(ax, "Training Loss Evolution", "Mean loss per training batch (log scale)")
    ax.set_yscale("log")
    ax.legend(loc="best", frameon=False)
    generated["training_loss_evolution"] = save(fig, "training_loss_evolution")

    # 4. Learning-rate schedule.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, [r["learning_rate"] for r in records], marker="o", markersize=3, color="#1b5eaa")
    line_axes(ax, "Learning-rate Schedule", "Learning rate")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    generated["learning_rate_schedule"] = save(fig, "learning_rate_schedule")

    # 5. Held-out test matrices.
    test = metrics["final_test_metrics"]
    matrix = np.array(test["confusion_matrix"])
    classes = test["class_order"]
    generated["test_confusion_matrix_counts"] = save(confusion_figure(matrix, classes, False), "test_confusion_matrix_counts")
    generated["test_confusion_matrix_row_normalized"] = save(confusion_figure(matrix, classes, True), "test_confusion_matrix_row_normalized")

    # 6. Per-class test metrics.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(classes))
    width = 0.23
    for shift, field, label, color in [(-width, "precision", "Precision", "#1b5eaa"), (0, "recall", "Recall", "#e67e22"), (width, "f1", "F1", "#238b45")]:
        values = [test["per_class"][name][field] * 100 for name in classes]
        ax.bar(x + shift, values, width, label=label, color=color)
    ax.set_xticks(x, classes)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Held-out Test Per-class Metrics", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=3, loc="lower left")
    generated["test_per_class_metrics"] = save(fig, "test_per_class_metrics")

    # 7. Frozen split class distribution.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    split_names = ["train", "validation", "test"]
    x = np.arange(len(classes))
    width = 0.23
    for i, (name, color) in enumerate(zip(split_names, ["#1b5eaa", "#e67e22", "#238b45"])):
        ax.bar(x + (i - 1) * width, [split["class_counts"][name][klass] for klass in classes], width, label=name.title(), color=color)
    ax.set_xticks(x, classes)
    ax.set_ylabel("Sample count")
    ax.set_title("Frozen Split Class Distribution", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    generated["split_class_distribution"] = save(fig, "split_class_distribution")

    # 8. Selected checkpoint context.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, [m["acc"] * 100 for m in validation], marker="o", markersize=3, color=COLORS["accuracy"], label="Validation Accuracy")
    ax.plot(epochs, [m["macro_f1"] * 100 for m in validation], marker="o", markersize=3, color=COLORS["macro"], label="Validation Macro-F1")
    add_best_epoch(ax, best_epoch, best_macro)
    ax.annotate("Held-out test evaluated\nwith this checkpoint", xy=(best_epoch, best_macro * 100), xytext=(15, 67),
                arrowprops={"arrowstyle": "->", "color": "#4d4d4d"}, fontsize=9)
    line_axes(ax, "Checkpoint Selection Evolution", "Score (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", frameon=False)
    generated["checkpoint_evolution"] = save(fig, "checkpoint_evolution")

    manifest = {
        "protocol": metrics["protocol"],
        "source_data": {"run_b_metrics": str(METRICS_PATH.relative_to(ROOT)), "split_manifest": str(SPLIT_PATH.relative_to(ROOT))},
        "figures": [
            {"figure_name": name, "files": files, "source_data": [str(METRICS_PATH.relative_to(ROOT))] + ([str(SPLIT_PATH.relative_to(ROOT))] if name == "split_class_distribution" else []),
             "metrics_used": ({"validation": ["acc", "macro_f1", "weighted_f1"]} if name == "validation_performance_evolution" else
                              {"validation": ["macro_precision", "macro_recall"]} if name == "validation_precision_recall_evolution" else
                              {"training_loss": ["total_mean", "hce_mean", "kl_mean", "ortho_mean"]} if name == "training_loss_evolution" else
                              {"learning_rate": "learning_rate"} if name == "learning_rate_schedule" else
                              {"final_test_metrics": "confusion_matrix"} if name.startswith("test_confusion_matrix") else
                              {"final_test_metrics": "per_class"} if name == "test_per_class_metrics" else
                              {"split_manifest": "class_counts"} if name == "split_class_distribution" else
                              {"validation": ["acc", "macro_f1"], "test_checkpoint_epoch": "test_checkpoint_epoch"}),
             "best_epoch_if_applicable": best_epoch if name in {"validation_performance_evolution", "validation_precision_recall_evolution", "checkpoint_evolution"} else None,
             "description": name.replace("_", " ")}
            for name, files in generated.items()
        ],
    }
    (OUT_DIR / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    readme = f"""# VIBE-GECV-RECON-v1.0 Baseline Evolution Figures

These figures are derived solely from recorded RUN B metrics; no training was performed to generate them.

- Source JSON: `{METRICS_PATH.relative_to(ROOT)}`
- Frozen split manifest: `{SPLIT_PATH.relative_to(ROOT)}`
- Best validation-selected epoch: **{best_epoch}** (macro-F1 **{best_macro:.16f}**)
- Test checkpoint epoch: **{metrics['test_checkpoint_epoch']}**
- Held-out test accuracy: **{test['acc']:.16f}**
- Held-out test macro-F1: **{test['macro_f1']:.16f}**
- Held-out test weighted-F1: **{test['weighted_f1']:.16f}**

## Metric definitions

- **Accuracy:** proportion of held-out or validation samples classified correctly.
- **Macro precision, recall, and F1:** unweighted mean of the corresponding per-class values over Positive, Neutral, and Negative.
- **Weighted F1:** per-class F1 averaged with class support as weights.
- **Training losses:** recorded per-epoch means over training batches: total objective, hierarchical cross-entropy (HCE), KL term, and orthogonality term. The loss plot uses a logarithmic y-axis because the recorded KL term dominates the scale; no dual axis is used.
- **Normalized confusion matrix:** each true-label row divided by its row total.

## Reconstruction caveats

This is a VIBE-GECV-RECON-v1.0 reconstruction-protocol result, with its frozen reconstructed corpus and split. Text/SAT are disabled. It is not a claim of exact original-paper protocol reproduction, does not use the unavailable official paper split, and should not be presented as an exact reproduction of the paper's three-seed benchmark.
"""
    (OUT_DIR / "README.md").write_text(readme)
    print("BASELINE PLOTTING COMPLETE")
    for path in sorted(OUT_DIR.iterdir()):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
