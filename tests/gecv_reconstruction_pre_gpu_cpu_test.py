#!/usr/bin/env python3
"""CPU-only frozen-manifest gate for VIBE-GECV-RECON-v1.0.

This performs exactly one real batch forward/backward/optimizer step.  It does
not invoke VIBETrainer.run, extraction, CUDA, or any HDF5 write operation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataloading.reconstruction_split import LABELS, load_split_files, source_id_from_path
from dataloading.vibe_dataset import VIBEDataset, vibe_collate_fn
from losses.loss_function import VIBE_Loss
from model.vibe_model import VIBE_Transformer
from trainer.training_utils import WarmupCosineScheduler, compute_evaluation_metrics


CONFIG_PATH = REPO / "config" / "train_gecv_reconstruction_v1.yaml"
REPORT_PATH = REPO / "reports" / "gecv_pre_gpu_fix_report.json"


def expected_label(source_id: str) -> int:
    return 0 if source_id.startswith("videoH") else 1 if source_id.startswith("videoN") else 2


def choose_one_per_split_and_class(split_files: dict[str, list[Path]]) -> list[Path]:
    requested = (("train", 0), ("validation", 1), ("test", 2))
    paths = []
    for split, label in requested:
        path = next((item for item in split_files[split] if expected_label(source_id_from_path(item)) == label), None)
        if path is None:
            raise AssertionError(f"No label {label} sample in {split}")
        paths.append(path)
    return paths


def finite(value: torch.Tensor) -> float:
    if not torch.isfinite(value).all():
        raise AssertionError("Non-finite loss")
    return float(value.detach().cpu())


def main() -> int:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    if cfg["enable_sat"] or float(cfg["lambda_sat"]) != 0.0:
        raise AssertionError("Text-off reconstruction config must disable L_sat")
    if cfg["scheduler"] != "warmup_cosine" or cfg["optimizer"].lower() != "adamw":
        raise AssertionError("Unexpected reconstruction optimizer/scheduler")

    split_files = load_split_files(REPO / cfg["split_manifest"], REPO / cfg["data_root"])
    split_sets = {name: {source_id_from_path(path) for path in paths} for name, paths in split_files.items()}
    if split_sets["train"] & split_sets["validation"] or split_sets["train"] & split_sets["test"] or split_sets["validation"] & split_sets["test"]:
        raise AssertionError("Frozen split overlap")
    if {name: len(paths) for name, paths in split_files.items()} != {"train": 282, "validation": 61, "test": 61}:
        raise AssertionError("Frozen split totals differ from approved counts")

    paths = choose_one_per_split_and_class(split_files)
    dataset = VIBEDataset(paths, max_k=cfg["max_k"], max_t=cfg["max_t"],
                          schema_mode=cfg["dataset_schema"], reconstruction_gamma_source=cfg["reconstruction_gamma_source"])
    batch = next(iter(DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0, collate_fn=vibe_collate_fn)))
    if batch is None or batch["env_feat"] is not None or batch["text_anch"] is not None:
        raise AssertionError("Reconstruction adapter/text-off contract failed")
    if batch["labels"].tolist() != [0, 1, 2]:
        raise AssertionError(f"Unexpected canonical labels: {batch['labels'].tolist()}")

    device = torch.device("cpu")
    torch.manual_seed(int(cfg["seed"]))
    model = VIBE_Transformer(
        input_dim=cfg["input_dim"], latent_dim=cfg["latent_dim"], num_classes=cfg["num_classes"],
        attention_heads=cfg["attention_heads"], transformer_depth=cfg["transformer_depth"],
        dropout=cfg["dropout"], drop_path=cfg["drop_path"],
    ).to(device)
    criterion = VIBE_Loss(lambda_sat=cfg["lambda_sat"], lambda_ortho=cfg["lambda_ortho"], enable_sat=cfg["enable_sat"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = WarmupCosineScheduler(optimizer, cfg["lr"], cfg["epochs"], cfg["warmup_epochs"], cfg["eta_min"])
    lr_epochs_1_to_5 = [scheduler.lr_for_epoch(epoch) for epoch in range(1, 6)]

    model.train()
    output = model.forward_with_gamma(
        local_p=batch["local_p"], global_v=batch["global_v"], env_feat=batch["env_feat"],
        audio_seq=batch["audio_seq"], gamma=batch["gamma"], agent_mask=batch["agent_mask"],
        local_time_mask=batch["local_time_mask"], audio_time_mask=batch["audio_time_mask"],
        global_time_mask=batch["global_time_mask"], reconstruction_mode=True,
    )
    if tuple(output["logits"].shape) != (3, 3) or not torch.isfinite(output["logits"]).all():
        raise AssertionError("Invalid logits")
    losses = criterion(output, batch["labels"], batch)
    loss_values = [finite(value) for value in losses]
    if loss_values[3] != 0.0:
        raise AssertionError(f"L_sat must be zero, got {loss_values[3]}")
    optimizer.zero_grad(set_to_none=True)
    losses[0].backward()
    optimizer.step()

    predictions = torch.argmax(output["logits"].detach(), dim=1).tolist()
    metrics = compute_evaluation_metrics(batch["labels"].tolist(), predictions)
    if metrics["class_order"] != ["Positive", "Neutral", "Negative"] or len(metrics["confusion_matrix"]) != 3:
        raise AssertionError("Canonical evaluation class order failed")

    report = {
        "status": "READY_FOR_GPU",
        "scope": "CPU-only frozen-manifest pre-GPU gate; no full training was run",
        "split_manifest": str(REPO / cfg["split_manifest"]),
        "split_counts": {name: len(paths) for name, paths in split_files.items()},
        "cpu_samples": {split: source_id_from_path(path) for split, path in zip(("train", "validation", "test"), paths)},
        "labels": batch["labels"].tolist(),
        "batch_shapes": {"local_p": list(batch["local_p"].shape), "audio_seq": list(batch["audio_seq"].shape),
                         "global_v": list(batch["global_v"].shape), "gamma": list(batch["gamma"].shape)},
        "training_config": {key: cfg[key] for key in ("epochs", "batch_size", "lr", "weight_decay", "scheduler", "warmup_epochs", "latent_dim", "attention_heads", "transformer_depth", "dropout", "mixup_alpha", "lambda_sat", "lambda_ortho", "best_metric")},
        "paper_vs_reconstruction": {
            "matches_paper_targets": {"epochs": 30, "batch_size": 16, "learning_rate": 1e-4, "optimizer": "AdamW", "weight_decay": 1e-3, "scheduler": "cosine annealing after 3-epoch warmup", "latent_dim": 512, "attention_heads": 8, "transformer_depth": 2, "dropout": 0.3, "mixup": 0.2, "lambda_ortho": 0.1},
            "explicit_reconstruction_deviation": "text is unavailable, so text=None, enable_sat=false, and lambda_sat=0.0; this is not an exact paper-faithful text modality.",
            "split_provenance": "The frozen seed-20260928 split is a reconstruction split, not the unavailable official paper GECV split."
        },
        "optimizer_scheduler": {
            "optimizer": "AdamW",
            "weight_decay": cfg["weight_decay"],
            "schedule": "linear warmup for epochs 1-3, then cosine annealing over epochs 4-30",
            "eta_min": cfg["eta_min"],
            "checkpoint_selection_metric": "validation macro_f1 only"
        },
        "lr_epochs_1_to_5": lr_epochs_1_to_5,
        "forward": {"logits_shape": list(output["logits"].shape), "finite": True},
        "losses": {"total": loss_values[0], "hce": loss_values[1], "kl": loss_values[2], "sat": loss_values[3], "ortho": loss_values[4]},
        "backward": "PASS",
        "optimizer_step": "PASS",
        "text_off_evaluation": metrics,
        "checkpoint_behavior": "Trainer selects model_best.pth using validation macro_f1 only; test evaluation occurs after best checkpoint loading.",
        "remaining_blockers": ["No official paper GECV split is available; this is an explicitly labelled reconstruction split.", "A separate GPU smoke-test authorization is required before GPU work."],
        "gpu_readiness": "READY_FOR_GPU for a separate batch-16 reconstruction smoke test; 24 GB is the recommended first GPU class.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
