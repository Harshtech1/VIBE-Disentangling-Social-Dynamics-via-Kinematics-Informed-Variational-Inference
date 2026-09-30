#!/usr/bin/env python3
"""One-batch GPU smoke test for the frozen VIBE-GECV-RECON-v1.0 protocol.

This script intentionally performs one training batch and one tiny validation
inference only.  It never invokes the 30-epoch trainer, feature extraction, or
any HDF5 write operation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataloading.reconstruction_split import SMOKE_ONLY_IDS, load_split_files, source_id_from_path
from dataloading.vibe_dataset import VIBEDataset, vibe_collate_fn
from losses.loss_function import VIBE_Loss
from model.vibe_model import VIBE_Transformer
from trainer.training_utils import WarmupCosineScheduler, compute_evaluation_metrics


CONFIG_PATH = REPO / "config" / "train_gecv_reconstruction_v1.yaml"
REPORT_PATH = REPO / "reports" / "gecv_gpu_smoke_test.json"
EXPECTED_LR = [3.333333e-5, 6.666667e-5, 1.0e-4, 9.966192e-5, 9.865224e-5]


def report_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def gpu_memory_mb() -> dict[str, float]:
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024 ** 2, 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024 ** 2, 2),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024 ** 2, 2),
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def model_forward(model: VIBE_Transformer, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return model.forward_with_gamma(
        local_p=batch["local_p"], global_v=batch["global_v"], env_feat=batch["env_feat"],
        audio_seq=batch["audio_seq"], gamma=batch["gamma"], agent_mask=batch["agent_mask"],
        local_time_mask=batch["local_time_mask"], audio_time_mask=batch["audio_time_mask"],
        global_time_mask=batch["global_time_mask"], reconstruction_mode=True,
    )


def first_validation_per_class(paths: list[Path]) -> list[Path]:
    selected: dict[str, Path] = {}
    for path in paths:
        source_id = source_id_from_path(path)
        key = "Positive" if source_id.startswith("videoH") else "Neutral" if source_id.startswith("videoN") else "Negative"
        selected.setdefault(key, path)
    if set(selected) != {"Positive", "Neutral", "Negative"}:
        raise RuntimeError(f"Validation split does not expose all classes: {sorted(selected)}")
    return [selected["Positive"], selected["Neutral"], selected["Negative"]]


def main() -> int:
    report: dict[str, Any] = {"protocol": "VIBE-GECV-RECON-v1.0", "scope": "one GPU train batch plus tiny validation inference; no full training"}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this process; run only in the researcher-owned GPU terminal.")
        cfg = yaml.safe_load(CONFIG_PATH.read_text())
        if cfg.get("enable_sat") or float(cfg.get("lambda_sat", 1.0)) != 0.0:
            raise RuntimeError("Text-off reconstruction config is invalid: L_sat must be disabled and zero.")
        if cfg.get("batch_size") != 16:
            raise RuntimeError("GPU smoke test requires frozen batch_size=16.")
        if cfg.get("scheduler") != "warmup_cosine":
            raise RuntimeError("GPU smoke test requires frozen warmup_cosine scheduler.")

        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(0)
        report["gpu"] = {"name": properties.name, "vram_gb": round(properties.total_memory / 1024 ** 3, 2),
                         "torch": torch.__version__, "torch_cuda": torch.version.cuda}
        split_files = load_split_files(REPO / cfg["split_manifest"], REPO / cfg["data_root"])
        memberships = {split: {source_id_from_path(path) for path in paths} for split, paths in split_files.items()}
        if any(source_id in ids for source_id in SMOKE_ONLY_IDS for ids in memberships.values()):
            raise RuntimeError("Smoke-only IDs leaked into the frozen train/validation/test manifest.")
        if memberships["train"] & memberships["validation"] or memberships["train"] & memberships["test"] or memberships["validation"] & memberships["test"]:
            raise RuntimeError("Frozen split overlap detected.")
        (REPO / cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        report["split"] = {"counts": {name: len(paths) for name, paths in split_files.items()},
                           "smoke_only_ids": list(SMOKE_ONLY_IDS), "checkpoint_directory": str(REPO / cfg["checkpoint_dir"]),
                           "test_data_used_for_training": False}

        train_dataset = VIBEDataset(split_files["train"], max_k=cfg["max_k"], max_t=cfg["max_t"],
                                    schema_mode=cfg["dataset_schema"], reconstruction_gamma_source=cfg["reconstruction_gamma_source"])
        train_batch = next(iter(DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0,
                                           pin_memory=True, collate_fn=vibe_collate_fn)))
        if train_batch is None or tuple(train_batch["local_p"].shape) != (16, 8, 32, 768):
            raise RuntimeError("Failed to load one real batch of 16 reconstruction samples.")
        if train_batch["text_anch"] is not None or train_batch["env_feat"] is not None:
            raise RuntimeError("Reconstruction text/env contract violated.")

        model = VIBE_Transformer(input_dim=cfg["input_dim"], latent_dim=cfg["latent_dim"], num_classes=cfg["num_classes"],
                                 attention_heads=cfg["attention_heads"], transformer_depth=cfg["transformer_depth"],
                                 dropout=cfg["dropout"], drop_path=cfg["drop_path"]).to(device)
        criterion = VIBE_Loss(lambda_sat=cfg["lambda_sat"], lambda_ortho=cfg["lambda_ortho"], enable_sat=cfg["enable_sat"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = WarmupCosineScheduler(optimizer, cfg["lr"], cfg["epochs"], cfg["warmup_epochs"], cfg["eta_min"])
        lr_sequence = [scheduler.lr_for_epoch(epoch) for epoch in range(1, 6)]
        if not all(abs(actual - expected) <= 1e-10 for actual, expected in zip(lr_sequence, EXPECTED_LR)):
            raise RuntimeError(f"Unexpected LR sequence: {lr_sequence}")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        train_batch = move_batch(train_batch, device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = model_forward(model, train_batch)
        losses = criterion(output, train_batch["labels"], train_batch)
        optimizer.zero_grad(set_to_none=True)
        losses[0].backward()
        gradients_finite = all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        loss_values = [float(value.detach().cpu()) for value in losses]
        if tuple(output["logits"].shape) != (16, 3) or not torch.isfinite(output["logits"]).all():
            raise RuntimeError("Invalid/non-finite train logits.")
        if not all(torch.isfinite(value).all() for value in losses) or loss_values[3] != 0.0:
            raise RuntimeError(f"Invalid loss values or nonzero L_sat: {loss_values}")
        if not gradients_finite:
            raise RuntimeError("Non-finite gradients.")
        report["train_batch"] = {
            "shapes": {"local_p": list(train_batch["local_p"].shape), "audio_seq": list(train_batch["audio_seq"].shape),
                       "global_v": list(train_batch["global_v"].shape), "gamma": list(train_batch["gamma"].shape),
                       "logits": list(output["logits"].shape)},
            "losses": {"total": loss_values[0], "hce": loss_values[1], "kl": loss_values[2], "sat": loss_values[3], "ortho": loss_values[4]},
            "gradients_finite": bool(gradients_finite), "optimizer_step": "PASS", "elapsed_seconds": round(elapsed, 4),
            "memory_mb": gpu_memory_mb(),
        }

        # Validation uses three real validation samples only.  It does not touch
        # the frozen test membership and executes no backward pass.
        validation_paths = first_validation_per_class(split_files["validation"])
        validation_dataset = VIBEDataset(validation_paths, max_k=cfg["max_k"], max_t=cfg["max_t"],
                                         schema_mode=cfg["dataset_schema"], reconstruction_gamma_source=cfg["reconstruction_gamma_source"])
        validation_batch = next(iter(DataLoader(validation_dataset, batch_size=3, shuffle=False, num_workers=0,
                                                pin_memory=True, collate_fn=vibe_collate_fn)))
        model.eval()
        with torch.no_grad():
            validation_batch = move_batch(validation_batch, device)
            validation_output = model_forward(model, validation_batch)
            predictions = torch.argmax(validation_output["logits"], dim=1).cpu().tolist()
            metrics = compute_evaluation_metrics(validation_batch["labels"].cpu().tolist(), predictions)
        if metrics["class_order"] != ["Positive", "Neutral", "Negative"] or len(metrics["confusion_matrix"]) != 3:
            raise RuntimeError("Canonical validation metric order failed.")
        report["validation"] = {"source_ids": [source_id_from_path(path) for path in validation_paths],
                                "labels": validation_batch["labels"].cpu().tolist(), "logits_shape": list(validation_output["logits"].shape),
                                "metrics": metrics}
        report["scheduler"] = {"epoch_1_to_5_lr": lr_sequence, "expected": EXPECTED_LR}
        report["full_training_command_not_executed"] = "python trainer/train_model.py --config config/train_gecv_reconstruction_v1.yaml"
        report["status"] = "GPU_SMOKE_PASS"
        exit_code = 0
    except Exception as exc:
        report["status"] = "GPU_SMOKE_FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        exit_code = 2
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=report_value) + "\n")
    print(json.dumps(report, indent=2, default=report_value))
    print(report["status"])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
