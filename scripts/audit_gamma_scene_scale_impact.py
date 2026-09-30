"""Read-only impact assessment for the v1.0 gamma/scene-scale broadcast issue.

The corrected path is evaluated only in memory by wrapping ``scene_scaler`` so
that it returns [B, 1, 1].  This script does not save a checkpoint or modify
model source, the corpus, the split, or any existing experimental artifact.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dataloading.reconstruction_split import load_split_files
from dataloading.vibe_dataset import VIBEDataset, vibe_collate_fn
from model.vibe_model import VIBE_Transformer
from trainer.training_utils import compute_evaluation_metrics


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "config/train_gecv_reconstruction_v1.yaml"
SPLIT = ROOT / "data/manifests/gecv_reconstruction_v1_split_seed20260928.json"
DATA = ROOT / "data/gecv_reconstruction_v1"
CHECKPOINT = ROOT / "checkpoints/gecv_reconstruction_v1_runB/model_best.pth"
OUT_JSON = ROOT / "reports/project_audit/gecv_gamma_scene_scale_impact.json"
OUT_MD = ROOT / "reports/project_audit/gecv_gamma_scene_scale_impact.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SceneScaleWithTimeAxis(nn.Module):
    """In-memory-only adapter for the intended [B, 1, 1] scene scale."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, style_token: torch.Tensor) -> torch.Tensor:
        return self.base(style_token).unsqueeze(-1)


@contextmanager
def corrected_scene_scale(model: VIBE_Transformer):
    original = model.scene_scaler
    model.scene_scaler = SceneScaleWithTimeAxis(original)
    try:
        yield
    finally:
        model.scene_scaler = original


def build_model(config: dict) -> VIBE_Transformer:
    model = VIBE_Transformer(
        input_dim=config["input_dim"], latent_dim=config["latent_dim"],
        num_classes=config["num_classes"], attention_heads=config["attention_heads"],
        transformer_depth=config["transformer_depth"], dropout=config["dropout"],
        drop_path=config["drop_path"],
    ).eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    return model


def forward(model: VIBE_Transformer, batch: dict) -> torch.Tensor:
    return model.forward_with_gamma(
        batch["local_p"], batch["global_v"], batch["env_feat"], batch["audio_seq"], batch["gamma"],
        agent_mask=batch["agent_mask"], local_time_mask=batch["local_time_mask"],
        audio_time_mask=batch["audio_time_mask"], global_time_mask=batch["global_time_mask"],
        reconstruction_mode=True,
    )["logits"]


def evaluate(model: VIBE_Transformer, loader: DataLoader, corrected: bool) -> tuple[dict, torch.Tensor]:
    targets, predictions, logits = [], [], []
    context = corrected_scene_scale(model) if corrected else nullcontext()
    with context, torch.no_grad():
        for batch in loader:
            output = forward(model, batch)
            if not torch.isfinite(output).all():
                raise ValueError("Non-finite logits during impact assessment")
            logits.append(output.cpu())
            predictions.extend(torch.argmax(output, dim=1).cpu().tolist())
            targets.extend(batch["labels"].cpu().tolist())
    return compute_evaluation_metrics(targets, predictions), torch.cat(logits)


def batch_sensitivity(model: VIBE_Transformer, items: list[dict], corrected: bool) -> dict:
    def run(samples: list[dict]) -> torch.Tensor:
        batch = vibe_collate_fn(samples)
        with torch.no_grad():
            return forward(model, batch)

    context = corrected_scene_scale(model) if corrected else nullcontext()
    with context:
        same = run([items[0], items[0]])
        companion = run([items[0], items[1]])
        reordered = run([items[1], items[0]])
    return {
        "same_sample_companion_sensitivity": float((same[0] - companion[0]).abs().max()),
        "order_sensitivity": float((companion[0] - reordered[1]).abs().max()),
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(BASE_CONFIG.read_text())
    files = load_split_files(SPLIT, DATA)["test"]
    dataset = VIBEDataset(files, max_k=8, max_t=32, schema_mode="gecv_reconstruction_v1", reconstruction_gamma_source="stored")
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0, collate_fn=vibe_collate_fn)

    current_model = build_model(config)
    corrected_model = build_model(config)
    current_metrics, current_logits = evaluate(current_model, loader, corrected=False)
    corrected_metrics, corrected_logits = evaluate(corrected_model, loader, corrected=True)
    if current_logits.shape != corrected_logits.shape:
        raise ValueError(f"Unexpected logit-shape mismatch: {tuple(current_logits.shape)} vs {tuple(corrected_logits.shape)}")

    per_sample = (current_logits - corrected_logits).abs().max(dim=1).values
    current_sensitivity = batch_sensitivity(current_model, [dataset[0], dataset[1]], corrected=False)
    corrected_sensitivity = batch_sensitivity(corrected_model, [dataset[0], dataset[1]], corrected=True)
    ids = [path.name.removesuffix("_recon_v1.h5") for path in files]

    report = {
        "protocol": "VIBE-GECV-RECON-v1.0 impact assessment before v1.1 source edit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "checkpoint": {"path": str(CHECKPOINT.relative_to(ROOT)), "sha256": sha256(CHECKPOINT)},
        "source_model": {"path": "model/vibe_model.py", "sha256_pre_fix": sha256(ROOT / "model/vibe_model.py")},
        "inputs": {"config": str(BASE_CONFIG.relative_to(ROOT)), "config_sha256": sha256(BASE_CONFIG), "split": str(SPLIT.relative_to(ROOT)), "split_sha256": sha256(SPLIT), "test_samples": len(dataset), "batch_size": config["batch_size"]},
        "operation": {
            "current_v1_0": "gamma [B,1,1] * scene_scale [B,1] -> [B,B,1]",
            "in_memory_corrected": "gamma [B,1,1] * scene_scale.reshape(B,1,1) -> [B,1,1]",
            "temporal_conditioning": "gamma_prime is passed to GammaGatedTransformer / DecoupledAdaLN",
            "classifier_fusion": "gamma_scalar = gamma_prime.mean(dim=1) is concatenated with h_final",
        },
        "current_v1_0_test_metrics": current_metrics,
        "in_memory_corrected_test_metrics": corrected_metrics,
        "logit_difference_current_vs_corrected": {
            "max_abs_per_sample": [{"source_id": source_id, "max_abs_logit_difference": float(delta)} for source_id, delta in zip(ids, per_sample.tolist())],
            "max": float(per_sample.max()), "mean": float(per_sample.mean()),
        },
        "batch_sensitivity": {"before_correction": current_sensitivity, "after_correction": corrected_sensitivity, "probe_source_ids": ids[:2]},
        "in_memory_only": "The corrected operation was implemented by a temporary scene_scaler wrapper. No corrected checkpoint was saved.",
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    md = [
        "# Gamma/Scene-Scale Broadcasting Impact Assessment",
        "",
        "This is a read-only pre-v1.1 assessment using the existing Run B checkpoint. The corrected operation was evaluated only in memory; no checkpoint, dataset, split, or source model was changed by this script.",
        "",
        "## Operation", "",
        f"- Current v1.0: `{report['operation']['current_v1_0']}`.",
        f"- In-memory correction: `{report['operation']['in_memory_corrected']}`.",
        "- `gamma_prime` conditions both the temporal AdaLN path and the final `[h_final, gamma_scalar]` classifier fusion.",
        "",
        "## Held-out test comparison", "",
        "| Operation | Accuracy | Macro-F1 | Weighted-F1 |", "|---|---:|---:|---:|",
        f"| Current v1.0 | {current_metrics['acc']:.10f} | {current_metrics['macro_f1']:.10f} | {current_metrics['weighted_f1']:.10f} |",
        f"| In-memory corrected shape | {corrected_metrics['acc']:.10f} | {corrected_metrics['macro_f1']:.10f} | {corrected_metrics['weighted_f1']:.10f} |",
        "",
        "## Batch sensitivity", "",
        f"- Before correction, same-sample companion sensitivity: `{current_sensitivity['same_sample_companion_sensitivity']:.12g}`; order sensitivity: `{current_sensitivity['order_sensitivity']:.12g}`.",
        f"- After correction, same-sample companion sensitivity: `{corrected_sensitivity['same_sample_companion_sensitivity']:.12g}`; order sensitivity: `{corrected_sensitivity['order_sensitivity']:.12g}`.",
        f"- Maximum per-sample absolute logit difference between current and corrected shape paths: `{report['logit_difference_current_vs_corrected']['max']:.12g}`.",
        "",
        "The JSON companion records each held-out sample's maximum absolute logit difference and the full metric outputs.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(json.dumps({"status": "PASS", "report": str(OUT_JSON.relative_to(ROOT)), "current": current_metrics, "corrected": corrected_metrics, "sensitivity": report["batch_sensitivity"]}, indent=2))


if __name__ == "__main__":
    main()
