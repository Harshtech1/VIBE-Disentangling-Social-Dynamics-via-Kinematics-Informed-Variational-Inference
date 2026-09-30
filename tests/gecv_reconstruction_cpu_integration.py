#!/usr/bin/env python3
"""CPU-only real-data interface gate for VIBE-GECV-RECON-v1.0.

Reads three completed HDF5 files without modifying them.  It does not extract
features, download models, use CUDA, train a corpus, or save a checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataloading.vibe_dataset import VIBEDataset, vibe_collate_fn
from losses.loss_function import VIBE_Loss
from model.vibe_model import VIBE_Transformer


H5_ROOT = REPO / "data" / "gecv_reconstruction_v1"
REPORT = REPO / "reports" / "gecv_training_interface_audit.json"
SOURCE_IDS = ("videoH001", "videoN001", "videoS001")
TOLERANCE = {"atol": 1e-5, "rtol": 1e-5}


def finite_scalar(value: torch.Tensor) -> float:
    if not torch.isfinite(value).all():
        raise AssertionError("Loss is non-finite")
    return float(value.detach().cpu())


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def model_forward(model: VIBE_Transformer, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return model.forward_with_gamma(
        local_p=batch["local_p"],
        global_v=batch["global_v"],
        env_feat=batch["env_feat"],
        audio_seq=batch["audio_seq"],
        gamma=batch["gamma"],
        agent_mask=batch["agent_mask"],
        local_time_mask=batch["local_time_mask"],
        audio_time_mask=batch["audio_time_mask"],
        global_time_mask=batch["global_time_mask"],
        reconstruction_mode=True,
    )


def main() -> int:
    # Explicitly CPU-only, even if a researcher happens to run this in a GPU terminal.
    device = torch.device("cpu")
    torch.manual_seed(20260928)
    paths = [H5_ROOT / f"{source_id}_recon_v1.h5" for source_id in SOURCE_IDS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Required real HDF5 files are missing: {missing}")

    dataset = VIBEDataset(paths, max_k=8, max_t=32, schema_mode="gecv_reconstruction_v1", reconstruction_gamma_source="stored")
    loader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0, collate_fn=vibe_collate_fn)
    batch = next(iter(loader))
    if batch is None:
        raise AssertionError("Dataset returned no usable samples")
    if batch["env_feat"] is not None:
        raise AssertionError("Reconstruction adapter must not fabricate env_feat")
    if batch["text_anch"] is not None:
        raise AssertionError("Reconstruction text ablation requires text_anch=None")
    if batch["gamma_source"] != "stored_reconstruction_gamma":
        raise AssertionError(f"Unexpected gamma source: {batch['gamma_source']}")

    model = VIBE_Transformer(input_dim=768, latent_dim=512, num_classes=3).to(device)
    criterion = VIBE_Loss(lambda_sat=0.0, lambda_ortho=0.1, enable_sat=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Mask invariance is checked in eval mode because VIB sampling/dropout would
    # otherwise make equivalent inputs non-deterministic.  Only invalid positions
    # are changed; valid observed values remain untouched.
    model.eval()
    with torch.no_grad():
        baseline = model_forward(model, batch)["logits"]
        mutated = clone_batch(batch)
        local_invalid = ~mutated["local_time_mask"]
        audio_invalid = ~mutated["audio_time_mask"]
        local_mutation_count = int(local_invalid.sum())
        audio_mutation_count = int(audio_invalid.sum())
        if local_mutation_count:
            mutated["local_p"][local_invalid] = 17.25
        if audio_mutation_count:
            mutated["audio_seq"][audio_invalid] = -11.5
        masked_logits = model_forward(model, mutated)["logits"]
        max_abs_difference = float((baseline - masked_logits).abs().max())
        mask_invariant = bool(torch.allclose(baseline, masked_logits, **TOLERANCE))
    if not mask_invariant:
        raise AssertionError(f"Masked values changed logits (max abs diff {max_abs_difference})")

    # One real batch, forward/loss/backward/optimizer step only; this is an
    # integration gate, not a training run.
    model.train()
    output = model_forward(model, batch)
    if tuple(output["logits"].shape) != (3, 3):
        raise AssertionError(f"Unexpected logits shape: {tuple(output['logits'].shape)}")
    losses = criterion(output, batch["labels"], batch)
    total, hce, kl, sat, ortho = (finite_scalar(loss) for loss in losses)
    optimizer.zero_grad(set_to_none=True)
    losses[0].backward()
    optimizer.step()

    report = {
        "status": "PASS",
        "scope": "CPU-only real-data Dataset -> collate -> model -> loss -> backward -> one optimizer step",
        "source_ids": list(SOURCE_IDS),
        "device": str(device),
        "hdf5_to_model_mapping": {
            "local_features": "local_p [B,8,32,768]",
            "local_time_mask": "local_time_mask [B,8,32]",
            "agent_mask": "agent_mask [B,8]",
            "audio_features": "audio_seq [B,32,768] via explicit structural compatibility adapter",
            "audio_time_mask": "audio_time_mask [B,32]",
            "global_scene": "global_v [B,768]; one scene-calibration vector, not a temporal sequence",
            "physics_sync": "sync_mat [B,8,8] retained in the batch; the current model consumes gamma rather than sync_mat directly",
            "gamma": "stored extraction scalar -> gamma [B,1,1] for global-adaptive temporal modulation",
            "env_feat": "None; no tensor is fabricated and current VIBE_Transformer does not read this legacy argument",
        },
        "component_trace": {
            "dataset": "VIBEDataset(schema_mode='gecv_reconstruction_v1') detects protocol VIBE-GECV-RECON-v1.0 and applies only the explicit alias adapter.",
            "collate": "vibe_collate_fn batches masks and uses stored HDF5 gamma [B,1,1]; it retains physics_sync as sync_mat.",
            "model": "VIBE_Transformer consumes local_p, global_v, audio_seq, gamma, and masks. env_feat remains a legacy API argument but has no read site in the model.",
            "loss": "VIBE_Loss consumes model output/labels; semantic alignment is disabled because text is absent and enable_sat=False.",
            "physics_sync": "The present model interface does not consume sync_mat directly; its extractor-produced scalar gamma is the synchrony signal passed to the model.",
        },
        "text_status": "OFF; text_anch=None",
        "l_sat_status": "DISABLED; enable_sat=False and lambda_sat=0.0",
        "batch_shapes": {
            "local_p": list(batch["local_p"].shape),
            "local_time_mask": list(batch["local_time_mask"].shape),
            "agent_mask": list(batch["agent_mask"].shape),
            "audio_seq": list(batch["audio_seq"].shape),
            "audio_time_mask": list(batch["audio_time_mask"].shape),
            "global_v": list(batch["global_v"].shape),
            "sync_mat": list(batch["sync_mat"].shape),
            "gamma": list(batch["gamma"].shape),
            "env_feat": None,
            "labels": batch["labels"].tolist(),
        },
        "forward": {"logits_shape": list(output["logits"].shape), "all_logits_finite": bool(torch.isfinite(output["logits"]).all())},
        "losses": {"total": total, "hce": hce, "kl": kl, "sat": sat, "ortho": ortho},
        "backward": "PASS",
        "optimizer_step": "PASS",
        "mask_invariance": {
            "status": "PASS",
            "tolerance": TOLERANCE,
            "mutated_local_masked_positions": local_mutation_count,
            "mutated_audio_masked_positions": audio_mutation_count,
            "max_abs_logit_difference": max_abs_difference,
            "note": "Audio mutation count may be zero when all 32 reconstruction audio bins are valid.",
        },
        "remaining_blockers": [
            "This is VIBE-GECV-RECON-v1.0, not a paper-faithful VIBE claim.",
            "Full training/evaluation remains outside this integration gate.",
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
