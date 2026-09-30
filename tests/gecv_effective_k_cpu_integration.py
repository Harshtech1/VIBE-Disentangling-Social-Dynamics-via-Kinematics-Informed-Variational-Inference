"""CPU gate for the K=8-storage effective-K adapter.

This test does not alter HDF5 files. It verifies that effective K masks slots,
derives gamma from only active valid synchrony pairs, and blocks inactive-agent
features from changing evaluation logits.
"""
import argparse
import json
from pathlib import Path

import torch

from dataloading.vibe_dataset import VIBEDataset, vibe_collate_fn
from dataloading.reconstruction_split import load_split_files
from model.vibe_model import VIBE_Transformer


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/gecv_reconstruction_v1"
SPLIT = ROOT / "data/manifests/gecv_reconstruction_v1_split_seed20260928.json"
TOLERANCE = {"atol": 1e-5, "rtol": 1e-5}


def _manual_gamma(batch):
    active = batch["agent_mask"][0]
    pairs = batch["synchrony_pair_mask"][0].bool()
    non_self = ~torch.eye(active.numel(), dtype=torch.bool)
    valid = pairs & non_self & active.unsqueeze(1) & active.unsqueeze(0)
    values = batch["sync_mat"][0][valid]
    return values.mean() if values.numel() else torch.zeros(())


def _batch_for(files, effective_k):
    dataset = VIBEDataset(
        files, max_k=8, max_t=32, schema_mode="gecv_reconstruction_v1",
        reconstruction_gamma_source="stored", effective_k=effective_k,
    )
    for index in range(len(dataset)):
        item = dataset[index]
        if item is not None and int(item["agent_mask"].sum()) >= 3:
            return vibe_collate_fn([item])
    raise RuntimeError("No reconstruction sample with at least three active agents")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()
    split_files = load_split_files(SPLIT, DATA_ROOT)
    files = split_files["train"]
    batches = {k: _batch_for(files, k) for k in (3, 5, 8)}

    gamma_report = {}
    for k, batch in batches.items():
        observed = batch["gamma"][0, 0, 0]
        expected = _manual_gamma(batch)
        assert torch.allclose(observed, expected, **TOLERANCE), (k, observed, expected)
        assert batch["gamma_source"] == "derived_from_active_valid_sync_pairs"
        assert not batch["agent_mask"][0, k:].any()
        assert not batch["local_time_mask"][0, k:].any()
        gamma_report[str(k)] = float(observed)
    assert not torch.allclose(batches[3]["gamma"], batches[5]["gamma"]), gamma_report

    torch.manual_seed(20260928)
    batch = batches[3]
    model = VIBE_Transformer(
        input_dim=768, latent_dim=512, num_classes=3, attention_heads=8,
        transformer_depth=2, dropout=0.3, drop_path=0.1,
    ).eval()
    forward = dict(
        global_v=batch["global_v"], env_feat=batch["env_feat"], audio_seq=batch["audio_seq"],
        gamma=batch["gamma"], agent_mask=batch["agent_mask"],
        local_time_mask=batch["local_time_mask"], audio_time_mask=batch["audio_time_mask"],
        global_time_mask=batch["global_time_mask"], reconstruction_mode=True,
    )
    with torch.no_grad():
        logits_before = model.forward_with_gamma(batch["local_p"], **forward)["logits"]
        mutated = batch["local_p"].clone()
        inactive = ~batch["agent_mask"].unsqueeze(-1).unsqueeze(-1)
        mutated = torch.where(inactive, torch.full_like(mutated, 123.0), mutated)
        logits_after = model.forward_with_gamma(mutated, **forward)["logits"]
    assert torch.allclose(logits_before, logits_after, **TOLERANCE)

    report = {
        "status": "PASS",
        "storage_max_k": 8,
        "effective_k_gamma": gamma_report,
        "gamma_source": "derived_from_active_valid_sync_pairs",
        "inactive_agent_logit_invariance": {
            "max_abs_logit_difference": float((logits_before - logits_after).abs().max()),
            "tolerance": TOLERANCE,
        },
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
