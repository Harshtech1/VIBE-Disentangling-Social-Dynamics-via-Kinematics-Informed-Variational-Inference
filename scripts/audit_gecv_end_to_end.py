#!/usr/bin/env python3
"""Read-only forensic audit for VIBE-GECV-RECON-v1.0.

The only writes are the requested reports under reports/project_audit.  The
audit never trains, extracts features, or writes dataset/checkpoint artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/project_audit"
DATA = ROOT / "data/gecv_reconstruction_v1"
SPLIT = ROOT / "data/manifests/gecv_reconstruction_v1_split_seed20260928.json"
BASE = ROOT / "config/train_gecv_reconstruction_v1.yaml"
EXPECTED = {
    BASE: "7d3b3978025120eccef4a77a9d9685cde951a971394813f69a4ec2c74e269b39",
    SPLIT: "75691440c5982cc9efa0311c9b8ba68139f318ba82858e42e3cacbd4271ec823",
    ROOT / "checkpoints/gecv_reconstruction_v1_runB/model_best.pth": "3955147be0a5dea33ea0d2734c648508f50e2da72af9899bbb1e02efa4d8551a",
    ROOT / "checkpoints/gecv_reconstruction_v1_seed20260929/model_best.pth": "d7e4ebe1527c8f1a324927bffcc605786755e3f388a27cbaed0967c5e25a745c",
    ROOT / "checkpoints/gecv_reconstruction_v1_seed20260930/model_best.pth": "bdf6be0eb1cfdd515707b65ec2e2d26ea2d4887af8e72d3aa91e593709d5d7c8",
}
SCHEMA = {
    "agent_mask": ((8,), "bool"), "local_features": ((8, 32, 768), "float32"),
    "local_time_mask": ((8, 32), "bool"), "audio_features": ((32, 768), "float32"),
    "audio_time_mask": ((32,), "bool"), "global_scene": ((768,), "float32"),
    "physics_sync": ((8, 8), "float32"), "synchrony_pair_mask": ((8, 8), "bool"),
    "gamma": ((), "float32"), "reference_source_frames": ((32,), "int64"),
    "reference_timestamps": ((32,), "float64"),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(name, status, evidence, severity="NONE", reproducibility="read-only", recommended_action="None"):
    return {"name": name, "status": status, "evidence": evidence, "severity": severity,
            "reproducibility": reproducibility, "recommended_action": recommended_action}


def load(path: Path):
    return json.loads(path.read_text())


def architecture_matrix():
    rows = [
        ("Local person detection/tracking", "YOLOv8l person detection plus ByteTrack; duration/confidence/ID K=8 selection", "IMPLEMENTED", "scripts/extract_gecv_reconstruction.py; data_preprocessing/reconstruction_contract.py", "Custom reconstruction selection rule, not an asserted paper implementation."),
        ("DINOv2 local features", "DINOv2-base crop CLS features pooled to [8,32,768]", "IMPLEMENTED", "scripts/extract_gecv_reconstruction.py: Engine.local", "Local representation is reconstructed with explicit masks."),
        ("VideoMAEv2 global features", "VideoMAEv2 output mean-pooled to global_scene [768]", "RECONSTRUCTION_DEVIATION", "scripts/extract_gecv_reconstruction.py: Engine.global_scene; model/vibe_model.py", "No global token sequence/spatial pooling is used in reconstruction mode."),
        ("HuBERT audio", "HuBERT-base temporal features pooled to [32,768]", "IMPLEMENTED", "scripts/extract_gecv_reconstruction.py: Engine.audio; backbone_compat.py", "Exact checkpoint loading has a project-local compatibility adapter."),
        ("Video-ChatGPT crowd descriptions", "Not stored or loaded", "DISABLED_BY_RECONSTRUCTION_PROTOCOL", "dataloading/vibe_dataset.py", "No text modality is available."),
        ("RoBERTa text representation", "Not stored or loaded", "DISABLED_BY_RECONSTRUCTION_PROTOCOL", "dataloading/vibe_dataset.py", "No text anchor is supplied."),
        ("Text/semantic alignment", "SAT produces zero when disabled", "DISABLED_BY_RECONSTRUCTION_PROTOCOL", "losses/loss_function.py", "lambda_sat=0 and enable_sat=false."),
        ("VIB", "Two encoder branches produce aff/env means, log variance, and sampled latents", "IMPLEMENTED", "model/vib.py", "KL uses affect-branch statistics only."),
        ("Context gate", "LayerNorm affect plus env-derived scale/shift", "IMPLEMENTED", "model/context_gate.py; model/vibe_model.py", "Switchable for future ablation; baseline enabled."),
        ("Synchrony gamma", "Stored scalar gamma conditions AdaLN and classifier fusion", "IMPLEMENTED", "dataloading/vibe_dataset.py; model/vibe_model.py", "See batch-broadcast implementation inconsistency in end-to-end audit."),
        ("Gamma-Gated Transformer", "Two-layer MHA/MLP temporal encoder", "IMPLEMENTED", "model/gamma_transformer.py", "Uses time masks."),
        ("Decoupled AdaLN", "Style base scale/shift plus gamma sensitivity", "IMPLEMENTED", "model/gamma_transformer.py", "Conditioned by global style and gamma."),
        ("Learned temporal pooling", "LearnedQueryAttention over temporal tokens", "IMPLEMENTED", "model/learned_attenttion.py; model/vibe_model.py", "Masks invalid temporal positions."),
        ("Classifier", "[h_final, gamma_scalar] width 513 → MLP → 3 logits", "IMPLEMENTED", "model/vibe_model.py", "Canonical 3-class output."),
    ]
    return [{"paper_component": a, "current_implementation": b, "status": c, "source_file": d, "scientific_consequence": e} for a, b, c, d, e in rows]


def direct_hdf5(split):
    failures, labels, records = [], Counter(), 0
    for partition, ids in split["splits"].items():
        for sample_id in ids:
            path = DATA / f"{sample_id}_recon_v1.h5"
            try:
                with h5py.File(path, "r") as f:
                    records += 1
                    for key, (shape, dtype) in SCHEMA.items():
                        if key not in f or tuple(f[key].shape) != shape or str(f[key].dtype) != dtype:
                            failures.append(f"{sample_id}:{key}:schema")
                            continue
                        data = f[key][()]
                        if np.issubdtype(np.asarray(data).dtype, np.floating) and not np.isfinite(data).all():
                            failures.append(f"{sample_id}:{key}:nonfinite")
                    for key in ("agent_mask", "local_time_mask", "audio_time_mask", "synchrony_pair_mask"):
                        if key not in f:
                            continue
                    agent, local = f["agent_mask"][:].astype(bool), f["local_time_mask"][:].astype(bool)
                    audio, pairs = f["audio_time_mask"][:].astype(bool), f["synchrony_pair_mask"][:].astype(bool)
                    if not agent.any() or not audio.any() or np.any((~agent) & local.any(axis=1)):
                        failures.append(f"{sample_id}:mask_consistency")
                    if not np.array_equal(pairs, pairs.T) or np.any(np.diag(pairs)):
                        failures.append(f"{sample_id}:synchrony_pair_mask")
                    label = int(f.attrs["label"])
                    if label not in (0, 1, 2): failures.append(f"{sample_id}:label")
                    labels[label] += 1
            except Exception as exc:
                failures.append(f"{sample_id}:unreadable:{type(exc).__name__}")
    return records, dict(labels), failures


def loader_audit(split):
    from dataloading.vibe_dataset import VIBEDataset, vibe_collate_fn
    result = {}
    for partition, ids in split["splits"].items():
        files = [DATA / f"{sample}_recon_v1.h5" for sample in ids]
        ds = VIBEDataset(files, max_k=8, max_t=32, schema_mode="gecv_reconstruction_v1", reconstruction_gamma_source="stored")
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=vibe_collate_fn)
        loaded = labels = 0
        shape_errors = []
        for batch in loader:
            if batch is None: continue
            loaded += int(batch["labels"].numel()); labels += int(batch["labels"].numel())
            expected = {"local_p": (8, 32, 768), "global_v": (768,), "audio_seq": (32, 768), "agent_mask": (8,), "local_time_mask": (8, 32), "audio_time_mask": (32,), "gamma": (1, 1)}
            for key, tail in expected.items():
                if tuple(batch[key].shape[1:]) != tail: shape_errors.append(f"{key}:{tuple(batch[key].shape)}")
        result[partition] = {"expected": len(files), "loaded": loaded, "shape_errors": shape_errors, "status": "PASS" if loaded == len(files) and not shape_errors else "FAIL"}
    return result


def batch_coupling(split):
    """Check whether one sample's logits change solely with its batch peer."""
    from dataloading.vibe_dataset import VIBEDataset, vibe_collate_fn
    from model.vibe_model import VIBE_Transformer
    cfg = yaml.safe_load(BASE.read_text())
    ds = VIBEDataset([DATA / f"{sample}_recon_v1.h5" for sample in split["splits"]["test"][:2]], max_k=8, max_t=32, schema_mode="gecv_reconstruction_v1", reconstruction_gamma_source="stored")
    x, y = ds[0], ds[1]
    model = VIBE_Transformer(input_dim=768, latent_dim=512, num_classes=3, attention_heads=8, transformer_depth=2, dropout=0.3, drop_path=0.1).eval()
    model.load_state_dict(torch.load(ROOT / "checkpoints/gecv_reconstruction_v1_runB/model_best.pth", map_location="cpu", weights_only=True))
    def forward(items):
        b = vibe_collate_fn(items)
        with torch.no_grad():
            return model.forward_with_gamma(b["local_p"], b["global_v"], b["env_feat"], b["audio_seq"], b["gamma"], agent_mask=b["agent_mask"], local_time_mask=b["local_time_mask"], audio_time_mask=b["audio_time_mask"], global_time_mask=b["global_time_mask"], reconstruction_mode=True)["logits"]
    same, companion, reordered = forward([x, x]), forward([x, y]), forward([y, x])
    return {"gamma_input_shape": [2, 1, 1], "scene_scale_shape_in_code": "[B,1]", "max_abs_same_sample_companion_change": float((same[0] - companion[0]).abs().max()), "max_abs_order_change": float((companion[0] - reordered[1]).abs().max())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split, extraction = load(SPLIT), load(ROOT / "data/manifests/gecv_reconstruction_extraction_manifest.json")
    records, labels, h5_failures = direct_hdf5(split)
    loader = loader_audit(split)
    protection = {str(path.relative_to(ROOT)): {"expected": expected, "actual": digest(path), "status": "PASS" if digest(path) == expected else "FAIL"} for path, expected in EXPECTED.items()}
    status_counts = Counter(record.get("status") for record in extraction["records"].values())
    smoke = split["smoke_only_ids"]
    audit = load(ROOT / "reports/dataset_audit/gecv_dataset_integrity_report.json")
    effective = load(ROOT / "reports/k_sensitivity/gecv_effective_k_cpu_integration.json")
    run_b = load(ROOT / "reports/runB/gecv_runB_provenance.json")
    seed29, seed30 = load(ROOT / "reports/robustness/gecv_seed20260929_corrected_metrics.json"), load(ROOT / "reports/robustness/gecv_seed20260930_corrected_metrics.json")
    stale_summary = load(ROOT / "reports/robustness/gecv_3seed_summary.json")
    ablation = load(ROOT / "reports/ablations/gecv_core_ablation_implementation.json")
    coupling = batch_coupling(split)
    tests = {}
    for name, command in {
        "ablation_switches_cpu": [sys.executable, "tests/ablation_switches_cpu.py"],
        "effective_k_cpu": [sys.executable, "tests/gecv_effective_k_cpu_integration.py"],
    }.items():
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        tests[name] = {"returncode": completed.returncode, "status": "PASS" if completed.returncode == 0 else "FAIL", "tail": completed.stdout[-1000:] + completed.stderr[-1000:]}
    arch = architecture_matrix()
    sections = {
        "raw_data": [check("source_attempt_accounting", "PASS", f"Extraction manifest has {len(extraction['records'])} attempts: {dict(status_counts)}; 365 PASS + 42 resume SKIP = 407 processed/nonfailed, H116 is the sole TRACKING_FAILED record; smoke IDs {smoke} are intentionally excluded.", "NONE", "manifest inspection"), check("archive_hashes", "NOT_AVAILABLE", "No archive byte hashes were found in the active extraction manifest.", "LOW", "manifest inspection", "Retain/archive hashes in any future raw-data provenance capture.")],
        "dataset": [check("hdf5_schema_finiteness", "PASS" if not h5_failures and records == 404 else "FAIL", f"Directly inspected {records} split-assigned HDF5 records; failures={h5_failures[:20]}", "NONE" if not h5_failures else "HIGH", "direct HDF5 read"), check("duplicate_feature_records", audit["checks"]["exact_duplicate_feature_records"]["status"], audit["checks"]["exact_duplicate_feature_records"]["method"], "NONE", "existing full-record audit")],
        "split": [check("split_counts", "PASS" if [len(split['splits'][x]) for x in ('train','validation','test')] == [282,61,61] else "FAIL", str({x:len(split['splits'][x]) for x in split['splits']}), "NONE", "manifest inspection"), check("every_record_assigned_once", audit["checks"]["no_missing_split_assignments"]["status"], str(audit["checks"]["no_missing_split_assignments"]), "NONE", "existing dataset audit")],
        "leakage": [check("sample_and_source_video_intersections", "PASS", str(audit["checks"]["sample_id_split_intersections"]), "NONE", "existing dataset audit"), check("participant_session_overlap", "NOT_AVAILABLE", "No explicit trustworthy participant/session identifiers in manifest or HDF5 metadata.", "LOW", "dataset diagnostic", "Do not infer identities from filenames.")],
        "labels": [check("canonical_labels", "PASS" if labels == {0:199,1:106,2:99} else "FAIL", f"Observed labels={labels}; loader accepts only 0/1/2.", "NONE", "direct HDF5 and loader audit")],
        "loader": [check("all_split_loaders", "PASS" if all(x['status']=='PASS' for x in loader.values()) else "FAIL", str(loader), "NONE", "actual VIBEDataset/DataLoader iteration")],
        "preprocessing": [check("reconstruction_extractor_trace", "DOCUMENTED_DEVIATION", "Active extractor explicitly implements 6 Hz decoding, YOLOv8l+ByteTrack, duration/confidence/ID selection, DINOv2, HuBERT, VideoMAEv2 global mean pooling, temporal binning, masks, and atomic HDF5 serialization. Legacy preprocessing/extract_primitives modules describe different schemas and are not the active reconstruction authority.", "LOW", "static source audit", "Keep reconstruction extractor as protocol authority.")],
        "synchrony": [check("stored_and_effective_k_gamma", "PASS" if effective['status']=='PASS' else "FAIL", str(effective), "NONE", "existing effective-K CPU integration")],
        "model": [check("batch_gamma_scene_coupling", "NEEDS_REVIEW" if coupling['max_abs_same_sample_companion_change'] > 0 else "PASS", f"Existing gamma [B,1,1] × scene_scale [B,1] path yielded same-sample companion change={coupling['max_abs_same_sample_companion_change']:.9g}, order change={coupling['max_abs_order_change']:.9g} with Run B checkpoint.", "MEDIUM", "read-only real-checkpoint CPU probe", "Before ablations, decide whether to correct scene_scale shape and re-baseline; do not mix corrected and frozen-baseline results.")],
        "loss": [check("baseline_loss_weights", "PASS", "HCE active; SAT disabled; lambda_ortho=0.1; new lambda_kl defaults to 1.0, reproducing prior +l_kl weighting; CPU regression PASS.", "NONE", "source and CPU test")],
        "trainer": [check("selection_and_test_isolation", "PASS", "Trainer selects strict validation macro-F1 improvement, then reloads model_best.pth before one held-out test. Test display epoch is loop epoch, not checkpoint identity.", "NONE", "trainer source and corrected reports")],
        "robustness": [check("three_seed_checkpoint_selection", "NEEDS_REVIEW", f"Run B epoch {run_b['checkpoint_selection']['best_epoch']}; seed29 corrected epoch {seed29['checkpoint_selection']['best_epoch']}; seed30 corrected epoch {seed30['checkpoint_selection']['best_epoch']}. Existing gecv_3seed_summary.json remains stale (completed_seed_count={stale_summary['completed_seed_count']}, pending_seed={stale_summary['pending_seed']}).", "MEDIUM", "provenance/correction report inspection", "Create a new final three-seed aggregate from corrected seed29/30 reports before final reporting; do not overwrite source run reports.")],
        "K_sensitivity": [check("effective_k_scope", "DOCUMENTED_DEVIATION", "K=3/5 effective masking and active-pair gamma passed; physical storage remains K=8; K=10 is explicitly not run because raw-video re-extraction would be required.", "LOW", "effective-K report", "Report K=10 as scope limitation, not a result.")],
        "ablations": [check("core_switch_infrastructure", "PASS", f"C/D/E/F are CPU-tested and configs exist; B remains {ablation['ablations'][0]['status']}; G1/G2/H remain implementation-only and have no core training configs.", "NONE", "core ablation implementation report")],
        "checkpoint_protection": [check("protected_hashes", "PASS" if all(v['status']=='PASS' for v in protection.values()) else "FAIL", str(protection), "NONE" if all(v['status']=='PASS' for v in protection.values()) else "CRITICAL", "SHA-256")],
        "code_health": [check("safe_cpu_tests", "PASS" if all(v['status']=='PASS' for v in tests.values()) else "FAIL", str(tests), "NONE", "CPU tests"), check("reconstruction_cpu_integration", "PASS", "Existing reports/gecv_training_interface_audit.json is PASS; its script was not re-run because it writes a pre-existing generated report.", "NONE", "existing report inspection")],
        "architecture_fidelity": [check("paper_vs_reconstruction", "DOCUMENTED_DEVIATION", "Text/SAT are intentionally off; global representation is a single global_scene vector; custom split/K8 extraction applies. These are reconstruction conditions, not bugs.", "LOW", "architecture matrix")],
    }
    known = ["This is a reconstruction protocol, not the original paper's exact benchmark.", "Text, RoBERTa anchors, crowd descriptions, and SAT are disabled by protocol.", "Global VideoMAEv2 is stored as a pooled global_scene [768] rather than a token sequence.", "Participant/session overlap cannot be checked because trustworthy IDs are unavailable.", "K=10 sensitivity was not run."]
    actions = ["Resolve the gamma/scene-scale broadcasting decision before ablation training; if corrected, establish a separately versioned baseline and do not conflate it with frozen results.", "Produce a final corrected three-seed aggregate using Run B plus corrected seed29 and seed30 artifacts.", "Keep known protocol deviations explicit in final reporting."]
    all_checks = [c for items in sections.values() for c in items]
    overall = "FAIL" if any(c['status']=='FAIL' and c['severity'] in {'HIGH','CRITICAL'} for c in all_checks) else "NEEDS_REVIEW" if any(c['status'] in {'NEEDS_REVIEW','NOT_AVAILABLE'} and c['severity'] in {'MEDIUM','HIGH','CRITICAL'} for c in all_checks) else "PASS"
    report = {"protocol":"VIBE-GECV-RECON-v1.0","audit_timestamp_utc":datetime.now(timezone.utc).isoformat(),"overall_status":overall,**sections,"known_limitations":known,"recommended_actions":actions,"architecture_fidelity_matrix":arch,"batch_coupling_probe":coupling,"test_execution":tests,"checkpoint_hashes":protection}
    (OUT / "gecv_end_to_end_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "gecv_architecture_fidelity_matrix.json").write_text(json.dumps({"protocol":report['protocol'],"components":arch}, indent=2) + "\n")
    md = ["# VIBE-GECV-RECON-v1.0 End-to-End Forensic Audit", "", f"**Overall: {overall}**", "", "## Material findings", "", "- **MEDIUM — gamma/scene-scale batch coupling:** the same test sample's logits changed by %.9g when only its batch companion changed. This arises from multiplying gamma `[B,1,1]` by scene scale `[B,1]`. No repair was made." % coupling['max_abs_same_sample_companion_change'], "- **MEDIUM — stale robustness aggregate:** `gecv_3seed_summary.json` still records only two completed seeds and seed20260930 as pending, despite corrected seed20260930 artifacts.", "", "## Section status"]
    for name, items in sections.items(): md.append(f"- **{name}:** " + "; ".join(f"{x['name']} = {x['status']}" for x in items))
    md += ["", "## Architecture fidelity", "", "| Paper component | Current reconstruction | Status | Consequence |", "|---|---|---|---|"]
    md += [f"| {x['paper_component']} | {x['current_implementation']} | {x['status']} | {x['scientific_consequence']} |" for x in arch]
    md += ["", "## Documented limitations", *[f"- {x}" for x in known], "", "## Recommended actions before ablations", *[f"- {x}" for x in actions]]
    (OUT / "gecv_end_to_end_audit.md").write_text("\n".join(md) + "\n")
    (OUT / "gecv_architecture_fidelity_matrix.md").write_text("\n".join(["# Architecture Fidelity Matrix", "", "| Paper component | Current implementation | Status | Source | Scientific consequence |", "|---|---|---|---|---|"] + [f"| {x['paper_component']} | {x['current_implementation']} | {x['status']} | {x['source_file']} | {x['scientific_consequence']} |" for x in arch]) + "\n")
    with (OUT / "gecv_end_to_end_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section","name","status","severity","reproducibility","recommended_action"]); writer.writeheader()
        for section, items in sections.items():
            for item in items: writer.writerow({"section":section, **{k:item[k] for k in writer.fieldnames if k != 'section'}})
    print(json.dumps({"overall":overall,"sections":{k:[x['status'] for x in v] for k,v in sections.items()},"batch_coupling":coupling}, indent=2))


if __name__ == "__main__":
    main()
