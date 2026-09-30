# VIBE--GECV code map

This map is based on the current working tree. **Active V1.1** means a file is
on the documented corrected reconstruction path or validates it. **Historical**
means preserved evidence or an older generic interface; it is not the source of
the V1.1 headline results. No entry below implies that a script should be run
without first reading its command-line help and prerequisites.

| File | Purpose | Called by | Inputs | Outputs | Research role |
|---|---|---|---|---|---|
| `data_preprocessing/reconstruction_contract.py` | Deterministic T=32 binning, K=8 selection, and synchrony helpers. | Reconstruction extractor and validation scripts. | Tracks, feature times, frame geometry. | Padded arrays, masks, pair mask, gamma. | Active V1.1 data-contract source. |
| `data_preprocessing/backbone_compat.py` | Loads HuBERT with the recorded compatibility behaviour. | Reconstruction extractor/validation. | Model ID, device. | HuBERT model and load metadata. | Active extraction support. |
| `scripts/extract_gecv_reconstruction.py` | Resumable GPU HDF5 corpus extractor. | Manual CLI only. | Raw archives, model weights, `ffmpeg`, GPU. | HDF5 records and extraction manifest. | Active reconstruction extractor; expensive. |
| `data_preprocessing/preprocessing_gecv.py` | Older generic GECV preprocessing path. | Manual CLI only. | Legacy GECV layout. | Legacy processed data. | Historical/legacy; not V1.1 corpus authority. |
| `data_preprocessing/preprocessing_vgaf.py` | Older VGAF preprocessing and helper classes. | Legacy inference/preprocessing. | VGAF videos. | Legacy primitives. | Historical; outside GECV V1.1. |
| `data_preprocessing/extract_primitives.py` | Older multi-modal primitive extractor. | Legacy inference/manual CLI. | Generic frames/tubes/audio/text. | Legacy primitive tensors. | Historical; not the V1.1 extractor. |
| `dataloading/reconstruction_split.py` | Builds/validates/resolves the frozen reconstruction split. | Trainer, audits, tests. | HDF5 directory and split manifest. | Three ordered file lists. | Active V1.1 reproducibility-critical. |
| `dataloading/vibe_dataset.py` | HDF5 schema adapter, masks, gamma selection, collator. | Trainer, audits, CPU tests. | One HDF5 path per sample. | Batch dictionary for model. | Active V1.1 reproducibility-critical. |
| `dataloading/__init__.py` | Package marker. | Python importer. | — | Import namespace. | Active infrastructure. |
| `model/vib.py` | Dual affect/environment Gaussian encoders and reparameterization. | `VIBE_Transformer`. | Flattened local feature vectors. | `mu`, `logvar`, affect/env latents. | Active V1.1 model. |
| `model/context_gate.py` | Environment-conditioned affect-token modulation. | `VIBE_Transformer`. | `z_aff`, `z_env`. | Context-modulated affect tokens. | Active V1.1 model. |
| `model/learned_attenttion.py` | Learned-query pooling and visual/audio attention block. | `VIBE_Transformer`. | Visual/audio tokens and masks. | Refined tokens or pooled vector. | Active V1.1 model. |
| `model/gamma_transformer.py` | Decoupled AdaLN and masked gamma-gated temporal transformer. | `VIBE_Transformer`. | `[B,T,L]`, gamma, scene style, time mask. | Temporal latent sequence. | Active V1.1 model. |
| `model/vibe_model.py` | Top-level model assembly and `forward_with_gamma`. | Trainer, tests, audits. | Local/audio/scene/masks/gamma. | Logits, latents, pooling/attention outputs. | Active V1.1 model authority. |
| `model/__init__.py` | Package marker. | Python importer. | — | Import namespace. | Active infrastructure. |
| `losses/loss_function.py` | Hierarchical classification, KL, SAT, and orthogonality losses. | Trainer and CPU tests. | Model output, labels, batch. | Total and component losses. | Active V1.1 objective. |
| `losses/__init__.py` | Package marker. | Python importer. | — | Import namespace. | Active infrastructure. |
| `trainer/training_utils.py` | Canonical metrics and warmup/cosine scheduler. | Trainer, tests, audits. | Predictions/targets or optimizer/epoch. | Metrics or learning rate. | Active V1.1 reproducibility-critical. |
| `trainer/train_model.py` | Training loop, validation selection, test reload, run provenance. | Manual training CLI. | YAML config, HDF5 loaders. | Checkpoints, TensorBoard logs, JSON report. | Active V1.1 trainer authority. |
| `trainer/vizualization.py` | Optional training visualizations. | Trainer when enabled. | Trainer buffers. | Local figures. | Optional active helper; disabled in V1.1 configs. |
| `trainer/__init__.py` | Package marker. | Python importer. | — | Import namespace. | Active infrastructure. |
| `testing/test_model.py` | Generic checkpoint tournament/testing interface. | Manual legacy CLI. | Legacy `data_root/val` files and generic config. | Comparative CSV/plots. | Historical; not compatible with frozen V1.1 evaluation. |
| `testing/inference.py` | Raw-video generic inference wrapper. | Manual legacy CLI. | Video, legacy config, checkpoint. | Prediction/rationale. | Historical; not V1.1 reconstruction authority. |
| `scripts/validate_gecv_reconstruction_v1.py` | Read-only HDF5 validation. | Manual CLI. | Reconstruction HDF5 corpus. | Validation JSON/report. | Active audit utility. |
| `scripts/audit_gecv_dataset_integrity.py` | Read-only sample/split/feature/checkpoint audit. | Manual CLI. | HDF5, extraction/split manifests. | Dataset audit JSON/MD/CSV. | Active audit utility. |
| `scripts/audit_gecv_schema.py` | Schema/H116 forensic audit. | Manual CLI. | HDF5 and manifests. | Schema audit. | Audit-only historical evidence. |
| `scripts/audit_gecv_end_to_end.py` | Read-only end-to-end project audit. | Manual CLI. | Existing code/artifacts. | Project audit reports. | Audit-only; V1.0 findings are historical. |
| `scripts/audit_gamma_scene_scale_impact.py` | Measures the historical V1.0 broadcast issue. | Manual CLI. | V1.0 checkpoint/code path. | Impact report. | Audit-only historical evidence. |
| `scripts/gecv_gpu_smoke_test.py` | One-batch GPU interface smoke test. | Manual CLI. | Frozen HDF5/config. | Smoke-test report. | Active validation utility; no training run. |
| `scripts/phase5c_gecv_3video_diagnostic.py` | Three-video diagnostic extraction. | Manual CLI. | Limited raw media. | Diagnostic outputs. | Historical pilot; never canonical corpus. |
| `scripts/phase5f_videomaev2_sanity.py` | VideoMAEv2 environment sanity check. | Manual CLI. | Runtime/backbone. | Sanity output. | Historical environment check. |
| `scripts/phase6_gecv_reconstruction_validation.py` | Focused reconstruction validation. | Manual CLI. | Existing HDF5/code. | Validation evidence. | Audit/validation support. |
| `scripts/plot_gecv_reconstruction_results.py` | Plots historical V1.0 Run B metrics. | Manual CLI. | Recorded Run B JSON. | V1.0 figures. | Historical reporting only. |
| `scripts/plot_gecv_k_sensitivity.py` | Plots completed V1.0 effective-K records. | Manual CLI. | Historical K reports. | K-sensitivity figures. | Historical reporting only. |
| `scripts/summarize_gecv_3seed_robustness.py` | Summarizes historical V1.0 robustness reports. | Manual CLI. | V1.0 reports. | V1.0 summary. | Historical reporting only. |
| `scripts/build_gecv_v1_1_baseline_package.py` | Builds the authoritative V1.1 3-seed aggregate/figures. | Manual CLI. | V1.1 seed reports/configs/checkpoints. | `reports/final_baseline_v1_1/`. | Active V1.1 reporting utility. |
| `scripts/build_gecv_v1_1_submission_package.py` | Builds curated V1.1 presentation package. | Manual CLI. | V1.1 baseline package. | Submission-style package. | Active presentation utility. |
| `scripts/build_gecv_final_reconstruction_package.py` | Builds an earlier final reconstruction package. | Manual CLI. | Existing reports/assets. | Historical package. | Historical presentation utility. |
| `scripts/build_final_report_assets.py` | Copies report assets for LaTeX compilation. | Final report generator workflow. | Current source/figures. | Report-local assets. | Documentation build support. |
| `scripts/generate_gecv_final_latex_report.py` | Generates the final technical-report source. | Manual CLI. | Recorded reports/assets. | LaTeX report source. | Documentation build support. |
| `tests/gamma_scene_scale_invariance_cpu.py` | Tests V1.1 sample-wise gamma/scene scaling. | Manual CPU test. | Base config, protected checkpoint, synthetic tensors. | PASS/FAIL JSON on stdout. | Active V1.1 regression guard. |
| `tests/gecv_reconstruction_cpu_integration.py` | Reads real HDF5 records through loader/model/loss. | Manual CPU test. | Three real HDF5 records. | Interface-audit report. | Active integration guard. |
| `tests/gecv_reconstruction_pre_gpu_cpu_test.py` | Checks frozen split and one CPU optimizer step. | Manual CPU test. | V1.0 reconstruction config/data. | Pre-GPU report. | Reproducibility guard. |
| `tests/gecv_effective_k_cpu_integration.py` | Checks logical effective K under physical K=8 storage. | Manual CPU test. | Frozen split/HDF5. | PASS/FAIL report. | Historical V1.0 K-sensitivity guard. |
| `tests/ablation_switches_cpu.py` | Checks approved switch contracts on synthetic tensors. | Manual CPU test. | Current model/loss. | PASS/FAIL result. | Infrastructure only; no V1.1 ablation result. |
| `tests/smoke_test.py` | Synthetic generic HDF5/model/loss smoke test. | Manual CPU test. | Temporary synthetic files. | PASS/FAIL stdout. | Legacy-compatible general contract check. |

## Generated and preserved source copies

`reports/final_report/assets/code/` contains report-local snapshots used when
building the PDF. They are documentation assets, not independently executable
source of truth. The authoritative editable files are the paths listed above.
The two `reports/submission/` package directories deliberately duplicate small
reports, source-report copies, and figures so a supervisor can inspect a
standalone package. They are preserved as publication artifacts rather than
deduplicated at the risk of breaking package provenance.

## Configuration map

| Configuration family | Status | Notes |
|---|---|---|
| `train_gecv_reconstruction_v1_1*.yaml` | Current V1.1 | Three-seed corrected baseline; only seed/output paths vary. |
| `train_gecv_reconstruction_v1*.yaml` | Historical V1.0 | Pre-fix baseline/seed/effective-K configurations. |
| `ablations/train_gecv_ablation_*.yaml` | Prepared controls | Switch infrastructure exists; results are not V1.1 ablation results. |
| `reconstruction_gecv_v1.yaml` | Data contract | Reconstruction shape/text-off settings. |
| `train_gecv.yaml`, `test_gecv.yaml`, `train_vgaf.yaml`, `test_vgaf.yaml`, `inference.yaml`, `data_extraction.yaml` | Legacy/generic | Retained for earlier interfaces; not current V1.1 authority. |
