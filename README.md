# VIBE--GECV reconstruction

This repository contains an audited reconstruction of the GECV component of
**VIBE: Disentangling Social Dynamics via Kinematics-Informed Variational
Inference for Behavioral Emotion**. The current scientific reference is
**VIBE-GECV-RECON-v1.1**, not an exact reproduction of the original published
benchmark.

## Project

The reconstruction transforms available GECV video records into fixed-shape
multimodal HDF5 samples, trains a VIBE-style classifier, selects a checkpoint
using validation macro-F1 only, and evaluates the selected checkpoint once on
a frozen held-out test set. All current results, hashes, and limitations are
recorded in [`reports/final_baseline_v1_1/`](reports/final_baseline_v1_1/).

## VIBE paper and GECV scope

The original study describes a broader VIBE benchmark. This project reconstructs
only the available GECV path and explicitly does **not** claim exact paper
protocol reproduction. In particular, the current corpus/split differ from the
paper description, text/SAT is disabled, global visual features are stored as a
pooled scene vector, and participant/session identifiers are unavailable.

## Dataset

- Readable raw-video inventory: 408 records.
- Experimental HDF5 corpus: 404 records; `videoH116` is documented as a
  tracking failure and three smoke-only records are excluded from the split.
- Frozen split manifest:
  [`data/manifests/gecv_reconstruction_v1_split_seed20260928.json`](data/manifests/gecv_reconstruction_v1_split_seed20260928.json)
- Split seed: `20260928`; train / validation / test = **282 / 61 / 61**.
- Labels: Positive = 0, Neutral = 1, Negative = 2.
- Storage contract: physical Kmax = 8, T = 32, feature width = 768.

The split SHA-256 is
`75691440c5982cc9efa0311c9b8ba68139f318ba82858e42e3cacbd4271ec823`.
The source-corpus hash is
`d84c8fed0f72de4b9a485e3502d41c107a2b9c0c89ff000f5b880177448854b9`.

## Pipeline

```text
raw GECV video
  -> 6 Hz decoding -> YOLOv8l + ByteTrack -> deterministic K=8 selection
  -> DINOv2 local features / VideoMAEv2 scene feature / HuBERT audio feature
  -> T=32 temporal bins + masks + kinematic synchrony gamma
  -> HDF5 record -> frozen split loader -> dual-stream VIB
  -> context gate -> visual/audio cross-modal attention
  -> gamma-gated transformer -> learned-query temporal pooling
  -> three-class classifier -> validation-selected checkpoint -> held-out test
```

[`docs/EXECUTION_FLOW.md`](docs/EXECUTION_FLOW.md) traces the actual functions,
tensor shapes, configurations, and outputs. [`docs/CODE_MAP.md`](docs/CODE_MAP.md)
is the quick file-level map.

## Repository structure

| Path | Role |
|---|---|
| `data_preprocessing/` | reconstruction extraction contract and legacy preprocessing |
| `dataloading/` | HDF5 adapter, masks, and deterministic split resolver |
| `model/` | VIB, context, attention, gamma-conditioned transformer, classifier |
| `losses/` | hierarchical classification, KL, orthogonality, optional SAT losses |
| `trainer/` | training, validation-only selection, provenance, metrics |
| `testing/` | legacy generic inference/testing interfaces; not the V1.1 authority |
| `tests/` | CPU smoke, integration, effective-K, ablation-switch, and gamma tests |
| `config/` | V1.1, historical V1.0, effective-K, and ablation configurations |
| `scripts/` | extraction, validation/audit, plotting, and report/package utilities |
| `reports/` | recorded results, audits, final packages, figures, and provenance |
| `docs/` | data contract, code map, execution flow, supervisor walkthrough |

## Main entry points

- Reconstruction extractor (expensive; do not run unintentionally):
  `scripts/extract_gecv_reconstruction.py`
- Active trainer: `trainer/train_model.py`
- Frozen split resolver: `dataloading/reconstruction_split.py`
- Dataset adapter: `dataloading/vibe_dataset.py`
- Current model: `model/vibe_model.py`
- V1.1 aggregate builder:
  `scripts/build_gecv_v1_1_baseline_package.py`
- Read-only dataset audit: `scripts/audit_gecv_dataset_integrity.py`
- V1.1 gamma regression: `tests/gamma_scene_scale_invariance_cpu.py`

`testing/test_model.py` and `testing/inference.py` support an older generic
interface and do not consume the frozen reconstruction split/model contract.
They are retained as historical code and must not be used to claim V1.1 test
metrics. The V1.1 authoritative evaluation is the post-selection test inside
`VIBETrainer.run()` and the recorded per-seed reports.

## Configuration

The active base configuration is
[`config/train_gecv_reconstruction_v1_1.yaml`](config/train_gecv_reconstruction_v1_1.yaml).
The seed-specific V1.1 configurations are
[`config/train_gecv_reconstruction_v1_1_seed20260929.yaml`](config/train_gecv_reconstruction_v1_1_seed20260929.yaml)
and
[`config/train_gecv_reconstruction_v1_1_seed20260930.yaml`](config/train_gecv_reconstruction_v1_1_seed20260930.yaml).

All three use the same corpus, frozen split, Kmax=8, T=32, input dimension 768,
latent dimension 512, 8 attention heads, depth 2, dropout 0.3, AdamW,
`lr=1e-4`, weight decay `1e-3`, 3-epoch warmup/cosine schedule, 30 epochs,
mixup 0.2, gradient clipping 1.0, and `lambda_ortho=0.1`. Seed-specific files
change only seed and isolated output locations.

## Training

Training is not run by any audit or reporting command. The recorded entry point
is below; use an isolated log/checkpoint/report location for each seed.

```bash
PYTHONPATH="$PWD" python trainer/train_model.py \
  --config config/train_gecv_reconstruction_v1_1.yaml \
  --log-dir logs/gecv_reconstruction_v1_1_seed20260928 \
  --checkpoint-dir checkpoints/gecv_reconstruction_v1_1_seed20260928 \
  --run-report reports/robustness/gecv_v1_1_seed20260928_metrics.json
```

During a run, validation macro-F1 selects `model_best.pth`; after all 30
epochs, the trainer reloads that checkpoint before evaluating the held-out test
set. Test metrics do not select checkpoints.

## Evaluation

Authoritative V1.1 results are already recorded in
[`reports/final_baseline_v1_1/gecv_v1_1_three_seed_aggregate.json`](reports/final_baseline_v1_1/gecv_v1_1_three_seed_aggregate.json).
The three completed seeds use checkpoint epochs 10, 12, and 14. The descriptive
aggregate is:

| Metric | Mean ± sample SD across three seeds |
|---|---:|
| Held-out accuracy | 95.08 ± 0.00% |
| Held-out macro-F1 | 0.937 ± 0.002 |
| Held-out weighted-F1 | 0.951 ± 0.000 |

These values are measurements on one frozen test split. They are not claims of
statistical significance, participant-level generalization, or direct
comparability with the original paper benchmark.

## Tests

Run non-training CPU checks from the repository root:

```bash
PYTHONPATH="$PWD" python tests/gamma_scene_scale_invariance_cpu.py
PYTHONPATH="$PWD" python tests/gecv_reconstruction_cpu_integration.py --report /tmp/gecv_cpu_integration.json
PYTHONPATH="$PWD" python tests/gecv_reconstruction_pre_gpu_cpu_test.py --report /tmp/gecv_pre_gpu_cpu.json
PYTHONPATH="$PWD" python tests/gecv_effective_k_cpu_integration.py
PYTHONPATH="$PWD" python tests/ablation_switches_cpu.py
PYTHONPATH="$PWD" python scripts/validate_gecv_reconstruction_v1.py --report /tmp/gecv_schema_validation.json
```

The gamma test is a V1.1 regression guard: `gamma` and `scene_scale` are each
`[B,1,1]`, preventing historical cross-sample broadcasting.

## Known deviations and limitations

- V1.1 corrects a reconstruction implementation-level gamma/scene-scale
  broadcasting issue. V1.0 is preserved as historical evidence; V1.0 and V1.1
  results must never be pooled.
- This is a 404-record reconstruction with a frozen 282/61/61 split, not the
  paper's exact population or split.
- Text descriptions, RoBERTa anchors, and SAT are disabled by the approved
  reconstruction protocol.
- The global video feature is stored as pooled `global_scene [768]`, rather
  than a global temporal-token representation.
- The current KL term uses the affect branch only; this is recorded as a
  reconstruction fidelity deviation.
- Participant/session overlap cannot be checked because trustworthy identifiers
  are not stored.
- Corrected V1.1 K=3/K=5 and V1.1 ablation training remain future work; K=10
  requires raw-video re-extraction because physical storage is Kmax=8.

## Reproducibility

Install the declared Python packages with `pip install -r requirements.txt`.
The GPU extractor additionally requires raw archives, `ffmpeg`, YOLO weights,
the listed pretrained backbones, and CUDA; it intentionally fails early if
these prerequisites are absent. Audits/tests use existing HDF5/checkpoint
artifacts and do not train or re-extract data.

Start with [`docs/SUPERVISOR_CODE_WALKTHROUGH.md`](docs/SUPERVISOR_CODE_WALKTHROUGH.md),
then inspect the package under
[`reports/submission/gecv_vibe_final_reconstruction_package/`](reports/submission/gecv_vibe_final_reconstruction_package/).

## Citation

Please cite the VIBE paper and clearly describe this repository as a
VIBE--GECV reconstruction. The exact protocol, source paths, and limitations
are documented in the final package and its provenance manifests.
