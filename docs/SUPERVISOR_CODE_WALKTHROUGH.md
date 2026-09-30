# Supervisor code walkthrough

This is a reading order for a supervisor who wants to inspect the actual
VIBE--GECV reconstruction without relying on a high-level claim. It emphasizes
the current corrected V1.1 path and identifies historical code explicitly.

## Before reading code: establish the protocol

1. Read the active configuration:
   [`config/train_gecv_reconstruction_v1_1.yaml`](../config/train_gecv_reconstruction_v1_1.yaml).
2. Inspect the frozen split:
   [`data/manifests/gecv_reconstruction_v1_split_seed20260928.json`](../data/manifests/gecv_reconstruction_v1_split_seed20260928.json).
3. Read the V1.1 aggregate and integrity audit in
   [`reports/final_baseline_v1_1/`](../reports/final_baseline_v1_1/).

The intended experiment is 404 HDF5 records, a frozen 282/61/61 split,
Positive/Neutral/Negative labels 0/1/2, Kmax=8, T=32, and three V1.1 seeds.
It is an audited reconstruction, not the original paper benchmark.

## 1. Preprocessing and feature storage

Start with `data_preprocessing/reconstruction_contract.py`, then read
`scripts/extract_gecv_reconstruction.py`.

- `shared_time_grid()` constructs 32 source-time targets from a decoded clip.
- `bin_pool()` puts every source observation into exactly one midpoint-defined
  temporal bin. It does not invent an observation for an empty bin; the mask
  says that bin is invalid.
- `select_k8_duration_confidence_id()` makes the K=8 rule explicit: duration,
  then detector confidence, then numerical track ID. This is a documented
  reconstruction convention.
- `shared_interval_synchrony()` forms normalized centroid velocities for common
  consecutive sampled-frame intervals. Its pairwise cosine means form
  `physics_sync`; only valid non-self pairs contribute to `gamma`.

The extraction script performs 6-Hz decoding, YOLOv8l/ByteTrack tracking,
DINOv2 local crops, HuBERT audio, VideoMAEv2 scene extraction, and atomic HDF5
writing. It checks shapes/finite values before accepting a record. It should be
read and run only when raw extraction is authorized; it is not part of normal
training or evaluation.

The HDF5 model-facing fields are `local_features [8,32,768]`, masks,
`audio_features [32,768]`, `global_scene [768]`, `physics_sync [8,8]`,
`synchrony_pair_mask [8,8]`, and scalar `gamma`. Variable-length
`selected_agent_ids` is provenance metadata, not model input.

## 2. Dataset and frozen partition

Read `dataloading/reconstruction_split.py` before `dataloading/vibe_dataset.py`.

`reconstruction_split.py` does two important jobs. It can construct the
deterministic manifest, and `load_split_files()` verifies the current corpus
hash and membership before it returns the explicit train/validation/test file
lists. This prevents a silent reshuffle or a changed corpus from being used
with the frozen manifest.

`VIBEDataset.__getitem__()` opens one HDF5 record and validates the active
schema. `vibe_collate_fn()` stacks records. Notice the two gamma paths:

- The normal reconstruction path uses the stored scalar gamma.
- Effective-K tests derive gamma from the currently active valid synchrony
  pairs, not from zero-filled padding.

Masks, not just zero values, determine which agents and times are valid. The
loader requires labels in `{0,1,2}` and preserves the canonical mapping.

## 3. Model architecture

Read `model/vibe_model.py` top-to-bottom, following its imports in this order:

1. `model/vib.py` contains `CausalVIB`. For each local 768-D feature it predicts
   affective and environmental Gaussian parameters and samples during training.
   At evaluation it uses the mean, making predictions deterministic given the
   checkpoint and input.
2. `model/context_gate.py` contains `ContextInjectionGate`. The environment
   latent predicts an elementwise scale/shift for a normalized affective latent.
3. `model/learned_attenttion.py` contains two mechanisms: visual queries attend
   to projected audio and then self-attend; a learned query pools time to one
   clip vector.
4. `model/gamma_transformer.py` contains `DecoupledAdaLN` and the masked
   `GammaGatedTransformer`. Scene style supplies baseline shift/scale and gamma
   supplies a separate sensitivity term.

In `forward_with_gamma()`, local input is `[B,8,32,768]`, audio is
`[B,32,768]`, and scene is `[B,768]`. VIB produces `[B,8,32,512]` latents.
After cross-modal processing, valid agents are pooled at each time to
`[B,32,512]`. The transformer output is pooled to `h_final [B,512]`; a scalar
gamma summary is appended before the classifier, producing logits `[B,3]`.

### V1.1 correction to inspect closely

At `model/vibe_model.py` lines around 129--144, the model computes:

```python
scene_scale = self.scene_scaler(style_token).unsqueeze(-1)  # [B,1,1]
gamma_prime = gamma * scene_scale                           # [B,1,1]
```

Historical V1.0 used a `[B,1]` scene-scale shape with gamma `[B,1,1]`. PyTorch
can then broadcast across samples. The V1.1 extra singleton dimension makes
the multiplication sample-wise. This is a correction to this reconstruction
implementation; it is not a statement about the published paper.

## 4. Losses

Read `losses/loss_function.py`.

`HierarchicalSmoothingLoss` uses a fixed soft target matrix that makes adjacent
emotion classes partially similar. `VIBE_Loss` then combines that classification
term, an affect-branch KL term, optional SAT, and a squared cosine
orthogonality term between affective and environmental latents.

Important scope detail: V1.1 disables SAT because the reconstruction has no
text anchors. The KL calculation currently uses the affect branch only even
though the VIB has two branches. This is retained and disclosed as a
reconstruction fidelity deviation; it is not hidden or silently changed.

## 5. Training and selection

Read `trainer/training_utils.py`, then `trainer/train_model.py`.

`WarmupCosineScheduler` exposes an epoch-indexed three-epoch linear warmup and
cosine decay. `compute_evaluation_metrics()` fixes class order and produces all
metrics from one place.

`VIBETrainer.__init__()` seeds Python, NumPy, and PyTorch, resolves the frozen
split, creates loaders, builds the model/loss/AdamW/scheduler, and starts a
provenance record. `run()` trains one epoch at a time and calls `validate()` on
the validation loader. `best_metric=macro_f1` is a strict-improvement rule:
ties leave the earlier selected checkpoint in place.

Only after all 30 epochs does `run()` load `model_best.pth` and call
`validate(..., mode='Test')` on the isolated test loader. Therefore a terminal
heading that says `Test Report | Epoch 30` does not mean epoch-30 weights were
tested; `test_checkpoint_epoch` in the JSON report records the selected model.

## 6. Evaluation and evidence

For V1.1, inspect the following in order:

1. `reports/robustness/gecv_v1_1_seed20260928_metrics.json`
2. `reports/robustness/gecv_v1_1_seed20260929_metrics.json`
3. `reports/robustness/gecv_v1_1_seed20260930_metrics.json`
4. `reports/final_baseline_v1_1/gecv_v1_1_three_seed_aggregate.json`
5. `reports/final_baseline_v1_1/gecv_v1_1_three_seed_integrity_audit.md`

The V1.1 checkpoints were selected at epochs 10, 12, and 14. The recorded
three-seed descriptive test aggregate is 95.08 ± 0.00% accuracy and
0.937 ± 0.002 macro-F1. These numbers are not a significance claim and are
not directly comparable to the paper's reported benchmark.

`testing/test_model.py` and `testing/inference.py` are older generic interfaces.
They do not resolve the frozen reconstruction split or provide the V1.1
evaluation protocol, so they should not be used as evidence for the V1.1
result.

## 7. Tests and audits

Run the CPU tests from the repository root with `PYTHONPATH="$PWD"`:

- `tests/gamma_scene_scale_invariance_cpu.py`: V1.1 gamma regression test.
- `tests/gecv_reconstruction_cpu_integration.py`: real HDF5 loader/model/loss
  integration.
- `tests/gecv_reconstruction_pre_gpu_cpu_test.py`: frozen split and one step.
- `tests/gecv_effective_k_cpu_integration.py`: historical effective-K adapter.
- `tests/ablation_switches_cpu.py`: no-training shape/behavior checks for
  prepared ablation switches.

Then read the dataset audit in `reports/dataset_audit/` and the final package
in `reports/submission/gecv_vibe_final_reconstruction_package/`. The audit
documents feature finiteness, split counts, labels, duplicate records, leakage
scope, and protected checkpoint hashes.

## Current versus historical material

**Current:** V1.1 source, `train_gecv_reconstruction_v1_1*.yaml`, three V1.1
run reports/checkpoints, and `reports/final_baseline_v1_1/`.

**Historical and preserved:** V1.0 Run B, V1.0 seed robustness, V1.0 effective
K=3/5/8, legacy VGAF/generic interfaces, and report-local source snapshots.
They remain useful provenance, but their metrics are never pooled with V1.1.

For a shorter cross-reference, use [`CODE_MAP.md`](CODE_MAP.md). For exact
call-chain and shape information, use [`EXECUTION_FLOW.md`](EXECUTION_FLOW.md).
