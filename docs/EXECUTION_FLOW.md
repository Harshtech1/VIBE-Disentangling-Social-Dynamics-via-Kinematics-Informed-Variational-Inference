# Actual V1.1 execution flow

This document follows imports and calls in the current code. It describes the
**VIBE-GECV-RECON-v1.1** path. It does not infer unrecorded behavior from the
paper or from similarly named legacy files.

## 1. Offline extraction: raw video to HDF5

**Entry point:** `scripts/extract_gecv_reconstruction.py` (manual, GPU-heavy).
It is not executed by training or by the audits.

1. `main()` checks CUDA, `lap`, `ffmpeg`, archives, and YOLO weights; it creates
   a temporary scratch directory and updates
   `data/manifests/gecv_reconstruction_extraction_manifest.json` atomically.
2. `Engine.decode()` reads each source video with OpenCV and samples it at
   approximately 6 Hz: `step = round(fps / 6)`.
3. `Engine.tracks()` uses YOLOv8l person tracking with ByteTrack, smooths each
   track's boxes, and records detection confidence.
4. `select_k8_duration_confidence_id()` in
   `data_preprocessing/reconstruction_contract.py` ranks tracks by observed
   duration, mean confidence, then numeric ID. It selects at most eight.
5. `Engine.local()` crops selected people, runs DINOv2 CLS extraction, then
   `map_local_tracks()` pools variable-time observations onto an endpoint-
   inclusive 32-bin grid. It returns:
   - `local_features`: `[8,32,768] float32`
   - `local_time_mask`: `[8,32] bool`
   - `agent_mask`: `[8] bool`
6. `Engine.audio()` decodes 16-kHz mono PCM with `ffmpeg`, extracts HuBERT
   states, and calls `bin_pool()` to create `audio_features [32,768]` and
   `audio_time_mask [32]`.
7. `Engine.global_scene()` passes 16 sampled frames to VideoMAEv2, then mean
   pools non-feature axes to `global_scene [768]`.
8. `shared_interval_synchrony()` calculates normalized centroid velocities over
   common consecutive sampled-frame intervals. It returns `physics_sync [8,8]`,
   `synchrony_pair_mask [8,8]`, and scalar `gamma` as the mean valid pairwise
   cosine similarity (or zero when no pairs are valid).
9. `validate()` checks shapes, labels, masks, and finite values before an
   atomic HDF5 replacement. The record stores label metadata, feature metadata,
   and `text_status=OFF` / `l_sat_status=DISABLED_ZERO`.

The extractor is a documented reconstruction choice, not evidence that every
unpublished paper-preprocessing detail was reproduced.

## 2. Frozen split: records to train/validation/test lists

**Entry point:** `dataloading/reconstruction_split.py`.

- `build_split_manifest()` inspects completed HDF5 files, removes the three
  smoke-only IDs, uses NumPy child seed sequences per label, and records fixed
  counts.
- `validate_split_manifest()` rejects overlapping IDs, missing/extra eligible
  IDs, or class-count differences.
- The active trainer calls `load_split_files(manifest_path, data_root)`, which
  re-inspects HDF5 metadata, verifies the corpus hash, validates membership,
  and returns explicit `train`, `validation`, and `test` lists.

The frozen manifest has seed `20260928`, 282 training records, 61 validation
records, and 61 test records. It is the only split source for V1.1 training.

## 3. HDF5 adapter and collation

**Entry point:** `VIBEDataset.__getitem__()` and `vibe_collate_fn()` in
`dataloading/vibe_dataset.py`.

For each V1.1 record, the adapter checks the reconstruction protocol and reads:

| Field in batch | Shape | Meaning |
|---|---:|---|
| `local_p` | `[B,8,32,768]` | Padded per-agent DINOv2 features. |
| `local_time_mask` | `[B,8,32]` | Valid local feature positions. |
| `agent_mask` | `[B,8]` | Active agent slots. |
| `audio_seq` | `[B,32,768]` | Temporal HuBERT features. |
| `audio_time_mask` | `[B,32]` | Valid audio bins. |
| `global_v` | `[B,768]` | Pooled VideoMAEv2 scene feature. |
| `sync_mat` | `[B,8,8]` | Pairwise synchrony matrix. |
| `synchrony_pair_mask` | `[B,8,8]` | Valid non-self synchrony pairs. |
| `gamma` | `[B,1,1]` | Stored reconstruction gamma, one scalar per sample. |
| `labels` | `[B]` | Positive=0, Neutral=1, Negative=2. |

The normal V1.1 path uses the stored HDF5 gamma. The derived-gamma branch is
for compatible data/effective-K experiments and averages **only active, valid,
non-self pairs**. `env_feat` and `text_anch` are `None` under the active
reconstruction protocol; the loader does not invent them.

## 4. Model construction and forward pass

**Constructor:** `VIBE_Transformer` in `model/vibe_model.py`.
**Call site:** `VIBETrainer.__init__()` and
`VIBETrainer.run()/validate()` in `trainer/train_model.py`.

`forward_with_gamma()` consumes the batch fields above in this sequence:

1. It enforces `local_time_mask &= agent_mask[..., None]` and fills absent
   masks with conservative all-valid defaults only for legacy callers.
2. With reconstruction mode, `global_v [B,768]` passes directly to the
   global-style projector. V1.1 does not reshape it into a temporal global
   token sequence.
3. `global_style_proj` creates `style_token [B,64]`.
   `scene_scaler(style_token).unsqueeze(-1)` creates
   `scene_scale [B,1,1]`.
4. `CausalVIB` flattens local vectors to `[B*8*32,768]` and returns affective
   and environmental means/log-variances/latents. The model restores them to
   `[B,8,32,512]` and masks padding.
5. During training only, mixup jointly mixes affect/env latents, style token,
   gamma, and targets. At validation/test the VIB returns deterministic means
   (`z = mu`).
6. `ContextInjectionGate` maps `[B,8,32,512]` affect/environment latents to a
   context-modulated affective sequence. The active switch is enabled.
7. The visual sequence is reshaped to `[B,32*8,512]`; projected audio is
   `[B,32,512]`. `CrossModalAttentionBlock` applies visual queries over audio,
   then visual self-attention and an FFN, using masks.
8. Visual tokens are reshaped to `[B,32,8,512]` and mean-pooled across valid
   agents to `[B,32,512]`. `temporal_mask` is `[B,32]`.
9. `gamma_prime = gamma * scene_scale` remains `[B,1,1]` in V1.1. The
   `GammaGatedTransformer` expands the scalar across time inside `DecoupledAdaLN`
   and applies two masked attention/MLP blocks.
10. `LearnedQueryAttention` pools temporal tokens to `h_final [B,512]`.
    The scalar gamma summary is concatenated to `[B,513]`, then the classifier
    returns logits `[B,3]` in canonical class order.

## 5. Objective, optimizer, and schedule

`VIBE_Loss` in `losses/loss_function.py` returns:

```text
total = hierarchical_smoothing_classification
      + lambda_sat * SAT
      + lambda_ortho * squared_cosine(z_aff, z_env)
      + lambda_kl * affect_branch_KL
```

In active V1.1, text/SAT is off (`lambda_sat=0`), `lambda_ortho=0.1`, and the
explicit KL coefficient defaults to 1.0. The KL term uses the affect branch
only; this is an acknowledged reconstruction fidelity deviation, not silently
changed by V2.

`VIBETrainer` creates AdamW and `WarmupCosineScheduler`. The V1.1
configuration uses batch size 16, 30 epochs, `lr=1e-4`, weight decay `1e-3`,
three warmup epochs, eta-min 0, mixup 0.2, and gradient clip 1.0.

## 6. Training, checkpoint selection, and test evaluation

**Entry point:**

```bash
PYTHONPATH="$PWD" python trainer/train_model.py --config <config> \
  --log-dir <logdir> --checkpoint-dir <checkpointdir> --run-report <report.json>
```

`VIBETrainer.run()` performs all configured epochs. Each epoch:

1. sets the scheduled learning rate;
2. trains only the training loader;
3. validates only the frozen validation loader;
4. saves `model_best.pth` only if validation `macro_f1` is strictly greater
   than the prior best;
5. records losses, validation metrics, the selected checkpoint path/hash, and
   scheduler state in the JSON run report.

After the final epoch, the trainer reloads `model_best.pth` and evaluates the
frozen test loader. `compute_evaluation_metrics()` uses canonical class order
to produce accuracy, weighted/macro F1, macro precision/recall, classwise
metrics, and the confusion matrix. The terminal label `Test Report | Epoch 30`
names the completed training-loop epoch; `test_checkpoint_epoch` in provenance
identifies the actual selected checkpoint.

## 7. Validation and regression entry points

| Command | What it checks | It does not do |
|---|---|---|
| `python tests/gamma_scene_scale_invariance_cpu.py` | V1.1 `[B,1,1]` gamma/scene shapes, no companion/order effect, logits `[B,3]`. | Train or modify a checkpoint. |
| `python tests/gecv_reconstruction_cpu_integration.py --report /tmp/gecv_cpu_integration.json` | Real HDF5 loader/model/loss interface. | Extract the corpus or train an epoch. |
| `python tests/gecv_reconstruction_pre_gpu_cpu_test.py --report /tmp/gecv_pre_gpu_cpu.json` | Frozen split, canonical labels, one CPU step. | Full training. |
| `python tests/gecv_effective_k_cpu_integration.py` | Effective-K masking and derived gamma. | Change physical HDF5 K=8 storage. |
| `python tests/ablation_switches_cpu.py` | Switch tensor contracts/baseline equivalence. | Produce an ablation result. |
| `python scripts/audit_gecv_dataset_integrity.py` | Read-only data, split, duplicate, label, and Run-B-hash checks. | Repair data or split. |

## Outputs and provenance

- Per-run provenance: `reports/robustness/gecv_v1_1_seed*_metrics.json`.
- Three-seed aggregate and integrity audit:
  `reports/final_baseline_v1_1/`.
- Supervisor package:
  `reports/submission/gecv_vibe_final_reconstruction_package/`.
- Final technical report:
  `reports/final_delivery/VIBE_GECV_FINAL.pdf`.

The active V1.1 source and run/report hashes are listed in the aggregate
integrity audit. V1.0 files remain historical evidence and are intentionally
not used to make V1.1 claims.
