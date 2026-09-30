# VIBE-GECV beginner codebase guide

This guide explains the active VIBE-GECV reconstruction path without changing
runtime behavior. It is intended for someone reading the repository for the
first time. Paths below are the source of truth; older VGAF-oriented modules
and historical V1.0 reports are not the active V1.1 baseline.

## Start here: what happens to one sample

1. A reconstruction HDF5 record is read by
   `dataloading/vibe_dataset.py`.
2. The loader returns padded numeric tensors plus Boolean masks. Padding is
   necessary because clips do not all have the same number of people or valid
   temporal bins.
3. `vibe_collate_fn` stacks records into a batch and retains the stored
   physical-time synchrony scalar `gamma`.
4. `model/vibe_model.py` combines local person features, audio, a global
   scene vector, gamma, and masks to emit three class logits.
5. `losses/loss_function.py` computes the classification, KL, and
   orthogonality terms. Text alignment is deliberately disabled for the
   active reconstruction protocol.
6. `trainer/train_model.py` trains for 30 epochs, chooses the best checkpoint
   using validation macro-F1 only, reloads that checkpoint, and finally
   evaluates the test split.

`0=Positive`, `1=Neutral`, and `2=Negative` everywhere in the active loader,
metrics helper, and reports.

## 1. Data contract: `dataloading/vibe_dataset.py`

### `VIBEDataset.__getitem__`

This method reads exactly one HDF5 file. It detects the reconstruction schema
from the stored protocol/keys, then returns a dictionary rather than a tuple.
The named dictionary avoids accidentally swapping modalities at call sites.

Important returned fields:

| Field | Shape after loading | Meaning |
|---|---:|---|
| `local_p` | `[8, 32, 768]` | DINOv2 appearance representation per selected person/time bin. |
| `global_v` | `[768]` | Pooled scene representation. It is a single vector, not 32 global video tokens. |
| `audio_seq` | `[32, 768]` | HuBERT representation per temporal bin. |
| `sync_mat` | `[8, 8]` | Pairwise motion-similarity matrix. |
| `agent_mask` | `[8]` | True only for real selected people. |
| `local_time_mask` | `[8, 32]` | True only where a person/time-bin feature is valid. |
| `audio_time_mask` | `[32]` | True only where audio is valid. |
| `sample_gamma` | scalar | Stored physical-time mean synchrony. |

The variable-length `selected_agent_ids` HDF5 field is provenance metadata.
It correctly has fewer than eight entries when fewer than eight people were
valid; the fixed-size model interface is represented by padded tensors plus
masks. It must not be treated as a malformed model input.

### `compute_masked_gamma`

This fallback is used only when stored gamma is absent. It constructs a valid
pair mask from active agents, removes self-pairs, optionally intersects the
stored synchrony-pair mask, then averages only the remaining values. The
returned shape is `[B, 1, 1]`, one scalar per batch item. This is important:
returning `[B]` or a shared scalar would risk unintended broadcasting.

### `vibe_collate_fn`

PyTorch's `DataLoader` gives the collate function a Python list of individual
records. The function stacks like-shaped fields into `[B, ...]` tensors. It
rejects a mixed batch in which some samples have stored gamma and others do
not, or some samples have environment features and others do not. Such
rejection is intentional: silently mixing schemas would make results hard to
interpret.

## 2. Latent encoders: `model/vib.py`

`CausalVIB` contains two similarly shaped encoders:

- `aff_encoder` emits affective Gaussian parameters.
- `env_encoder` emits environmental/context Gaussian parameters.

Each final linear layer has width `2 * latent_dim`, then `.chunk(2, dim=-1)`
splits it into `mu` (mean) and `logvar` (log variance). During training,
`reparameterize` samples `mu + eps * std`; during evaluation it returns `mu`.
This convention makes evaluation deterministic while retaining the usual
variational sampling behavior during training.

## 3. Context gate: `model/context_gate.py`

The gate normalizes affective features and obtains a scale and shift from the
environment feature:

`normalized_affect * (1 + scale) + shift`.

The final context projection is initialized to zero. At initialization the
gate therefore behaves as plain normalized affect features rather than making
an arbitrary context-dependent change before training has learned one.

## 4. Attention and temporal model

### `model/learned_attenttion.py`

`LearnedQueryAttention` scores every token against a learned query. A mask
sets invalid scores to a very negative number before softmax, effectively
giving padded positions zero weight.

`CrossModalAttentionBlock` first lets visual tokens attend to audio tokens,
then applies visual self-attention and a feed-forward network. The
`enable_cross_modal_attention=False` path is an ablation intervention: it
removes audio-to-visual attention but retains normalization, visual
self-attention, feed-forward processing, masks, and output shape.

### `model/gamma_transformer.py`

`DecoupledAdaLN` is adaptive layer normalization. The global scene style
predicts a baseline scale, a shift, and a gamma sensitivity. Gamma is expanded
or interpolated to the sequence length before the scale is applied. The
transformer repeats: normalize/attention/residual, then normalize/MLP/residual.
At the end of each layer invalid time steps are multiplied by the time mask so
padding cannot reappear through residual connections.

## 5. Main model: `model/vibe_model.py`

`forward_with_gamma` is the main execution path. Read it in these stages:

1. **Inputs and default masks.** The batch is `[B, K, T, D]`; defaults are
   created only for backward-compatible callers.
2. **Global scene.** `global_scene [B,768]` is projected into a style token.
3. **VIB.** Local features are flattened, encoded, reshaped to `[B,K,T,L]`,
   and immediately zeroed where `local_time_mask` is false.
4. **Mixup.** During training only, latent streams, scene style, gamma, and
   labels are consistently permuted/mixed together.
5. **Context and cross-modal attention.** Local tokens become `[B,T*K,L]` so
   each person/time token can use audio. Masks remain attached.
6. **Temporal aggregation.** Person tokens are averaged using real-agent/time
   weights, not a simple average over padding.
7. **Gamma conditioning.** Gamma and scene scale are combined and condition
   the temporal transformer.
8. **Pooling and classification.** A learned attention pool produces one
   vector per clip; it is concatenated with a gamma scalar and classified.

### Critical V1.0 to V1.1 correction

V1.0 multiplied `gamma [B,1,1]` by `scene_scale [B,1]`. Standard broadcasting
can align these as `[B,B,1]`, allowing sample A's gamma to interact with sample
B's scene scale. In V1.1 the code uses:

```python
scene_scale = self.scene_scaler(style_token).unsqueeze(-1)  # [B, 1, 1]
gamma_prime = gamma * scene_scale                            # [B, 1, 1]
```

This is a reconstruction implementation correction, not evidence of a bug in
the published paper. The CPU regression test checks companion and order
sensitivity are both zero after the correction.

## 6. Loss: `losses/loss_function.py`

The total loss is:

`hierarchical classification + lambda_sat * SAT + lambda_ortho * orthogonality + lambda_kl * KL`.

- Hierarchical classification is the supervised signal.
- KL regularizes the affect latent distribution toward a unit Gaussian.
- Orthogonality is squared cosine similarity between affect and environment
  latents; lower values encourage separability.
- SAT compares a rationale to a text anchor, but it is zero/disabled in V1.1.

Fidelity note: the active KL expression uses `mu_aff` and `logvar_aff`, not
the environmental branch. The paper describes both branches; this is a
documented reconstruction deviation, not a measured ablation result.

## 7. Training and evaluation: `trainer/train_model.py`

`VIBETrainer.__init__` sets seeds, constructs loaders from the frozen manifest,
builds the model/loss/optimizer/scheduler, and starts a JSON provenance record.
The training loop writes an epoch record only after that epoch has completed.
`best_metric` is validation macro-F1. Test evaluation occurs after all epochs:
the trainer loads `model_best.pth` before calling `validate(..., mode='Test')`.
The printed `Test Report | Epoch 30` is a reporting label, not evidence that
epoch-30 weights were used; corrected evaluation reports record the selected
checkpoint epoch separately.

## 8. Metrics and safe tests

`trainer/training_utils.py` establishes a single canonical class order and
uses scikit-learn for accuracy, macro/weighted F1, precision, recall, and a
3-by-3 confusion matrix. This centralization prevents inconsistent label names
between validation and test reporting.

Safe, no-training checks:

```bash
PYTHONPATH="$PWD" python tests/gamma_scene_scale_invariance_cpu.py
PYTHONPATH="$PWD" python tests/gecv_reconstruction_cpu_integration.py
PYTHONPATH="$PWD" python tests/ablation_switches_cpu.py
```

## 9. Reading order for a newcomer

1. `docs/data_contract.md` and this guide.
2. `dataloading/vibe_dataset.py`.
3. `model/vib.py`, `context_gate.py`, and `vibe_model.py`.
4. `losses/loss_function.py` and `trainer/training_utils.py`.
5. `trainer/train_model.py`.
6. `tests/gamma_scene_scale_invariance_cpu.py`.
7. `reports/submission/gecv_vibe_final_reconstruction_package/`.

Never combine historical V1.0 metrics with corrected V1.1 metrics.
