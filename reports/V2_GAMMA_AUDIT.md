# V2 gamma / scene-scale regression audit

**Scope:** VIBE--GECV--RECON-v1.1 current model path. This is a CPU regression
audit; it did not train, change data, alter a checkpoint, or write a model.

## Implementation location

- HDF5 gamma loading/collation:
  `dataloading/vibe_dataset.py` (`vibe_collate_fn`)
- Current multiplication:
  `model/vibe_model.py` (`VIBE_Transformer.forward_with_gamma`)
- Gamma-conditioned AdaLN:
  `model/gamma_transformer.py` (`DecoupledAdaLN.forward`)
- Regression test:
  `tests/gamma_scene_scale_invariance_cpu.py`

## Historical reconstruction issue

The historical V1.0 reconstruction multiplied `gamma [B,1,1]` by a
scene-derived scale shaped `[B,1]`. Under PyTorch broadcasting those singleton
dimensions can expand to `[B,B,1]`, creating an unintended route by which one
batch item's scene scale could affect another item's gamma conditioning. The
project's V1.0 impact probe recorded nonzero companion/order sensitivity. This
is an implementation-level reconstruction finding; it is not a claim about the
published VIBE paper.

## V1.1 correction

The current observed-scene path is:

```python
style_token = self.global_style_proj(global_feat_avg)       # [B,64]
scene_scale = self.scene_scaler(style_token).unsqueeze(-1)  # [B,1,1]
gamma_prime = gamma * scene_scale                            # [B,1,1]
```

The loader supplies stored reconstruction gamma as `[B,1,1]`. Keeping both
factors at `[B,1,1]` preserves one conditioning scalar per sample. The
temporal transformer receives `gamma_prime [B,1,1]`, expands it across time
only inside `DecoupledAdaLN`, and retains a temporal sequence `[B,32,512]`.
The final classifier concatenates `h_final [B,512]` with one scalar gamma
summary, hence fusion width 513 and logits `[B,3]`.

## Executed regression test

Command:

```bash
PYTHONPATH="$PWD" python tests/gamma_scene_scale_invariance_cpu.py
```

Result: **PASS**.

| Check | Observed result |
|---|---|
| Gamma input shape | `[2,1,1]` |
| Scene-scale shape | `[2,1,1]` |
| Gamma-prime shape | `[2,1,1]` |
| Temporal input shape | `[2,32,512]` |
| Fusion width | `513` |
| Logits shape | `[2,3]` |
| Same-sample companion perturbation | max absolute difference `0.0` |
| Batch-order permutation | max absolute difference `0.0` |
| Finiteness | PASS |
| Mask immutability | PASS |

The test loads the protected V1.0 Run B checkpoint only to exercise the
current V1.1 code path with fixed real weights; it does not save or change that
checkpoint. The checkpoint path used by the test is
`checkpoints/gecv_reconstruction_v1_runB/model_best.pth`.

## Conclusion

The current V1.1 model passes the intended per-sample gamma conditioning
invariance checks. V1.0 results are preserved as historical and are not mixed
with V1.1 results.
