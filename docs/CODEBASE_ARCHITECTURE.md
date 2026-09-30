# VIBE-GECV architecture map

```text
raw video
  -> data_preprocessing/ (6 Hz decoding, detection, tracking, features)
  -> reconstruction HDF5 record
  -> dataloading/vibe_dataset.py (tensors + masks + stored gamma)
  -> model/vibe_model.py
       -> model/vib.py                 (affective/environmental latents)
       -> model/context_gate.py        (context modulation)
       -> model/learned_attenttion.py  (audio/visual attention and pooling)
       -> model/gamma_transformer.py   (scene/gamma-conditioned temporal tokens)
  -> losses/loss_function.py
  -> trainer/train_model.py
       -> validation macro-F1 checkpoint selection
       -> held-out test after best-checkpoint reload
       -> reports, hashes, provenance
```

`dataloading/reconstruction_split.py` resolves the immutable 20260928 split
manifest. `tests/` guards shapes, masks, and gamma sample independence.
`scripts/` contains extraction, audits, plotting, and package generation.

## Major-file index

| File | Role | Input → output | Why it matters |
|---|---|---|---|
| `data_preprocessing/reconstruction_contract.py` | Storage rules | raw/extracted fields → schema expectations | Defines reconstruction interface. |
| `scripts/extract_gecv_reconstruction.py` | Extraction | videos → HDF5 records | Produces reproducible feature records. |
| `dataloading/vibe_dataset.py` | Adapter | HDF5 → batch tensors | Preserves masks/gamma/labels. |
| `dataloading/reconstruction_split.py` | Split | manifest → paths | Prevents reshuffling/leakage. |
| `model/vibe_model.py` | Model assembly | batch tensors → logits | Contains the V1.1 gamma correction. |
| `losses/loss_function.py` | Objective | logits/latents/labels → scalars | Documents affect-only KL deviation. |
| `trainer/train_model.py` | Lifecycle | config → checkpoints/reports | Enforces validation selection then test. |
| `tests/gamma_scene_scale_invariance_cpu.py` | Regression | checkpoint/batches → assertions | Guards batch independence. |
