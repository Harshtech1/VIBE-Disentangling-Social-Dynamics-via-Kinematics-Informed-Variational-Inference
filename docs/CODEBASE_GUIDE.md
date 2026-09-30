# VIBE-GECV codebase guide — start here

This is an audited reconstruction, not an exact paper benchmark reproduction.
The active baseline is **VIBE-GECV-RECON-v1.1**; V1.0 is historical.

## Recommended reading order

1. `README.md` — repository scope.
2. `config/train_gecv_reconstruction_v1_1.yaml` — frozen V1.1 recipe.
3. `dataloading/vibe_dataset.py` — HDF5 record to tensors/masks.
4. `dataloading/reconstruction_split.py` — frozen 282/61/61 membership.
5. `model/vibe_model.py` — complete forward path and V1.1 gamma correction.
6. `model/vib.py` and `model/context_gate.py` — latent branches/context.
7. `model/learned_attenttion.py` and `model/gamma_transformer.py` — audio
   interaction, pooling, and temporal reasoning.
8. `losses/loss_function.py` — current objective and fidelity note.
9. `trainer/train_model.py` — training/validation/test lifecycle.
10. `tests/gamma_scene_scale_invariance_cpu.py` — correction regression guard.
11. `reports/submission/gecv_vibe_final_reconstruction_package/` — verified
    results, provenance, and scientific scope.

## How one sample moves through the code

`*_recon_v1.h5` → `VIBEDataset.__getitem__` → `vibe_collate_fn` → batch
dictionary (`local_p`, `audio_seq`, `global_v`, `gamma`, masks, labels) →
`VIBE_Transformer.forward_with_gamma` → logits `[B,3]` → `VIBE_Loss`.

The fixed interface is K=8 agent slots, T=32 time bins, and width 768. Masks
state which slots are genuinely observed; zero padding alone is not evidence
of invalid data.

## How one training epoch moves through the code

Frozen train paths → DataLoader batches → model forward → loss/backward/
optimizer step → `validate(..., 'Val')` → validation macro-F1 checkpoint
selection → epoch provenance entry. After all 30 epochs, the trainer reloads
`model_best.pth` and evaluates the held-out test split once. Test values do not
choose the checkpoint.

## Read results safely

Do not combine historical V1.0 values with V1.1 values. V1.1 corrected the
gamma/scene-scale batch broadcasting issue. Corrected V1.1 K=3/K=5 and V1.1
ablations remain pending.
