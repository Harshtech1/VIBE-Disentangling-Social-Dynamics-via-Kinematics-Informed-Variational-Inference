# Professor walkthrough: exact places to inspect

| Question | Open this file | What to look for |
|---|---|---|
| What happens to one video? | `scripts/extract_gecv_reconstruction.py` | Reconstruction record creation. |
| Where is tracking implemented? | `data_preprocessing/preprocessing_gecv.py` | Detection/tracking and trajectories. |
| Where is gamma computed? | `dataloading/vibe_dataset.py` | `compute_masked_gamma`; stored gamma is preferred. |
| Where is VIB? | `model/vib.py` | Dual Gaussian latent encoders. |
| Where does audio enter? | `model/vibe_model.py`, `model/learned_attenttion.py` | Audio projection and cross-modal attention. |
| Where is the transformer? | `model/gamma_transformer.py` | AdaLN and masked self-attention. |
| Where is classification? | `model/vibe_model.py` | Gamma fusion and three logits. |
| Where is the loss? | `losses/loss_function.py` | HCE, affect KL, orthogonality, optional SAT. |
| Where is training controlled? | `trainer/train_model.py` | Seeds, loaders, epochs, scheduler, reports. |
| How is best checkpoint selected? | `trainer/train_model.py` | `best_metric=macro_f1` from validation only. |
| Where is the gamma fix? | `model/vibe_model.py` | V1.1 `[B,1,1]` scale comment and code. |
| Where are audits/results? | `reports/submission/gecv_vibe_final_reconstruction_package/` | Final reports, figures, hashes, limitations. |

For detailed explanations of shapes and each active model stage, see
`docs/beginner_codebase_guide.md`.
