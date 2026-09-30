# Canonical VIBE primitive HDF5 contract

Each sample is one file named `<clip_id>_vibe.h5` in either `train/` or `val/`.
The canonical dimensions are `K=8` agents, `T=32` timesteps, and feature dimension
`D=768`. Labels use the fixed order `0=Positive`, `1=Neutral`, `2=Negative`.

| HDF5 item | Required shape | Meaning |
| --- | --- | --- |
| `global_v_seq` | `[1, 1568, 768]` | VideoMAE token sequence. The current model reshapes this into `T=32` groups of 49 tokens. |
| `env_feat` | `[1, 768]` | Global DINO environmental feature. |
| `text_anch` | `[1, 768]` | RoBERTa text-anchor feature. |
| `person_sequences/p_0` through `p_7` | `[N_i, 768]`, `N_i >= 1` | One DINO feature sequence per available tracked person. Dataset pads/truncates each sequence to `[32, 768]`; fewer than eight people are zero-padded. |
| `physics_sync` | `[K_source, K_source]`, `1 <= K_source <= 8` | Pairwise synchrony matrix, zero-padded to `[8, 8]` by the Dataset. |
| `audio_seq` | `[N_audio, 768]`, `N_audio >= 1` | HuBERT sequence; Dataset pads/truncates it to `[32, 768]`. |
| attribute `label` | scalar integer in `{0,1,2}` | Canonical group-emotion target. |

The collated batch passed to `VIBE_Transformer.forward_with_gamma` is:

```text
local_p:   [B, 8, 32, 768]
global_v:  [B, 1568, 768]
env_feat:  [B, 768]
audio_seq: [B, 32, 768]
gamma:     [B, 1, 1]
labels:    [B]
```

`gamma` is computed from the non-padded `physics_sync` submatrix. The current model
accepts `env_feat` as part of its interface, although the present research model does
not consume it internally.
