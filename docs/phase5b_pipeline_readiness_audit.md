# Phase 5B pipeline readiness audit

Scope: read-only inspection of the current repaired repository. No raw GECV video was decoded and no preprocessing, feature extraction, HDF5 write, training, or inference was run.

## Executable code paths

| Stage | Current path | Behavior | Status |
| --- | --- | --- | --- |
| Raw GECV loader | `data_preprocessing/preprocessing_gecv.py:193-227` | Searches `GAVID/` or root `Train/train/Training/training`, `Val/...`, and `Test/...`, then recursively finds MP4s. | **WARNING**: available GECV archives use `positive/neutral/negative`, so no direct raw-GECV path is currently selected. |
| 6-Hz sampling | `preprocessing_gecv.py:330-351` | OpenCV decodes RGB frames; stride is `round(fps / target_fps)`, target 6 Hz. | Executable after GPU gate for downstream device use; does not force 32 frames. |
| YOLOv8-L + ByteTrack | `preprocessing_gecv.py:95-133` | `YOLO('yolov8l.pt')`, class 0, confidence 0.5, `tracker='bytetrack.yaml'`; Gaussian xywh smoothing with sigma `5/3`. | **BLOCKED** presently by CUDA driver visibility and checkpoint working-directory availability. |
| Tube construction | `preprocessing_gecv.py:135-156` | Crops each observed track box and resizes to 224x224. | Same gate as tracking. |
| HuBERT | `preprocessing_gecv.py:43-92` | `librosa.load(..., sr=16000, mono=True)` then `facebook/hubert-base-ls960`; returns native `[N,768]`, or `None` on exception. | Executable in dependency terms; needs model availability and a scientific audio-to-T rule. |
| Intermediate HDF5 | `preprocessing_gecv.py:353-384` | Writes `full_frames`, `audio_features`, `persons/<id>/{visual,boxes,raw_coords}` to `<output>/data/<split>/<id>.h5`. | Do not execute until raw layout/labels are resolved. |
| DINOv2 | `data_preprocessing/extract_primitives.py:29-32,85-133` | `facebook/dinov2-base`; middle full-frame CLS for `env_feat`, per-person crop CLS sequences. | Executable in dependency terms; needs weights and explicit person-selection policy. |
| Global VideoMAE | `extract_primitives.py:24-27,77-83` | Uses `MCG-NJU/videomae-base` with 16 uniformly selected frames. | **BLOCKED scientifically**; this is not an established VideoMAE-v2 path. |
| RoBERTa text | `extract_primitives.py:34-37,138-139` | `roberta-base`, applied to `Description`, output `pooler_output` `[1,768]`. | **BLOCKED scientifically** for GECV: no legitimate sample-specific description source. |
| Synchrony | `extract_primitives.py:39-68` | first up-to-eight insertion-order tracks; normalized centroid velocity; pairwise truncated cosine mean. | Executable mechanically; retention and time alignment remain undecided. |
| Final primitive HDF5 | `extract_primitives.py:150-207` | Writes canonical-named `<id>_vibe.h5` datasets. | **BLOCKED** until global/text/label decisions are resolved. |

## Current global representation

- Identity: `MCG-NJU/videomae-base`, loaded through Transformers `VideoMAEModel`, not a verified VideoMAE-v2 checkpoint.
- Input: 16 frames selected uniformly from all sampled frames; processor defaults are model-controlled.
- Output: expected/previously observed `[1,1568,768]` `last_hidden_state`.
- Model behavior: `model/vibe_model.py:55-64` computes `spatial_patches = 1568 // T` and views the token tensor as `[B*T, spatial_patches, D]`, then pools to `[B,T,D]`.
- Scientific status: **BLOCKED**. Divisibility by 32 is not evidence of a legitimate temporal token mapping. No mapping is invented here.

## Text behavior

The final extractor requires spreadsheet fields `Vid_name`, `Description`, and `Label` (`extract_primitives.py:160-201`). The raw preprocessor can read a `description`/`caption` annotation column (`preprocessing_gecv.py:286-321`) but the transferred GECV metadata audit found empty descriptions. Current GECV text is therefore not sample-specific and must not be used as a semantic-alignment target.

## K, T, masks, and labels

| Concern | Current behavior | Readiness |
| --- | --- | --- |
| K | Feature engine caps persons at 8 by existing person/track order (`extract_primitives.py:100`, `:43`); no quality ranking is defined. Dataset also truncates/pads to K=8 (`dataloading/vibe_dataset.py:37-53`). | **WARNING** |
| T | Raw decoding is 6 Hz but variable length. Feature extractor uses 16 global frames. Dataset truncates/pads person and audio sequences to T=32 (`vibe_dataset.py:40-65`). | **BLOCKED**: no approved real-time T=32 contract. |
| Agent masking | Dataset records `real_k`, pads to 8, and gamma excludes padded agents/diagonal (`vibe_dataset.py:85-100`). | Partially defined. |
| Temporal masking | No per-track temporal mask is propagated; padded local/audio positions are ordinary zeros. | **BLOCKED** scientific decision. |
| Raw preprocessor labels | `Positive->0`, `Negative->1`, `Neutral->2` (`preprocessing_gecv.py:187-191`). | **BLOCKED**: conflicts with canonical loader mapping. |
| Canonical loader labels | `0=Positive`, `1=Neutral`, `2=Negative` (`vibe_dataset.py:67-72`). | Required target contract. |
| Final extraction labels | copied directly from spreadsheet `Label` (`extract_primitives.py:201`). | Must be canonicalized before production. |

## What is executable now

Python imports, FFmpeg/ffprobe, and all named runtime libraries are verified. Read-only code inspection is complete. Media decode can run on CPU, but the raw-preprocessor default device is `cuda`; its tracker, HuBERT, and tube path will fail until the Lightning process exposes the GPU or an explicit CPU diagnostic decision is made.

## Required scientific decisions before Phase 5C

1. Exact VideoMAE-v2 checkpoint/revision and defensible global temporal representation.
2. Legitimate sample-specific GECV text source and its allowed use; otherwise an explicitly approved ablation.
3. Deterministic K=8 retention policy and temporal mask semantics.
4. Audio-to-T=32 alignment rule.
5. Canonical GECV label translation and a raw directory/annotation adapter for `positive/neutral/negative` archives.

## Final readiness status

**BLOCKED for H001/N001/S001 execution.** Environment imports are ready, but CUDA visibility is absent and the global/text/K/T/label contracts still require explicit resolution. No pipeline should be executed until those gates are closed.
