# GECV Phase 5A: three-video real-data pilot

Date: 2026-09-28. Scope: only `videoH001.mp4` (Positive), `videoN001.mp4` (Neutral), and `videoS001.mp4` (Negative). No VIBE code/architecture change, full extraction, training, or historical-checkpoint evaluation occurred.

## Status legend

- **PASS** — directly verified or checksum-verified forensic evidence.
- **WARNING** — incomplete or non-canonical evidence.
- **BLOCKED** — cannot be scientifically defended from available inputs.

## 1. Transfer verification — PASS

`sha256sum -c checksums/SHA256SUMS` reported `OK` for every listed archive, manifest, configuration, code reference, model weight, and audit. The named original source path is not mounted here; the transfer manifest identifies it as the origin of the checksum-identical archives. Only three requested MP4 members were extracted to `/tmp`.

## 2. Raw archive verification — PASS

| Archive | Class | MP4s | Bytes |
| --- | --- | ---: | ---: |
| `happy.tar.xz` | Positive | 201 | 3,443,031,656 |
| `neutral.tar.xz` | Neutral | 107 | 1,835,673,056 |
| `negative.tar.xz` | Negative | 100 | 1,984,093,712 |
| **Total** |  | **408** | 7,262,798,424 |

Counts are independently recorded in the checksum-verified source manifest. They describe the available 408-clip subset, not the paper-reported full GECV corpus.

## 3. Pilot MP4 verification

| Video | Class | Bytes | Container | Media details |
| --- | --- | ---: | --- | --- |
| `videoH001.mp4` | Positive | 13,434,081 | ISO MP4 Base Media v1 | **BLOCKED live**: `ffprobe` absent. Prior transferred Phase 4D record: 8.846803 s, 30 FPS, 263 frames, AAC 44.1-kHz stereo. No video codec/resolution retained. |
| `videoN001.mp4` | Neutral | 15,878,113 | ISO MP4 Base Media v1 | **BLOCKED**: no `ffprobe`, OpenCV, or prior N001 probe record. |
| `videoS001.mp4` | Negative | 22,422,192 | ISO MP4 Base Media v1 | **BLOCKED**: no `ffprobe`, OpenCV, or prior S001 probe record. |

No N001/S001 duration, FPS, frame count, resolution, video codec, audio codec, sample rate, or channels have been invented.

## 4. Tracking, DINOv2, HuBERT, and synchrony

### Current live run — BLOCKED

This runtime has no `ffmpeg`/`ffprobe`, OpenCV, Ultralytics, Transformers, Librosa, SoundFile, PyAV, Torchaudio, or CUDA. The isolated temporary dependency installation did not complete. Therefore no fresh 6-Hz decode, YOLOv8-L + ByteTrack, DINOv2, HuBERT, smoothing, synchrony, or HDF5 write ran for any pilot clip. No zero-filled or substitute tensor was created.

### Transferred H001 forensic cross-check — WARNING

| Component | H001 result from prior checksum-verified diagnostic |
| --- | --- |
| Sampling | 30 FPS; stride 5; 53 sampled frames |
| Tracking | YOLOv8-L + ByteTrack; 25 observed tracks |
| K=8 diagnostic | IDs `[1,7,6,19,21,18,14,2]`; lengths 49,36,33,33,33,31,28,21 |
| Smoothing | Gaussian `sigma=5/3`, xywh centroids |
| Synchrony | finite symmetric `[8,8]`; gamma `-0.00421175` |
| Local DINO | `facebook/dinov2-base`; 159 finite `[768]` CLS vectors |
| Audio/HuBERT | real 16-kHz mono PCM; `facebook/hubert-base-ls960` finite `[442,768]` |

This H001 evidence is forensic only and does not establish N001/S001 behavior.

## 5. Global VideoMAE status — BLOCKED

Current code calls `VideoMAEImageProcessor` and `VideoMAEModel` with `MCG-NJU/videomae-base`, despite its “VideoMAE V2” comment. The recorded path uses 16 selected 224px frames and emits `[1,1568,768]`. It supplies no VideoMAE-v2 checkpoint/revision, pooling rule, or defensible 1,568-token-to-`T=32` mapping. The model's divisibility-based reshape is not a documented temporal contract. No V2 model was substituted or downloaded.

## 6. Text status — BLOCKED

`extract_primitives.py` reads `row['Description']`, encodes with `roberta-base`, and creates `[1,768]`. The transferred metadata audit records empty descriptions for all 407 derived samples; no Video-ChatGPT captions, prompts, revision, or sample-specific text manifest exists. Legacy anchors have only two fingerprints across the corpus. No text was fabricated.

## 7. Legacy-artifact comparison — WARNING

Legacy HDF5 files themselves were not transferred, so only checksum-verified manifest metadata is available.

| Item | H001 | N001 | S001 |
| --- | --- | --- | --- |
| Legacy label/class | 1 / Positive | 2 / Neutral | 3 / Negative |
| Required key names | present | present | present |
| `global_v_seq` | `[1,1568,768]` float32 | same | same |
| `env_feat` | `[1,768]` float32 | corpus contract | corpus contract |
| `audio_seq` | `[0]` float32 | `[0]` float32 | `[0]` float32 |
| `text_anch` | `[1,768]` float16 | `[1,768]` float16 | `[1,768]` float16 |
| persons/lengths | 5 / 2,36,49,37,7 | 5 / 23,4,18,13,17 | 5 / 16,16,11,8,1 |
| `physics_sync` | `[5,5]` float64 | `[5,5]` float64 | `[5,5]` float64 |

Legacy `1/2/3` labels conflict with the repaired canonical `0=Positive, 1=Neutral, 2=Negative`; legacy K=5 and empty audio cannot be made valid by loader padding.

## 8. Current HDF5 contract — BLOCKED

No Phase 5A output may be declared valid. The current contract needs `global_v_seq`, `env_feat`, `text_anch`, `person_sequences`, `physics_sync`, `audio_seq`, canonical label, and an explicit `K=8`, `T=32`, `D=768` contract. Blockers: absent global contract, absent legitimate text, unavailable current HuBERT rerun, undefined K=8 retention/masks, and label mismatch.

## 9. Recommended next gate

| Issue | Status | Required gate |
| --- | --- | --- |
| Transfer integrity and three raw members | PASS | none |
| H001 prior diagnostic | WARNING | repeat beside N001/S001 in pinned environment |
| Current decode/tracking/DINO/HuBERT environment | BLOCKED | pinned ffmpeg, OpenCV, Ultralytics, Transformers, audio decoder, approved weights |
| VideoMAE-v2 and T=32 | BLOCKED | exact checkpoint/revision and documented representation rule |
| Sample-specific text | BLOCKED | legitimate source/provenance or approved ablation |
| K=8, T=32, masks, audio alignment | BLOCKED | freeze and test one contract |
| Corpus size | WARNING | 408 available, not paper-reported 627 |

**Final gate: do not run full 408-video extraction.** Fresh successful diagnostics for all three clips, a documented global contract, legitimate text strategy, and defined K/T/masking behavior are required first.
