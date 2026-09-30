# Phase 6 VIBE-GECV-RECON-v1.0 three-video validation

Classification: explicit reconstruction baseline; not exact VIBE reproduction.

## Frozen reconstruction contract

- Local: [8,32,768] with local_time_mask [8,32] and agent_mask [8].
- Audio: deterministic FFmpeg 16-kHz mono PCM, HuBERT pooled to [32,768] with audio_time_mask [32]; audio failure is invalid, never zero-filled.
- Global: OpenGVLab/VideoMAEv2-Base mean-pooled to one [768] scene-calibration vector; no token-to-T reshape.
- Text: OFF, no anchors, and L_sat=0.
- Synchrony: centroid directional cosine on common consecutive sampled-frame intervals only.
- K=8: custom duration, confidence, ID ordering; not an author policy.

## videoH001 — Positive

| Stage | Status | Seconds | Detail |
| --- | --- | ---: | --- |
| locate | PASS | 0.0 | completed |
| ffprobe | PASS | 0.035 | completed |
| decode_6hz | PASS | 0.806 | completed |
| shared_t32_grid | PASS | 0.0 | completed |
| track | PASS | 1.039 | completed |
| trajectory_smoothing | PASS | 0.002 | completed |
| k8_custom_selection | PASS | 0.0 | completed |
| dino_native_and_grid | PASS | 2.231 | completed |
| ffmpeg_pcm | PASS | 0.107 | completed |
| videomaev2_scene | PASS | 3.085 | completed |
| hubert_native_and_grid | PASS | 0.248 | completed |
| shared_interval_synchrony | PASS | 0.003 | completed |
| write_pilot_h5 | PASS | 0.004 | completed |

## videoN001 — Neutral

| Stage | Status | Seconds | Detail |
| --- | --- | ---: | --- |
| locate | PASS | 0.0 | completed |
| ffprobe | PASS | 0.047 | completed |
| decode_6hz | PASS | 0.881 | completed |
| shared_t32_grid | PASS | 0.0 | completed |
| track | PASS | 0.438 | completed |
| trajectory_smoothing | PASS | 0.001 | completed |
| k8_custom_selection | PASS | 0.0 | completed |
| dino_native_and_grid | PASS | 1.443 | completed |
| ffmpeg_pcm | PASS | 0.113 | completed |
| videomaev2_scene | PASS | 2.821 | completed |
| hubert_native_and_grid | PASS | 0.237 | completed |
| shared_interval_synchrony | PASS | 0.002 | completed |
| write_pilot_h5 | PASS | 0.003 | completed |

## videoS001 — Negative

| Stage | Status | Seconds | Detail |
| --- | --- | ---: | --- |
| locate | PASS | 0.0 | completed |
| ffprobe | PASS | 0.045 | completed |
| decode_6hz | PASS | 1.227 | completed |
| shared_t32_grid | PASS | 0.0 | completed |
| track | PASS | 0.585 | completed |
| trajectory_smoothing | PASS | 0.002 | completed |
| k8_custom_selection | PASS | 0.0 | completed |
| dino_native_and_grid | PASS | 3.427 | completed |
| ffmpeg_pcm | PASS | 0.131 | completed |
| videomaev2_scene | PASS | 2.803 | completed |
| hubert_native_and_grid | PASS | 0.235 | completed |
| shared_interval_synchrony | PASS | 0.005 | completed |
| write_pilot_h5 | PASS | 0.003 | completed |

## Guards

- Text input is disabled and L_sat is disabled; no captions or RoBERTa anchors were created.
- Global scene uses OpenGVLab/VideoMAEv2-Base pooled to one 768-D calibration vector; no 1568-to-32 reshape occurs.
- Pilot HDF5 files are limited to three validation clips and are not production corpus outputs.
