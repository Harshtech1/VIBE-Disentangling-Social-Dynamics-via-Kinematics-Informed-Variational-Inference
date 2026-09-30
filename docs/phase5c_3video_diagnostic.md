# Phase 5C GECV three-video diagnostic

Run UTC: 2026-09-28T01:03:00Z

No final canonical VIBE HDF5 files were written.

## videoH001 — Positive

Canonical label: `0`

| Stage | Status | Seconds | Detail |
| --- | --- | ---: | --- |
| locate | PASS | 60.833 | completed |
| ffprobe | PASS | 0.063 | completed |
| decode_and_6hz_sample | PASS | 1.197 | completed |
| yolov8l_bytetrack | PASS | 2.64 | completed |
| trajectory_smoothing | PASS | 0.018 | completed |
| k8_selection | WARNING | 0.0 | CUSTOM RECONSTRUCTION DEVIATION: the paper/repository does not define this K=8 retention ranking. |
| dinov2_person_features | PASS | 6.09 | completed |
| synchrony | PASS | 0.002 | completed |
| hubert_audio | PASS | 2.573 | completed |
| videomae_contract | BLOCKED | 0.0 | Current code names MCG-NJU/videomae-base, samples 16 frames, and yields a token sequence; it is not asserted as VideoMAE-v2 and no scientific T=32 mapping is applied. |
| text_contract | BLOCKED | 0.0 | completed |
| canonical_label | PASS | 0.0 | completed |

### K=8 selection (custom reconstruction deviation)

- Observed IDs: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 16, 17, 18, 19, 20, 21, 22, 26, 29, 31, 33, 35, 36, 37]`
- Observed lengths: `{'1': 49, '2': 21, '3': 6, '4': 5, '5': 17, '6': 33, '7': 36, '8': 20, '9': 15, '10': 17, '14': 28, '16': 2, '17': 11, '18': 31, '19': 33, '20': 21, '21': 33, '22': 8, '26': 10, '29': 10, '31': 9, '33': 4, '35': 1, '36': 1, '37': 1}`
- Ranked IDs: `[1, 7, 6, 19, 21, 18, 14, 2, 20, 8, 5, 10, 9, 17, 26, 29, 31, 22, 3, 4, 33, 16, 35, 36, 37]`
- Selected IDs: `[1, 7, 6, 19, 21, 18, 14, 2]`
- Rule: descending observed track length; ties broken by ascending numeric track ID
- Sensitivity: Alternative retention rules (confidence, area, centrality, or tracker order) can change local features and synchrony, so results must be sensitivity-checked.

## videoN001 — Neutral

Canonical label: `1`

| Stage | Status | Seconds | Detail |
| --- | --- | ---: | --- |
| locate | PASS | 60.332 | completed |
| ffprobe | PASS | 0.07 | completed |
| decode_and_6hz_sample | PASS | 1.934 | completed |
| yolov8l_bytetrack | PASS | 2.031 | completed |
| trajectory_smoothing | PASS | 0.007 | completed |
| k8_selection | WARNING | 0.0 | CUSTOM RECONSTRUCTION DEVIATION: the paper/repository does not define this K=8 retention ranking. |
| dinov2_person_features | PASS | 1.954 | completed |
| synchrony | PASS | 0.002 | completed |
| hubert_audio | PASS | 1.079 | completed |
| videomae_contract | BLOCKED | 0.0 | Current code names MCG-NJU/videomae-base, samples 16 frames, and yields a token sequence; it is not asserted as VideoMAE-v2 and no scientific T=32 mapping is applied. |
| text_contract | BLOCKED | 0.0 | completed |
| canonical_label | PASS | 0.0 | completed |

### K=8 selection (custom reconstruction deviation)

- Observed IDs: `[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 21, 25]`
- Observed lengths: `{'1': 28, '2': 13, '3': 8, '4': 3, '5': 13, '6': 13, '7': 11, '8': 17, '10': 6, '11': 28, '12': 1, '13': 20, '14': 22, '21': 2, '25': 8}`
- Ranked IDs: `[1, 11, 14, 13, 8, 2, 5, 6, 7, 3, 25, 10, 4, 21, 12]`
- Selected IDs: `[1, 11, 14, 13, 8, 2, 5, 6]`
- Rule: descending observed track length; ties broken by ascending numeric track ID
- Sensitivity: Alternative retention rules (confidence, area, centrality, or tracker order) can change local features and synchrony, so results must be sensitivity-checked.

## videoS001 — Negative

Canonical label: `2`

| Stage | Status | Seconds | Detail |
| --- | --- | ---: | --- |
| locate | PASS | 36.312 | completed |
| ffprobe | PASS | 0.073 | completed |
| decode_and_6hz_sample | PASS | 2.766 | completed |
| yolov8l_bytetrack | PASS | 2.824 | completed |
| trajectory_smoothing | PASS | 0.017 | completed |
| k8_selection | WARNING | 0.0 | CUSTOM RECONSTRUCTION DEVIATION: the paper/repository does not define this K=8 retention ranking. |
| dinov2_person_features | PASS | 4.065 | completed |
| synchrony | PASS | 0.001 | completed |
| hubert_audio | PASS | 0.959 | completed |
| videomae_contract | BLOCKED | 0.0 | Current code names MCG-NJU/videomae-base, samples 16 frames, and yields a token sequence; it is not asserted as VideoMAE-v2 and no scientific T=32 mapping is applied. |
| text_contract | BLOCKED | 0.0 | completed |
| canonical_label | PASS | 0.0 | completed |

### K=8 selection (custom reconstruction deviation)

- Observed IDs: `[1, 2, 3, 4, 5, 9, 10, 12, 16, 21, 24, 26, 28, 29, 30, 37, 39, 42, 43, 45, 46, 48, 49, 51, 53, 57, 58, 59, 61, 62, 63, 65, 69, 71, 72, 74, 75, 76, 77, 81]`
- Observed lengths: `{'1': 16, '2': 16, '3': 11, '4': 8, '5': 1, '9': 1, '10': 16, '12': 2, '16': 16, '21': 48, '24': 19, '26': 54, '28': 3, '29': 4, '30': 49, '37': 39, '39': 49, '42': 3, '43': 5, '45': 1, '46': 4, '48': 3, '49': 2, '51': 5, '53': 41, '57': 35, '58': 23, '59': 12, '61': 32, '62': 11, '63': 9, '65': 11, '69': 4, '71': 4, '72': 7, '74': 6, '75': 6, '76': 6, '77': 6, '81': 2}`
- Ranked IDs: `[26, 30, 39, 21, 53, 37, 57, 61, 58, 24, 1, 2, 10, 16, 59, 3, 62, 65, 63, 4, 72, 74, 75, 76, 77, 43, 51, 29, 46, 69, 71, 28, 42, 48, 12, 49, 81, 5, 9, 45]`
- Selected IDs: `[26, 30, 39, 21, 53, 37, 57, 61]`
- Rule: descending observed track length; ties broken by ascending numeric track ID
- Sensitivity: Alternative retention rules (confidence, area, centrality, or tracker order) can change local features and synchrony, so results must be sensitivity-checked.

## Scientific guards

- VideoMAE path is recorded as `MCG-NJU/videomae-base`; it is not called VideoMAE-v2 and no T=32 global mapping is made.
- K=8 track retention uses a custom length-then-ID reconstruction policy, not an original VIBE claim; alternative rules require sensitivity analysis.
- Native HuBERT time is reported without a T=32 conversion.
- No GECV text is generated and no final HDF5 is written.
