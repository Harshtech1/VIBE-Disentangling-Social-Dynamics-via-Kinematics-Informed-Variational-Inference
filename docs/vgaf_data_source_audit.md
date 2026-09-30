# VGAF data-source readiness audit

Audit date: 2026-09-27. This is a metadata-only audit except for the small text
label and five-fold manifests. No media, feature pickle, or Kaggle dataset archive
was downloaded.

## Kaggle access

No Kaggle CLI or Python package is installed, no `kaggle.json` configuration file
is present, and no `KAGGLE_*` environment variables are configured. Public Kaggle
metadata endpoints are reachable anonymously; authenticated/download access has not
been tested and is not assumed.

## Resource inventory

| Resource | Raw video | Audio | Faces | Features | Labels | Splits | Text descriptions | Usable for VIBE | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `amirabdrahimov/vgaf-dataset`, v3, 22,586,093,949 bytes | **No**. Tree has no video container (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) | Yes: 2,661 train WAV and 766 validation WAV | Yes: `Train_faces/` and `Val_faces/` trees | No VIBE primitive features | `Train_labels.txt`, `Val_labels.txt` | Official train/validation plus five folds over train | None found | **No for the current VIBE preprocessing path** | VIBE needs pixel-level video for OpenCV, YOLO/ByteTrack, DINOv2 tubes, VideoMAE, and trajectories. This resource contains pre-extracted faces and audio only. |
| `amirabdrahimov/vgaf-features`, v4, 8,544,654,980 bytes | No | Validation WAV only (766 files) | No raw face crops established | Yes: five `enet_b2_8_vgaf_*.pickle` artifacts | `Train_labels.txt`, `Val_labels.txt` | Labels and a validation-audio directory; no official video split tree | None found | **No** | Features are EfficientNet-derived and incompatible with the canonical VideoMAE/DINO/HuBERT primitive contract. |
| `levevt2012/vgaf-videos`, v1, 32,021,888 bytes | **No**: files are `.npy`, not decodable video containers | No | No | Yes: visual `.npy` representations | No label manifest found | `Train` 2,193, `Validation` 617, `Test` 617 `.npy` files | None found | **No** | Does not contain pixels/video stream/audio required by VIBE. Its split sizes also differ from VGAF official train/val/test counts. |
| `levevt2012/vgaf-faces`, v1, 437,088,640 bytes | No | No | Yes: face `.npy` arrays | Face-derived arrays | No label manifest found | `Train` 2,193, `Validation` 617, `Test` 617 `.npy` files | None found | **No** | Faces are insufficient for VIBE global context, audio, detector/tracker, or trajectory extraction. |

## Primary resource tree

```text
amirabdrahimov/vgaf-dataset/
  Train_labels.txt                         2,661 labeled IDs
  Val_labels.txt                             766 labeled IDs
  5_fold/5_fold/
    {1..5}_train_fold.txt                  2,128 or 2,129 entries each
    {1..5}_test_fold.txt                     532 or 533 entries each
  Train_VGAF_audio-20220823T161233Z-001/Train_VGAF_audio/
    <clip_id>.wav                         2,661 files
  Val_VGAF_audio-20220823T161237Z-001/Val_VGAF_audio/
    <clip_id>.wav                           766 files
  Train_faces/Train_faces/<clip_id>/...   pre-extracted face tree
  Val_faces/Val_faces/<clip_id>/...       pre-extracted face tree
```

No testing media, testing label file, or raw-video directory is present. The
accessible primary-data counts are 2,661 train and 766 validation clips. The known
VGAF total is 4,183 clips, so the expected 756-test-clip partition is absent.

## IDs, labels, and folds

`data/manifests/vgaf_source_inventory.csv` is a complete metadata-only inventory of
the 3,427 accessible labeled WAV files. Each row is a clip ID and its exact public
Kaggle path and byte size. It contains no invented video paths.

Observed raw label manifests have columns `Vid_name Label`:

| Split | Clips | Raw label 1 | Raw label 2 | Raw label 3 | Unique source prefixes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 2,661 | 802 | 923 | 936 | 195 |
| Validation | 766 | 302 | 280 | 184 | 65 |

There are no duplicate IDs within either manifest, no label-to-WAV omissions, and no
WAV-to-label omissions within either split. The 5-fold files partition the 2,661
training IDs; fold test sizes are 533, 532, 532, 532, and 532. They encode labels as
`0/1/2`, where an exact comparison to `Train_labels.txt` establishes:

```text
manifest raw 1 -> fold 0
manifest raw 2 -> fold 1
manifest raw 3 -> fold 2
```

The public manifests contain numeric IDs only; they do not define the semantic names
Positive/Neutral/Negative. The requested semantic mapping therefore remains an
external protocol assertion, not something demonstrated by the files themselves.

## Video decodability decision

No representative media sample was downloaded because no raw video container exists
in the resource. `ffprobe`, OpenCV decode, frame-rate/frame-count/resolution checks,
and audio-stream checks are **not applicable**: WAV files can verify audio only and
face arrays cannot serve as a video stream. Running those tests on a `.wav` or face
tree would not validate VIBE's required OpenCV/YOLO/VideoMAE path.

## Text-description investigation

Neither the repository nor the four audited Kaggle trees includes a clip-ID-aligned
English `Description`/caption manifest. Public discussion of the later VGAF-GEMS
extension says contextual narratives were generated with Video-ChatGPT, but no
public, clip-aligned VGAF description file, coverage table, text format, or RoBERTa
embedding artifact was located in the audited resources. Thus it is not a currently
usable source for VIBE. References: https://arxiv.org/abs/2507.22393 and
https://nanless.github.io/audio-paper-digest-blog/posts/2026-07-04-vibe-disentangling-social-dynamics/.

## Readiness conclusion

The primary Kaggle resource is sufficient to audit VGAF labels, official
train/validation IDs, fold membership, WAV availability, and pre-extracted faces. It
is **not sufficient to begin VIBE preprocessing**, because it lacks the original
pixel-level videos. The minimum missing acquisition is an authorized source of the
original MP4 clips for the 2,661 train and 766 validation IDs, plus a clip-aligned
English-description manifest. The 756 official test clips and labels are additionally
needed only if the target evaluation protocol requires the full 4,183-clip split.

No Kaggle command can acquire a VIBE-ready video subset from this primary resource:
it contains no video files. After locating an authorized raw-video source, first
acquire only three label-matched clips (raw labels 1, 2, and 3) and their source
manifest; then run `ffprobe` and OpenCV decode before any preprocessing.
