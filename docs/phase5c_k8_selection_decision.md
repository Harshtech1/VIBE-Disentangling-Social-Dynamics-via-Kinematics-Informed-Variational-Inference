# Phase 5C K=8 person-selection decision

Status: **CUSTOM RECONSTRUCTION DEVIATION** — not an original VIBE policy.

## Source finding

The available VIBE sources establish only an agent cap, not a retention rule:

- The repository README says performance peaks around `K=8` and declares
  `K_max=8`.
- `config/train_vgaf.yaml`, `config/train_gecv.yaml`, and
  `config/inference.yaml` set `max_k: 8`.
- `data_preprocessing/extract_primitives.py` takes the first up-to-eight
  available persons/trajectory keys; it does not rank by duration,
  confidence, box area, centrality, or affect relevance.
- The supplied VIBE paper and transferred forensic audit identify K=8 as the
  selected maximum, but no explicit rule for choosing eight tracks from more
  than eight detected people was found. The forensic audit separately records
  that the legacy path used insertion/track-ID order and had a K=5 cap.

Therefore, source material does **not** specify how to select K=8 people when
ByteTrack returns more than eight tracks.

## Custom Phase 5C reconstruction rule

For diagnostics only, the Phase 5C script applies:

1. collect every observed ByteTrack ID and its observed sampled-frame length;
2. rank tracks by descending observed length;
3. break equal-length ties by ascending numeric track ID;
4. select the first eight; and
5. when eight or fewer tracks exist, select every observed track.

The selected tracks occupy leading positions in an explicit K=8 representation;
remaining positions are zero-padded only downstream. Synchrony uses only the
selected tracks, excludes the diagonal, and excludes padded rows/columns when
computing gamma. No temporal mask is introduced by this decision.

## Required diagnostic record

For every pilot clip, `phase5c_3video_diagnostic.json` and its Markdown report
record all observed IDs, all observed lengths, the full ranking, selected IDs,
the exact rule, and this deviation status.

## Sensitivity analysis requirement

This policy can alter DINO person features, synchrony gamma, and ultimately
classification. Plausible alternatives include tracker/insertion order, mean
detection confidence, person-box area, frame coverage, or scene centrality.
Any result using this custom policy must be labelled as a reconstruction result
and accompanied by a K-selection sensitivity analysis before comparison with a
paper claim.

## Unchanged scientific gates

- No VideoMAE-v2 checkpoint is substituted and no global T=32 mapping is made.
- Native HuBERT sequences are not forced to T=32.
- No GECV text is fabricated.
- No canonical VIBE HDF5 is written.
- No training occurs.
