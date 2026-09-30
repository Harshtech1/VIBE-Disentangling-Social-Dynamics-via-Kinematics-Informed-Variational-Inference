# Phase 5D representation-contract audit

Date: 2026-09-28 UTC. This is a static audit of the supplied paper, current working tree, frozen official-source record, transferred GECV audits, and Phase 5C manifest. No model, media, extraction, HDF5-production, or training workload ran.

Status: PASS = direct evidence; WARNING = executable but not paper-complete; BLOCKED = production decision cannot be justified from current source.

## Source-backed specification

The paper-facing specification and README name four 768-dimensional modalities: global scene context via VideoMAE V2, local agent tubes via DINOv2, acoustic features via HuBERT, and text anchors via RoBERTa. The frozen paper/code cross-reference records X_glob [T,768], X_loc [K,T,768], and X_aud [T,768], with T=32 and K_max=8. The README also specifies 6-Hz sampling and uniformly sampled T=32 frames.

Thus T=32 is a model-facing fixed-length primitive contract, not just a DataLoader convenience. The source does not specify the temporal resampling operator, shared timestamp grid, or mask semantics.

| Representation | Explicit source | Not source-specified |
| --- | --- | --- |
| Global video | VideoMAE V2; [T,768]; T=32. | Checkpoint/revision, clip sampling, token pooling, and conversion to 32 vectors. |
| Local DINO tracks | DINOv2; [K,T,768]; K_max=8; T=32. | Track retention, missed detections, resampling, and masks. |
| HuBERT audio | HuBERT last state; [T,768]. | Audio time alignment and native-HuBERT-to-32 mapping. |
| Text | Video-ChatGPT descriptions and frozen RoBERTa. | Description corpus, prompt, generator revision, coverage, and pooling. |

### Global visual encoder — BLOCKED

The paper/README says VideoMAE V2, but supplies no checkpoint ID, revision, or hash. Released extraction instead instantiates MCG-NJU/videomae-base using Hugging Face VideoMAEImageProcessor and VideoMAEModel. This is a VideoMAE-base ID, not evidence of VideoMAEv2.

The transferred compatibility audit identifies OpenGVLab/VideoMAEv2-Base as a distinct family. Its published base configuration has 224-pixel input, 16 frames, patch size 16, tubelet size 2, 768-dimensional embeddings, and mean pooling. That does not establish it as the paper author's selected checkpoint.

Conclusion: the paper claims VideoMAE V2, while released code does not substantiate it. MCG-NJU/videomae-base must be reported as the released-code model ID, never as VideoMAE V2. The available materials establish no author-intended global representation beyond [32,768].

### Synchrony — PARTIALLY SOURCED

Released code converts xywh boxes to normalized centroids, takes successive 2-D differences, L2-normalizes velocities, computes pairwise dot products/cosines, and averages off-diagonal pairwise means. The loader then reduces active off-diagonal matrix entries to scalar gamma. This supports centroid-velocity/cosine synchrony as an implementation fact.

Neither the paper nor inspected release specifies how intermittent trajectories must align. It does not state whether unequal tracks are to be compared in observation order, truncated from their starts, or aligned by shared sampled-frame indices.

### Labels — PASS

| Class | Canonical label |
| --- | ---: |
| Positive / videoH### | 0 |
| Neutral / videoN### | 1 |
| Negative / videoS### | 2 |

Phase 5C applies exactly this mapping to H001/N001/S001. It is the only mapping accepted by the repaired VIBEDataset.

## Current implementation

| Component | Current behavior | Status |
| --- | --- | --- |
| Sampling | preprocessing_gecv.py and Phase 5C use step=max(1, round(source_fps/6)); retain zero-based frame indices divisible by step. Pilot: 30 FPS and 53/64/89 samples for H001/N001/S001. | PASS for 6-Hz diagnostic, not fixed-T. |
| Tracking/smoothing | YOLOv8-L yolov8l.pt, ByteTrack, class 0, confidence .5; Gaussian sigma=5/3 only for tracks with at least five observations. | PASS on three clips. |
| K retention | Release uses first available keys and legacy GECV caps at five. Phase 5C ranks observed length descending then numeric ID ascending. | WARNING; custom deviation. |
| Local visual | facebook/dinov2-base CLS crop embeddings. Pilot totals: H001 [264,768], N001 [154,768], S001 [347,768]. | PASS natively; no [K,32,768] contract. |
| Global visual | extract_primitives.py selects 16 frames and MCG-NJU/videomae-base emits [1,1568,768]. | BLOCKED. |
| Audio | librosa.load(video, sr=16000, mono=True), then facebook/hubert-base-ls960. Pilot native outputs: [442,768], [533,768], [741,768]. | PASS natively; BLOCKED for T=32. |
| Text | Spreadsheet Description is encoded by roberta-base to pooler output [1,768]. | BLOCKED for GECV provenance. |
| Synchrony | calculate_physics_sync compares truncated observation-order velocities of leading tracks. Phase 5C custom-K8 gamma: -0.0311224, 0.0764099, 0.0969545. | WARNING. |

### Current T=32 behavior

VIBEDataset truncates each local person sequence and audio sequence to the first 32 feature vectors and zero-pads shorter sequences. It pads agents to eight. It does not resample uniformly over elapsed video/audio time, and it performs no equivalent global temporal normalization.

VIBE_Transformer tries to convert a 3-D global sequence longer than T by setting spatial_patches = sequence_length // T, reshaping to 32 groups, and attention-pooling each group. For [1,1568,768], this is 49 tokens per nominal group. No evidence establishes that the 16-frame encoder token order forms 32 temporal groups of 49. This is a divisible tensor reshape, not a defensible temporal mapping.

### Current masking behavior

The loader zero-pads missing agents and time positions. Only real_k is mask-like: compute_masked_gamma excludes padded agent rows/columns and the diagonal in gamma reduction. Padded agents/time are not represented by masks in cross-modal attention, temporal attention, GammaGatedTransformer, VIB statistics, orthogonality loss, or classification. Multi-head attention receives no padding mask. Therefore zero padding participates as feature content.

### Current label conflicts

The final loader enforces Positive=0, Neutral=1, Negative=2. preprocessing_gecv.py maps Positive=0, Negative=1, Neutral=2, which conflicts. extract_primitives.py directly copies spreadsheet Label to HDF5. Transferred legacy primitives use 1/2/3. A production adapter is required.

### Text inventory — BLOCKED

The re-search found only GECV_processed/metadata/train_index.json. Its 407 derived description fields are empty and group_emotion is Unknown. No clip-aligned captions, Video-ChatGPT outputs, prompts, revision, decoding configuration, transcript, or authoring record exists. Legacy text anchors have only two fingerprints, so nonempty shape does not demonstrate sample-specific semantics.

No legitimate GECV text input currently exists for semantic alignment. A future approved text ablation would have to remove or gate L_sat and explicitly represent text absence in inputs, losses, and reporting. It would be a reconstruction ablation, not a VIBE reproduction. No ablation is implemented.

### Runtime/model provenance

Phase 5C ran in the researcher-owned L4/CUDA environment: Python 3.12.11; PyTorch 2.8.0+cu128; OpenCV 4.10.0; Ultralytics 8.4.112; Transformers 5.17.0; Librosa 0.10.2.post1; SoundFile 0.14.0; PyAV 18.1.0; Torchaudio 2.8.0+cu128; h5py 3.16.0; NumPy 1.26.4; Pandas 2.1.4; SciPy 1.11.4; scikit-learn 1.3.2; FFmpeg/ffprobe 8.0.1; pip check passing.

| Asset | Identity | Current provenance/cache status |
| --- | --- | --- |
| Detector | yolov8l.pt | Present in transferred reference package; used by Phase 5C. VIBE provides no URL, pin, or hash. |
| Local encoder | facebook/dinov2-base | Present in default HF cache; loaded in Phase 5C. |
| Audio encoder | facebook/hubert-base-ls960 | Present in default HF cache; loaded in Phase 5C. |
| Released global encoder | MCG-NJU/videomae-base | Not found in default HF cache; not verified VideoMAE V2. No download attempted. |
| Text encoder | roberta-base | Not found in default HF cache; not run because valid GECV text is absent. |

## Confirmed divergences

1. **Global backbone/representation — BLOCKED.** Paper-facing source says VideoMAE V2 and [32,768]; release uses MCG-NJU/videomae-base, 16 frames, and [1,1568,768]. The 49-token grouping is not temporal semantics.
2. **Fixed time — BLOCKED.** Local/audio use first-32 truncation, not uniform coverage. DINO/HuBERT pilot features are variable length.
3. **K selection — WARNING.** Paper has a cap of eight but no selection rule. Legacy release is insertion/order based with K=5. Phase 5C duration-then-ID is custom.
4. **Synchrony alignment — WARNING.** Code compares first min(n,m) observation-order velocity vectors, not guaranteed common source-frame intervals.
5. **Mask propagation — BLOCKED.** Zero padding is not masked downstream except in scalar gamma reduction.
6. **Text — BLOCKED.** Paper-facing source calls for Video-ChatGPT descriptions; transferred GECV metadata is empty.
7. **Labels — BLOCKED for production.** Raw preprocessing conflicts with canonical mapping; legacy labels are one-based.
8. **Audio decode — WARNING.** The PySoundFile failed / audioread fallback warning makes decode backend-dependent; code can silently produce empty audio after exceptions.

## Unresolved scientific questions

1. Which VideoMAEv2 checkpoint, revision, preprocessing, feature layer/pooling, and 32-step representation did the authors use?
2. Is T=32 raw-video timestamp sampling, per-modality pooling, a shared 32-bin timeline, or another procedure?
3. Should DINO use every 6-Hz sample, detected samples only, interpolated states, or 32-bin valid detections?
4. Should synchrony compare velocity only at common decimated frame intervals?
5. Which descriptions, generator/version/prompt, and RoBERTa representation support semantic alignment?
6. Is zero padding intended to be masked in attention, VIB, and losses; how do missing people differ from missing observations?
7. Which complete GECV corpus/split and raw label convention are paper-authoritative? The transfer contains 408 MP4s, not the paper-reported corpus size.

## Recommended reconstruction decisions

These are recommendations for a future approved reconstruction specification. They are not implemented and are not claimed as author behavior.

1. **Shared reference timeline.** Record 32 monotonically increasing source timestamps spanning each clip, including endpoint and short-clip behavior. Map every modality to those timestamps explicitly. First-32 truncation is not uniform coverage.
2. **Synchrony: align physical time.** Retain sampled-frame indices. For each pair, compute velocity only on shared consecutive intervals (t,t+1), average cosine over those intersections, exclude pairs with no shared interval, then reduce gamma over valid active pairs. This avoids comparing early motion of one person with late motion of another. It is a reconstruction decision without author confirmation.
3. **Native HuBERT.** First recover author mapping. If unavailable, timestamp-pool valid HuBERT states into the same 32 video-time bins, record empty-bin behavior and audio validity mask, and label it a reconstruction decision. Do not use first-32 truncation or unlabelled interpolation.
4. **Mask contract.** Carry agent_mask[B,K] and time_mask[B,K,T], plus audio/global time masks where applicable. Apply them to attention key padding, masked pooling, synchrony pair validity, VIB pooling, and sequence losses. A padded zero must not count as observed content.
5. **Audio decode.** For a future production adapter, decode with pinned FFmpeg to 16-kHz mono PCM/WAV, validate stream/sample count, then feed PCM to HuBERT. Record FFmpeg version, command, sample rate, channels, and input hash. This avoids SoundFile/audioread fallback but is an engineering reconstruction decision, not a source-confirmed paper method.
6. **Labels.** Use one ingestion adapter: Positive to 0, Neutral to 1, Negative to 2; reject unknown/one-based values and validate manifests before writing features.

### Future K=8 sensitivity design (not executed)

Retain Phase 5C baseline: descending observed track length, then ascending numeric ID. With fixed clips, detector output, smoothing, temporal map, and weights, compare:

| Arm | Selection policy | Required record |
| --- | --- | --- |
| A | Duration descending, ID tie-break | IDs, lengths, rank, selected IDs. |
| B | Tracker/insertion order (first eight observed IDs) | ByteTrack output order and selected IDs. |
| C | Descending mean detection confidence, then duration, then ID | Per-detection confidence, per-track mean, ranks, selected IDs. |

Report selected agents, valid pair count, synchrony/gamma, and local feature coverage before any downstream metric. This is reconstruction sensitivity analysis, never evidence of the original K rule.

## Decisions requiring explicit labeling as deviations

The following must remain labelled **CUSTOM RECONSTRUCTION DEVIATION** without author confirmation:

- duration-then-track-ID K=8 retention, or any alternate ranking;
- any VideoMAEv2 checkpoint selected by the reproduction;
- every global token pooling or [32,768] mapping;
- all DINO/HuBERT time binning, interpolation, padding, or truncation;
- common-frame synchrony alignment;
- explicit FFmpeg PCM decoding in place of released Librosa loading;
- a text ablation disabling/reweighting semantic alignment; and
- mask propagation through model paths that now consume zero padding.

No such decision supports an exact-paper claim. It can support a clearly named reconstruction experiment after the protocol is frozen.

## Production-gating checklist

Do **not** start 407/408-video extraction or training until every applicable gate is closed and recorded.

- [ ] **BLOCKED:** Obtain author-supported VideoMAE V2 checkpoint/revision, preprocessing, feature layer/pooling, and defensible [32,768] contract.
- [ ] **BLOCKED:** Obtain clip-aligned descriptions with provenance, or approve/label text ablation before code changes.
- [ ] **BLOCKED:** Freeze one T=32 timeline and modality mappings for global/local/HuBERT; prohibit silent first-32 truncation.
- [ ] **BLOCKED:** Freeze agent/time/audio/global mask semantics and audit attention, pooling, synchrony, VIB, and loss consumers.
- [ ] **WARNING:** Freeze custom K=8 policy and schedule the three-arm sensitivity analysis; retain deviation labels.
- [ ] **WARNING:** Decide/document common-frame synchrony versus released observation-order truncation and hand-validate tracks.
- [ ] **BLOCKED:** Add/validate canonical label ingestion before HDF5.
- [ ] **WARNING:** Freeze FFmpeg PCM decode and reject empty/failed HuBERT features instead of silently writing (0,).
- [ ] **WARNING:** Pin/checksum YOLO, ByteTrack config, DINOv2, HuBERT, resolved global model, RoBERTa, and software versions.
- [ ] **BLOCKED:** Reconcile 408-source subset, one omitted legacy derivative, and paper corpus/split before benchmark claims.

**Gate conclusion: BLOCKED.** Phase 5C proves real GECV decode, tracking, smoothing, custom K=8 selection, DINOv2, synchrony, and native HuBERT. It does not establish paper-faithful VideoMAE V2, T=32, text, masks, or production labels. No full extraction or training is justified yet.

