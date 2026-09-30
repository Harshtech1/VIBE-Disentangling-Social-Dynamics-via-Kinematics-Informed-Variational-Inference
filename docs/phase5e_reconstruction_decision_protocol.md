# Phase 5E reconstruction decision protocol

Date: 2026-09-28 UTC  
Purpose: executive freeze specification for the VIBE GECV reconstruction. It records evidence, provenance, and Harsh-approved VIBE-GECV-RECON-v1.0 reconstruction choices. Approval is limited to implementation and H001/N001/S001 validation; it does not authorize corpus extraction, training, or a paper-faithful claim.

Evidence classes: **AUTHOR/PAPER CONFIRMED**, **RELEASE-CODE CONFIRMED**, **OFFICIAL BACKBONE CONFIRMED**, **CUSTOM RECONSTRUCTION DEVIATION**, and **UNRESOLVED / HARD GATE**.

## Protocol approval and version

| Field | Value |
| --- | --- |
| Protocol | **VIBE-GECV-RECON-v1.0** |
| Status | **APPROVED FOR IMPLEMENTATION AND THREE-VIDEO VALIDATION** |
| Scope | Recovered/processable GECV subset; currently 407 videos. This approval authorizes H001/N001/S001 validation only, not corpus extraction. |
| Research classification | Explicit reconstruction baseline. |
| Researcher approval | Harsh |
| Supervisor approval | Not claimed. |
| Not claimed | Exact VIBE reproduction; original author preprocessing; original VideoMAEv2 checkpoint; original text generation; original evaluation split; or reproduction of the reported 91.85% metric. |

## Decision register

| Item | Paper / author source | Released code | External evidence | Proposed reconstruction decision | Provenance status | Scientific consequence | Approval required | Implementation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Global video / VideoMAE-V2 | VideoMAE V2, X_glob in R^(T x 768), T=32. | MCG-NJU/videomae-base, 16 frames, [1,1568,768]. | OpenGVLab VideoMAEv2 is a distinct family; no author checkpoint/pooling recovered. | VIBE-GECV-RECON-v1.0 uses OpenGVLab/VideoMAEv2-Base mean-pooled to one 768-D scene-calibration vector only; never group 1568 tokens to 32. | **CUSTOM RECONSTRUCTION DEVIATION**; paper-faithful path remains hard-gated. | Scene conditioning is not paper-faithful temporal global video. | Researcher / Harsh approved for v1.0; no supervisor approval claimed. | Approved for H001/N001/S001 only. |
| Text / Video-ChatGPT | Video-ChatGPT descriptions and RoBERTa anchors. | Reads Description and encodes roberta-base. | All transferred GECV descriptions empty; no prompt, model/version, or clip-aligned output recovered. | V1.0 text ablation: text OFF and L_sat=0; no captions or RoBERTa anchors fabricated. | **CUSTOM RECONSTRUCTION DEVIATION**; strict track remains hard-gated. | Semantic alignment is intentionally absent. | Researcher / Harsh approved for v1.0; no supervisor approval claimed. | Approved for H001/N001/S001 only. |
| Shared T=32 timeline | T=32 fixed primitives. | Local/audio first-32 truncation; no common time grid. | No author temporal mapping recovered. | Use 32 monotonically increasing source-video timestamps spanning the clip; define endpoint and short-clip behavior; map every modality to that grid. | **CUSTOM RECONSTRUCTION DEVIATION** | Enables fixed-shape fusion but changes released preprocessing. | Researcher / Harsh approved; no supervisor approval claimed. | Implemented for v1.0; approved for H001/N001/S001 validation only. |
| Local DINOv2 | DINOv2 local tubes, [K,T,768]. | Native CLS crop sequence; variable track lengths. | Phase 5C native DINO works; no author resampling/mask contract. | Map valid agent observations to the shared grid; retain validity; prohibit silent first-32 truncation. | **CUSTOM RECONSTRUCTION DEVIATION** | Local temporal coverage and missingness become protocol-dependent. | Researcher / Harsh approved; no supervisor approval claimed. | Implemented for v1.0; approved for H001/N001/S001 validation only. |
| HuBERT | HuBERT, [T,768]. | Native hidden states; first-32 truncation in loader. | Phase 5C HuBERT works; no author mapping to T=32. | Deterministically decode audio, map native states to common 32 bins, retain audio/empty-bin validity. | **CUSTOM RECONSTRUCTION DEVIATION** | Audio alignment becomes a reconstruction protocol. | Researcher / Harsh approved; no supervisor approval claimed. | Implemented for v1.0; approved for H001/N001/S001 validation only. |
| Masks | Fixed K/T implied; no source mask definition. | Zero padding; only scalar gamma excludes padded agents. | No author mask semantics recovered. | agent_mask [B,K], local_time_mask [B,K,T], audio_time_mask [B,T], global_time_mask [B,T]; apply to attention, pooling, VIB, synchrony, losses, and classifier inputs where relevant. | **CUSTOM RECONSTRUCTION DEVIATION** | Requires audited model/interface change; zeros no longer behave as observations. | Researcher / Harsh approved; no supervisor approval claimed. | Implemented for v1.0; approved for H001/N001/S001 validation only. |
| Synchrony | Centroids, velocity direction/cosine, off-diagonal aggregation, gamma. | Truncates unequal tracks in observation order. | No source rule for intermittent-track alignment. | Preserve sampled frame indices; compare only shared physical consecutive intervals; exclude no-overlap pairs; reduce over valid active pairs. | **CUSTOM RECONSTRUCTION DEVIATION** | Gamma values can differ from released code. | Researcher / Harsh approved; no supervisor approval claimed. | Implemented for v1.0; approved for H001/N001/S001 validation only. |
| K=8 selection | Paper reports K=8 operating point. | No selection rule; legacy effective cap K=5. | No author ranking recovered. | Duration descending, mean detection confidence descending, then numeric track-ID ascending; select first eight. Sensitivity: A duration+confidence+ID, B insertion order, C alternative confidence/coverage rules. | **CUSTOM RECONSTRUCTION DEVIATION** | Selected people, DINO coverage, and gamma can change. | Researcher / Harsh approved; no supervisor approval claimed. | Implemented for v1.0; approved for H001/N001/S001 validation only. |
| Audio decode | HuBERT modality; decoder unspecified. | Librosa may fall back from PySoundFile to audioread and can leave empty audio. | Phase 5C proves valid native HuBERT after decode. | FFmpeg to 16-kHz mono PCM/WAV, then HuBERT; log FFmpeg version, input hash, sample rate, channels, and validation. | **CUSTOM RECONSTRUCTION DEVIATION** | Removes backend ambiguity; differs from released call. | Researcher / Harsh approved; no supervisor approval claimed. | Implemented for v1.0; approved for H001/N001/S001 validation only. |
| Labels | Three affect classes. | Raw GECV mapping conflicts with repaired loader. | Canonical loader and Phase 5C establish Positive=0, Neutral=1, Negative=2. | Reject unknown, one-based, and conflicting raw labels; validate every manifest before HDF5. | **RELEASE-CODE CONFIRMED + CANONICAL RECONSTRUCTION CONTRACT** | Prevents label corruption. | Researcher / Harsh approved; no supervisor approval claimed. | Implemented for v1.0; approved for H001/N001/S001 validation only. |
| Dataset / split | Paper-reported corpus and evaluation protocol. | GECV train-folder random group split; not fully recoverable. | 408 raw MP4s, 407 legacy derived records, transferred split manifests; exact paper split unverified. | Record paper, recovered-raw, and processable counts separately; do not claim exact paper split/benchmark without reconciliation. | **UNRESOLVED / HARD GATE** | Paper-faithful metrics cannot be claimed. | Researcher + Supervisor. | Not authorized. |

## Approval matrix

| Decision | Proposed choice | Evidence class | Approval required | Status |
| --- | --- | --- | --- | --- |
| Global representation | V1.0: OpenGVLab/VideoMAEv2-Base mean-pooled scene calibration; paper-faithful checkpoint recovery continues separately. | Custom deviation / paper hard gate | Researcher / Harsh for v1.0; Supervisor for paper-faithful claim | Approved for three-video validation only |
| Text | V1.0 text OFF and L_sat=0; authentic-description recovery continues for paper-faithful work. | Custom deviation / paper hard gate | Researcher / Harsh for v1.0; Supervisor for paper-faithful claim | Approved for three-video validation only |
| Timeline, local/audio mapping, decode, synchrony, K policy | Apply the register's explicit reconstruction choices. | Custom deviation | Researcher / Harsh; no supervisor approval claimed | Implemented for v1.0; approved for three-video validation only |
| Masks | Add explicit masks and propagate them through the audited reconstruction path. | Custom deviation | Researcher / Harsh; no supervisor approval claimed | Implemented for v1.0; approved for three-video validation only |
| Canonical labels | Positive=0, Neutral=1, Negative=2; reject conflicts. | Canonical reconstruction contract | Researcher / Harsh; no supervisor approval claimed | Implemented for v1.0; approved for three-video validation only |
| Dataset/split claim | Reconcile corpus and split before paper comparison. | Hard gate | Researcher + Supervisor | Unapproved / blocked |

Harsh approval applies only to VIBE-GECV-RECON-v1.0 implementation and the three-video validation gate. Supervisor approval is not claimed.

## Final freeze decision

### Paper-faithful track

Continue author/source recovery for VideoMAEv2 checkpoint and representation, authentic text, temporal/mask contract, and authoritative split. **No production extraction.**

### Reconstruction track

The listed **CUSTOM RECONSTRUCTION DEVIATION** choices are implemented for VIBE-GECV-RECON-v1.0 and approved by Harsh for the three-video gate. Any resulting model/report must be labelled a **reconstruction/ablation**, not exact VIBE reproduction.

## Next step

Run Phase 6 real three-video L4 validation on H001, N001, and S001. **No full-corpus extraction and no training may begin until that gate passes.**
