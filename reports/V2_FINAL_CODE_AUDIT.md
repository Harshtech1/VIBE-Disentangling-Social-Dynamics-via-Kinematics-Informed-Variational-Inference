# V2 final code and repository audit

**Audit date:** 2026-09-30
**Scope:** current working tree of the GECV-only VIBE reconstruction.
**Safety:** no training, raw-video extraction, dataset rewrite, split rewrite,
checkpoint rewrite, Git-history rewrite, commit, or push was performed.

## A. Repository overview

The repository implements and documents a GECV reconstruction of VIBE. Its
current scientific protocol is **VIBE--GECV--RECON--v1.1**, a corrected
three-seed reconstruction baseline. The codebase has these major directories:
`config/`, `data/`, `data_preprocessing/`, `dataloading/`, `model/`, `losses/`,
`trainer/`, `testing/`, `tests/`, `scripts/`, `docs/`, `reports/`, `logs/`, and
`checkpoints/`.

At audit capture, the working tree was 27G: checkpoints 21G, data 455M,
reports 23M, logs 712K, and Git metadata 5.1G. Large size is expected because
the working tree intentionally retains experiment checkpoints and historical
evidence. No large scientific artifact was deleted.

## B. V1 baseline commit protection

- Branch: `main`
- Protected pre-V2 `HEAD`: `cde50422b52d601525f6d694cf7f8c80e2791e67`
- Commit summary: `Add GECV reconstruction implementation and validation`
- At capture, `main` and `origin/main` both pointed to `cde5042`.

The V1 commit/history was inspected only. No reset, rebase, amend, force push,
or history deletion was used. [`V2_PRE_AUDIT_BASELINE.md`](V2_PRE_AUDIT_BASELINE.md)
contains the capture record.

## C. V2 changes

V2 improves auditability and supervisor readability without changing the
scientific model/data protocol:

1. Replaced the legacy, paper-claim-oriented README with a concise active-V1.1
   research README, real commands, result scope, and explicit limitations.
2. Added `docs/CODE_MAP.md`, `docs/EXECUTION_FLOW.md`, and
   `docs/SUPERVISOR_CODE_WALKTHROUGH.md` based on the actual call graph.
3. Added this audit and `V2_GAMMA_AUDIT.md`.
4. Corrected the **validator classification**, not the data: production
   `audio_features`/`global_scene` aliases and the intentionally absent
   `reference_valid` field no longer cause legacy-strict-schema failures.
   Legacy differences remain recorded as warnings.
5. Added non-mutating `--report` options to the two real-data CPU tests so V2
   verification can write to `/tmp` rather than overwrite historical reports.
6. Updated an ablation-test comment/assertion to verify the current V1.1
   `[B,1,1]` gamma behavior.
7. Strengthened `.gitignore` for checkpoint extensions and common raw-media
   extensions.

## D. Source-code inventory

[`docs/CODE_MAP.md`](../docs/CODE_MAP.md) inventories every editable Python
source module in `data_preprocessing/`, `dataloading/`, `model/`, `losses/`,
`trainer/`, `testing/`, `tests/`, and `scripts/`. For each it records purpose,
caller, inputs, outputs, and whether it is active V1.1, historical, audit-only,
or documentation/build support.

Repository-wide supporting inventory findings:

| Category | Finding | V2 handling |
|---|---|---|
| Shell scripts | No editable `*.sh` files were present. | Nothing to move/delete. |
| Notebooks | No notebooks or notebook checkpoints were present. | Nothing to move/delete. |
| Generated source copies | `reports/final_report/assets/code/` mirrors selected source files for PDF compilation. | Preserved as report-local snapshots, not authoritative source. |
| Duplicate presentation packages | Two package directories contain intentionally copied reports/figures/source-report copies. | Preserved for standalone provenance; not code duplication. |
| Historical V1.0 reports/configs/checkpoints | Present under `reports/`, `config/`, `logs/`, and `checkpoints/`. | Preserved and labelled historical. |
| Current V1.1 reports/configs/checkpoints | Present in V1.1 seed paths and `reports/final_baseline_v1_1/`. | Preserved as current evidence. |
| Temporary/caches | Nine Python `__pycache__` directories found after validation commands. | Removed as disposable generated cache; exact paths below. |

## E. Actual execution graph

The verified executable V1.1 graph is:

```text
scripts/extract_gecv_reconstruction.py (offline, GPU-heavy)
  -> reconstruction_contract.py helpers
  -> HDF5 corpus + extraction manifest
  -> reconstruction_split.load_split_files(frozen manifest)
  -> VIBEDataset.__getitem__ + vibe_collate_fn
  -> VIBE_Transformer.forward_with_gamma
       -> CausalVIB -> ContextInjectionGate
       -> CrossModalAttentionBlock -> GammaGatedTransformer
       -> LearnedQueryAttention -> classifier logits [B,3]
  -> VIBE_Loss
  -> AdamW + WarmupCosineScheduler
  -> validation macro-F1 selection of model_best.pth
  -> reload model_best.pth -> isolated held-out test
  -> structured per-seed JSON provenance
```

`docs/EXECUTION_FLOW.md` provides call sites, transformations, tensor shapes,
and output locations. The frozen test loader is constructed separately and is
not used to select a checkpoint.

## F. Scientific implementation audit

| Area | Actual implementation observed | Status / scope |
|---|---|---|
| Labels | HDF5 attribute labels are checked as 0/1/2; metrics use Positive/Neutral/Negative order. | PASS. |
| Data masks | Loader emits agent/local/audio/pair masks and model re-applies agent masking to local time. | PASS. |
| Frozen split | `load_split_files()` validates manifest protocol, corpus hash, membership, and class counts. | PASS. |
| Temporal processing | Extraction samples about 6 Hz, maps observations to T=32 bins, retains invalid bins as masked zeros. | Documented reconstruction rule. |
| Agent selection | Up to K=8 tracks ranked by duration/confidence/ID. | Documented reconstruction deviation, not paper claim. |
| Local/audio/scene features | DINOv2 crop CLS, HuBERT temporal features, pooled VideoMAEv2 scene vector. | Implemented; pooled scene is a documented deviation. |
| Synchrony | Common-interval centroid velocity cosine similarities form pair mask/matrix/scalar gamma. | Implemented reconstruction rule. |
| VIB | Separate affect/environment Gaussian encoders; deterministic mean at evaluation. | Implemented. |
| KL | Current loss uses affect branch `mu_aff/logvar_aff` only. | Documented fidelity deviation; unchanged. |
| Context/audio fusion | Context gate and masked visual/audio cross-attention are in current model path. | Implemented. |
| Gamma transformer | Scene-conditioned Decoupled AdaLN, masked temporal attention/MLP, learned-query pooling. | Implemented. |
| Optimizer/schedule | AdamW, warmup/cosine, configured 30 epochs, mixup, gradient clipping. | V1.1 config-consistent. |
| Selection/test isolation | Best checkpoint is strictly selected by validation macro-F1 then reloaded for test. | PASS. |

No scientific behavior was silently changed during V2. The exact current model
path is documented in `docs/EXECUTION_FLOW.md` and
`docs/SUPERVISOR_CODE_WALKTHROUGH.md`.

## G. Data audit

Existing direct HDF5 and split audits establish:

- 408 readable source-video records; 404 usable split-assigned experimental
  records; documented tracking failure `videoH116`.
- Three smoke-only physical HDF5 records are intentionally unassigned, making
  407 physical accepted HDF5 files in the full schema-validator run.
- Frozen split: 282 train / 61 validation / 61 test.
- Class counts: train 139/74/69, validation 30/16/15, test 30/16/15 for
  Positive/Neutral/Negative.
- Labels, finiteness, masks, unique split assignment, sample/source-video
  split intersections, and exact duplicate-feature records passed.
- Participant/session overlap is **NOT_AVAILABLE**, not PASS: trustworthy
  identifiers are not stored and were not inferred.

Split SHA-256:
`75691440c5982cc9efa0311c9b8ba68139f318ba82858e42e3cacbd4271ec823`.
Source-corpus hash:
`d84c8fed0f72de4b9a485e3502d41c107a2b9c0c89ff000f5b880177448854b9`.

## H. Configuration audit

| Family | Classification | Finding |
|---|---|---|
| `train_gecv_reconstruction_v1_1.yaml` | Current V1.1 seed 20260928 base | Active corrected protocol. |
| `train_gecv_reconstruction_v1_1_seed20260929.yaml` | Current V1.1 seed config | Differs only by seed/output locations. |
| `train_gecv_reconstruction_v1_1_seed20260930.yaml` | Current V1.1 seed config | Differs only by seed/output locations. |
| `train_gecv_reconstruction_v1*.yaml` excluding v1.1 | Historical V1.0 | Preserved pre-fix baseline, seed, and effective-K settings. |
| `config/ablations/*.yaml` | Prepared/historical controls | Infrastructure; not V1.1 ablation headline results. |
| `reconstruction_gecv_v1.yaml` | Reconstruction data contract | K/T/text-off configuration support. |
| Generic GECV/VGAF/test/inference YAMLs | Legacy | Do not define current V1.1 evaluation. |

V2 parsed all YAML files. The three V1.1 configurations passed a strict
equivalence check after excluding only `seed`, `log_dir`, and `checkpoint_dir`.
All point to the same frozen split and preserve K=8, T=32, 768 input channels,
512 latent channels, 3 classes, 8 heads, depth 2, dropout 0.3, 30 epochs,
AdamW, `lr=1e-4`, weight decay `1e-3`, three warmup epochs, mixup 0.2,
gradient clipping 1.0, SAT off, and `lambda_ortho=0.1`.

## I. Gamma audit

The V1.1 per-sample gamma/scene-scale correction passed CPU regression testing.
See [`V2_GAMMA_AUDIT.md`](V2_GAMMA_AUDIT.md) for the implementation snippet,
shapes, invariance results, historical context, and command. Both `gamma` and
`scene_scale` were observed as `[B,1,1]`; same-sample companion and order
differences were exactly zero under the test tolerance.

## J. Security audit

No `.env`, `*.env`, `*.pem`, `*.key`, SSH private-key-like filename, credential
filename, or high-confidence key/token pattern was found outside `.git` and
large binary data. Secret values were never printed.

`.gitignore` protects Python caches, common secret files, archives, common raw
video/audio formats, HDF5 files, the reconstruction corpus directory,
checkpoints, logs, and common checkpoint extensions. It does not ignore source,
configuration, documentation, manifests, tests, or reports.

## K. Code-quality findings

### Safe V2 fixes made

- `scripts/validate_gecv_reconstruction_v1.py` now accepts the production
  extractor schema as valid while retaining strict-schema differences in a
  separate forensic field/warning. The former behavior falsely marked valid
  HDF5 records invalid due to legacy-only fields. Data/model behavior did not
  change.
- `tests/gecv_reconstruction_cpu_integration.py` and
  `tests/gecv_reconstruction_pre_gpu_cpu_test.py` accept optional `--report`
  paths, allowing non-mutating verification against `/tmp`.
- `tests/ablation_switches_cpu.py` now documents/asserts the V1.1
  `[B,1,1]` invariant for neutral global-scene conditioning.

### Retained, documented items

- `testing/test_model.py` and `testing/inference.py` are legacy generic
  interfaces. They do not drive V1.1 frozen-split metrics and are labelled as
  historical rather than rewritten.
- File names `learned_attenttion.py` and `vizualization.py` contain legacy
  spelling. Renaming would change imports/report snapshots; it was not needed
  for the active protocol.
- Some broad imports/legacy branches remain in historical modules. They were
  not removed by a broad stylistic rewrite.
- `VIBEDataset.__getitem__()` prints and returns `None` on load error, after
  which the collator drops that item. For this fixed audited corpus, schema
  validation passed; changing its error policy would change runtime behavior,
  so V2 documents rather than alters it.
- The extraction script has a user-home default transfer directory. It is a
  manual extraction default and not a hidden path in the V1.1 training path.

## L. Large-file audit

| Artifact class | Approximate size / examples | Category | V2 action |
|---|---|---|---|
| V1.1 checkpoints | Three seed directories; checkpoint-last files around 181M each, model files around 63M each. | Reproducibility-critical experiment artifacts. | Preserved; ignored. |
| V1.0/K/ablation checkpoints | Historical baseline, effective-K, and prior control files. | Historical scientific artifacts. | Preserved; ignored. |
| HDF5 corpus | 455M under `data/gecv_reconstruction_v1/`. | Source/reproducibility-critical data artifact. | Preserved; ignored. |
| Reports/packages | 23M. | Presentation/provenance artifacts. | Preserved. |
| `.git` | 5.1G, including historical large objects. | V1 history. | Preserved; no history rewrite. |

No current working-tree tracked file exceeded 10 MiB in the audit check. Large
runtime artifacts remain outside Git tracking through ignore rules.

## M. Tests executed

| Check | Result | Notes |
|---|---|---|
| `python -m compileall` on all editable Python directories | PASS | Syntax compilation succeeded. |
| Core V1.1 imports | PASS | Data/model/loss/trainer imports succeeded. |
| Broad editable-module import audit | NEEDS_ENVIRONMENT | 34/36 imported; optional visualization/test modules require absent `seaborn`. |
| `tests/gamma_scene_scale_invariance_cpu.py` | PASS | Shapes, invariance, finiteness, masks. |
| `tests/gecv_effective_k_cpu_integration.py` | PASS | Physical K=8/effective K behavior. |
| `tests/gecv_reconstruction_cpu_integration.py --report /tmp/...` | PASS | Real data, loader, model, loss, backward, one optimizer step. |
| `tests/gecv_reconstruction_pre_gpu_cpu_test.py --report /tmp/...` | PASS | Frozen manifest, labels, schedule, one CPU step. |
| `tests/ablation_switches_cpu.py` | PASS | Switch shape/regression checks; no training. |
| `tests/smoke_test.py` | PASS | Synthetic generic contract. |
| `scripts/validate_gecv_reconstruction_v1.py --report /tmp/...` | PASS | 407 accepted physical HDF5 files; H116 remains documented failed source. |
| V1.1 config equivalence and frozen split resolution | PASS | Three configs/scientific settings and split counts verified. |
| README/docs local link and command-path checks | PASS | 18 Markdown files checked; README Python paths exist. |

## N. Tests passed

All core scientific-contract tests listed above passed. The V1.1 gamma test
reported zero companion/order sensitivity, real-data integration used the
actual stored schema, and full physical HDF5 validation accepted all 407
materialized records (including the three intentionally smoke-only records).

## O. Tests failed or unavailable

The broad import audit did not import `trainer.vizualization` or
`testing.test_model` because the current environment lacks `seaborn`. This is
an environment dependency issue: `seaborn>=0.12.0` is already declared in
`requirements.txt`. It does not invalidate the active V1.1 training/evaluation
path, which imported successfully. No dependency installation was performed
by V2.

## P. Files moved

None.

## Q. Files renamed

None. Legacy filenames were not renamed because they are referenced by imports
and report snapshots.

## R. Files deleted

Only clearly disposable generated caches were removed:

```text
data_preprocessing/__pycache__/
dataloading/__pycache__/
losses/__pycache__/
model/__pycache__/
reports/final_report/__pycache__/
scripts/__pycache__/
testing/__pycache__/
tests/__pycache__/
trainer/__pycache__/
```

Evidence: these directories contained only Python `*.pyc` interpreter cache
files generated by imports/compilation. They are ignored by Git, reproducible,
and contain no source, data, result, manifest, report, or checkpoint.

## S. Files preserved

All source, configs, manifests, HDF5 data, checkpoints, logs, figures,
historical V1.0 artifacts, V1.1 artifacts, audits, report-local PDF assets, and
submission packages were preserved. Duplicate report/package copies are kept
because their standalone provenance/package roles are explicit.

## T. Files archived

None. No archive/move was necessary or safe in this audit pass.

## U. Known deviations

1. The 404-record reconstruction corpus and frozen 282/61/61 split are not
   the original paper's exact dataset/split protocol.
2. Text descriptions/RoBERTa anchors/SAT are disabled in V1.1.
3. `global_scene [768]` is a pooled VideoMAEv2 scene representation, not the
   paper-described temporal global representation.
4. K=8 selection and common-interval synchrony rules are explicit
   reconstruction decisions.
5. The implemented KL regularizes the affect branch only.
6. Participant/session identifiers are unavailable for leakage checks.
7. V1.0 and V1.1 are distinct protocols; their metrics must not be pooled.

## V. Known limitations

The V1.1 three-seed baseline supports a reproducible result on one frozen
split. It does not establish exact paper reproduction, statistical
significance, participant-level generalization, text/SAT contribution,
corrected V1.1 K=3/K=5 sensitivity, K=10 behavior, or V1.1 ablation effects.
K=10 needs new raw-video feature extraction because physical storage is
Kmax=8.

## W. Final supervisor presentation structure

1. State the narrow claim: audited V1.1 GECV reconstruction, not exact paper
   reproduction.
2. Show data accounting and the frozen split.
3. Walk through extraction, HDF5, masks, loader, and label mapping.
4. Walk through VIB, context/audio fusion, gamma transformer, pooling, and
   classifier using `docs/SUPERVISOR_CODE_WALKTHROUGH.md`.
5. Explain the V1.1 gamma/scene-scale correction and its zero-invariance test.
6. Present per-seed and mean ± sample-SD results without significance claims.
7. Show provenance: config/split/corpus/checkpoint hashes and audit reports.
8. End with limitations and clearly separate future K/ablation research.

The final report PDF is `reports/final_delivery/VIBE_GECV_FINAL.pdf`. The
current source/code/report reading order is README → `docs/CODE_MAP.md` →
`docs/EXECUTION_FLOW.md` → `docs/SUPERVISOR_CODE_WALKTHROUGH.md` → V1.1
aggregate/integrity audit.
