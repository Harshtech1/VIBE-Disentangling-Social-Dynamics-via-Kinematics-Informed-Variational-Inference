# V2 Pre-Audit Baseline

This document records the repository state before the V2 supervisor-ready audit and documentation pass. It is a protection record, not a scientific result.

## Protected V1 baseline

- Current branch: `main`
- Current `HEAD`: `cde50422b52d601525f6d694cf7f8c80e2791e67`
- Current `HEAD` summary: `Add GECV reconstruction implementation and validation`
- Current remote relationship: `main` and `origin/main` both point to `cde5042` at capture time.
- V1 baseline identification: the commit above is the already-published first research snapshot. V2 must be prepared in the current working tree without rewriting, rebasing, amending, resetting, force-pushing, or deleting this history.

## Capture details

- Timestamp (UTC): `2026-09-30T12:06:29Z`
- Repository working-tree size: `27G`
- Git metadata size: `5.1G`
- Data directory size: `455M`
- Checkpoints directory size: `21G`
- Logs directory size: `712K`
- Tracked paths at capture: `99`
- Tracked modified/deleted paths at capture: `0`
- Untracked top-level report paths at capture: `14`

The untracked paths are reports and presentation artifacts from prior work, including `reports/ablations/`, `reports/dataset_audit/`, `reports/figures/`, `reports/final_report/`, `reports/project_audit/`, `reports/robustness/`, and `reports/submission/`. They are preserved as V2 working material and are not treated as disposable clutter.

## Major directories present

`checkpoints/`, `config/`, `data/`, `data_preprocessing/`, `dataloading/`, `docs/`, `images/`, `logs/`, `losses/`, `model/`, `reports/`, `scripts/`, `testing/`, `tests/`, and `trainer/`.

## Safety statement

No V1 commit, dataset, frozen split, historical report, checkpoint, or Git history was modified to create this baseline record. Subsequent V2 changes must be reviewed as uncommitted working-tree changes before any commit is considered.
