# V2 final change manifest

This manifest records the meaningful V2 working-tree changes relative to the
protected V1 commit `cde50422b52d601525f6d694cf7f8c80e2791e67`. It is a review
aid only; V2 has **not** been committed or pushed.

## Files added

| Path | Reason |
|---|---|
| `reports/V2_PRE_AUDIT_BASELINE.md` | Captures branch/commit/status/size before V2 changes. |
| `reports/V2_GAMMA_AUDIT.md` | Records the current V1.1 gamma correction and regression result. |
| `reports/V2_FINAL_CODE_AUDIT.md` | Full repository, implementation, data, test, security, and artifact audit. |
| `reports/V2_FINAL_CHANGE_MANIFEST.md` | This final-review manifest. |
| `docs/CODE_MAP.md` | Maps every editable source module to actual callers, inputs, outputs, and research role. |
| `docs/EXECUTION_FLOW.md` | Traces the actual V1.1 call graph, tensor interfaces, selection, and outputs. |
| `docs/SUPERVISOR_CODE_WALKTHROUGH.md` | Provides an ordered technical reading guide for supervision/presentation. |

## Files modified

| Path | Reason | Scientific behavior changed? |
|---|---|---|
| `README.md` | Replaced legacy paper-style claims/obsolete commands with concise V1.1 reconstruction scope, real entry points, results, limitations, and tests. | No. |
| `docs/data_contract.md` | Labels the existing generic contract as legacy and points readers to the V1.1 contract documentation. | No. |
| `.gitignore` | Adds common raw-media/checkpoint patterns and the HDF5 corpus directory; protects large local artifacts. | No. |
| `scripts/validate_gecv_reconstruction_v1.py` | Corrects validator classification of production alias HDF5 fields; preserves legacy differences as warnings. | No; audit interpretation only. |
| `tests/gecv_reconstruction_cpu_integration.py` | Adds optional `--report` output path for non-mutating verification. | No; default behavior preserved. |
| `tests/gecv_reconstruction_pre_gpu_cpu_test.py` | Adds optional `--report` output path for non-mutating verification. | No; default behavior preserved. |
| `tests/ablation_switches_cpu.py` | Updates a stale gamma-broadcast comment and strengthens the V1.1 neutral-scene shape/value assertion. | No; test-only assertion/documentation. |

## Files moved or renamed

None. Existing names/locations are retained to preserve imports, report asset
references, and historical provenance.

## Files deleted

Only generated Python caches were removed:

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

Each contained only `*.pyc` cache files, was ignored by Git, and can be
regenerated automatically. No source, HDF5 record, manifest, checkpoint,
configuration, report, figure, package, or history was deleted.

## Files archived

None.

## Explicitly preserved material

- Protected V1 commit/history and all historical V1.0 artifacts.
- V1.1 source/configs, three-seed reports/checkpoints, final baseline package,
  figures, final PDF, and provenance.
- Frozen split and all HDF5 records.
- Historical effective-K and ablation-preparation evidence.
- Standalone report/package copies and report-local PDF source snapshots.

## Review requirements before a future commit

1. Inspect `git status --short`, `git diff --stat`, and `git diff --name-status`.
2. Confirm no raw data, checkpoints, logs, secrets, or private keys are staged.
3. Review the single environment limitation: install `seaborn` from
   `requirements.txt` before expecting legacy visualization imports to work.
4. Do not combine V1.0 and V1.1 numerical results.

## Proposed V2 commit message

```text
docs(audit): add supervisor-ready V2 execution and integrity documentation
```
