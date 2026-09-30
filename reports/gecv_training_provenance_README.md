# VIBE-GECV-RECON-v1.0 training provenance

This package describes the completed 30-epoch reconstruction experiment without rerunning any computation.

## Directly verified

- The frozen configuration, split manifest, best checkpoint, and source-corpus hashes are recorded in `gecv_training_run_manifest.json`.
- `checkpoint_last.pth` records epoch 30; `model_best.pth` is tensor-identical to the epoch-9 checkpoint and differs from epoch 30.
- The trainer code reloads `model_best.pth` before its only held-out test evaluation.
- The frozen train/validation/test lists are disjoint, exclude the smoke-only clips and failed `videoH116`, and their HDF5 labels match Positive=0, Neutral=1, Negative=2.
- The TensorBoard event file records validation metrics for epochs 1--30 and final test metrics. They are transcribed verbatim, at TensorBoard precision, in `gecv_training_metrics_epoch1_30.json`.
- The final confusion matrix sums to 61 and reproduces the reported held-out accuracy, macro-F1, and weighted-F1.

## Unavailable runtime detail

The exact terminal transcript was not found locally, so `gecv_training_terminal.log` was intentionally not created. The following cannot be recovered from the existing artifacts: per-batch losses, gradient-norm history, exact optimizer-step messages, full learning-rate printout, and raw terminal warnings/errors.

The absence of that transcript is a provenance limitation, not evidence of failure: the epoch-30 checkpoint contains finite model and optimizer state, and the TensorBoard event stream includes the completed validation and test metrics.

## Scientific scope

This is a **VIBE-GECV-RECON-v1.0 reconstruction result**. It is not the official VIBE paper result, does not use the official paper GECV split, disables text and `L_sat`, and uses the approved custom reconstruction protocol. It must not be presented as directly comparable to the paper's reported metric.
