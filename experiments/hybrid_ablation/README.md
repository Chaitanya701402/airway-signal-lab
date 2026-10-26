# Hybrid ablation experiment output

This directory is reserved for actual, reproducible output from
`experiments/hybrid_ablation.py`. It intentionally contains no reported ablation
numbers until the original prepared source-split dataset is provided and run.

Required input is a `.npz` file containing `X_train`, `y_train`, `X_val`,
`y_val`, `X_test`, and `y_test`, plus a CSV manifest with `split`, `group_id`,
and `class_name`. The fixed class order is normal, ambient,
movable_synthetic, fixed_synthetic. A source group must occur in only one
split. Features must be 32-band log-Mel tensors normalized from training data
only.

After a completed run, this directory will contain the requested ablation table,
detailed reports, E0/E3 confusion matrices, training curves, metadata and
separate experimental checkpoints. These results must be interpreted as a
custom synthetic-surrogate experiment, not clinical validation.
