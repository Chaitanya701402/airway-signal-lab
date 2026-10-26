# Hybrid ablation execution report

## Execution status

No E0, E1, E2, or E3 training run was executed. This is not a partial model
result: the required source-split training corpus is absent, and local Python
3.10 has neither TensorFlow/Keras nor an available NVIDIA runtime (`nvidia-smi`
is not installed). The repository contains only
`exports/ready_run/signals.npz`, which is a saved Streamlit demonstration
timeline. It has no verified train/validation/test labels or source-group
provenance and must not be converted into a training set.

The repository has historical E3 numbers in `RESULTS.md` and
`experiment/e3_results.json`, but no original training code or corpus from
which to reproduce them. They are historical references, not results of this
hybrid ablation and are intentionally excluded from the table below.

## Measured results

No numerical comparison is available. `ablation_results.csv` has its schema
only; `detailed_reports.json` records the blocked state. No confusion-matrix
or training-curve image exists because no model was trained.

| Model | Accuracy | Macro-F1 | Macro-F1 change vs E0 | Normal recall | Parameters | Model size | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | Not run | Not run | Not applicable | Not run | Not measured | Not measured | Not measured |
| E1 | Not run | Not run | Not applicable | Not run | Not measured | Not measured | Not measured |
| E2 | Not run | Not run | Not applicable | Not run | Not measured | Not measured | Not measured |
| E3 | Not run | Not run | Not applicable | Not run | Not measured | Not measured | Not measured |

Therefore, no model has the highest macro-F1 or normal recall, and no E3-versus-E0 improvement, cost, seed consistency, or edge-deployment recommendation can be calculated.

## Reliability and leakage status

Source-group separation, duplicate/overlapping-window checks, identical test
sets and train-only normalization cannot be verified without the original
prepared arrays plus manifest. No claim of leakage-free generalization is made.

## Exact next steps for Colab

1. Supply the original `prepared_source_split.npz` containing `X_train`,
   `y_train`, `X_val`, `y_val`, `X_test`, `y_test`.
2. Supply the original `window_manifest.csv` with at least `split`, `group_id`,
   and `class_name`. Every source group must have one split only.
3. In Google Colab, open `notebooks/airway_hybrid_ablation.ipynb`, select a GPU
   runtime if available, mount Drive or upload both files, and set `DATA_NPZ`
   and `MANIFEST_CSV`.
4. Run the notebook. It validates the supplied artifact, trains all four fixed
   variants with validation-loss checkpointing, and generates the requested
   metrics, plots and result package from actual held-out predictions.

Any results remain a custom four-class synthetic-surrogate experiment: ambient
includes real cough, and synthetic obstruction classification is not clinical
obstruction detection or an official ICBHI benchmark.
