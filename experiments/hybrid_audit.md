# Hybrid experiment audit (2026-10-26)

## Repository state

- Branch inspected: `main`; remote: `Chaitanya701402/airway-signal-lab`.
- The working tree already contained untracked deployment assets, source-credit files, saved exports and experiment JSON. They were not removed or overwritten.
- The Streamlit app, `engine.py`, `config.ready.json`, `demo_assets.zip`, deployment TFLite model and historical results were preserved.

## Confirmed deployment protocol

- Fixed order: `normal`, `ambient`, `movable_synthetic`, `fixed_synthetic`.
- Cough is a real ambient confounder, not a fifth output class.
- `config.ready.json` declares mono 16 kHz, 2 s windows, log-Mel with 32 bands, FFT 512, hop 256, `center:false`, and global scalar normalization (mean `-64.69202423095703`, standard deviation `26.35917091369629`).
- The checked-in inference model is `assets/trained_demo.tflite` (33,696 bytes). It has no Keras checkpoint or original model-training source alongside it.
- Historical report only: a 7,344-parameter CNN trained for 35 epochs, held-out accuracy 88.99% and macro F1 0.8816. These are not reproduced by this change.

## Located material

- Deployment audio/model assets and a saved scenario replay under `assets/` and `exports/ready_run/`.
- Historical training/evaluation summaries under `experiment/`: `training_history.json`, `e3_results.json`, `e4_results.json`, and `demo_selection.json`.
- Dataset metadata, annotations and licences under `source_credits/`, but no original prepared source-audio archive.
- `demo_assets.zip` is a deployment-asset bundle; it is not a complete training archive.

## Missing artifacts that block a genuine ablation result

1. Original training code and its exact 7,344-parameter CNN definition.
2. Prepared training/validation/test feature arrays and labels.
3. The `window_manifest.json` / source-group split manifest describing parent recording/participant, derived examples, transform parameters and split.
4. Original Keras checkpoint/weights and exact feature normalization-fitting code.
5. Raw source recordings required to run a source-separated noise-robustness experiment.

`exports/ready_run/signals.npz` is a single saved demonstration timeline, not a labeled train/validation/test corpus and must not be repurposed for training.

## Resulting implementation status

`experiments/hybrid_ablation.py` and `notebooks/airway_hybrid_ablation.ipynb` are a runnable, guarded Colab experiment package. The notebook accepts the missing source-split artifacts through Drive or upload, validates arrays and group integrity, trains E0-E3, and creates outputs only from actual training. It does not create substitute data or report invented ablation numbers.

The optional noise experiment remains blocked until raw recordings and reliable group metadata are supplied.
