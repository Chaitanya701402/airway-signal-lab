# Implementation delta: initial repository vs. current work

## Initial repository state

The repository was a Streamlit deployment/demo package. It contained:

- `app.py` and `engine.py` for replaying saved audio and invoking the existing
  four-class TFLite model.
- `assets/trained_demo.tflite`, saved scenario clips, configuration and a
  historical `RESULTS.md` report.
- Dataset licences, source annotations and historical JSON summaries, but no
  original training script, Keras checkpoint, prepared train/validation/test
  arrays, or source-group training manifest.
- A local dependency definition that included an invalid
  `librosa==1.0.0` pin. The machine running Streamlit also lacked `soundfile`.

The initial code did not contain a reproducible E0/E1/E2/E3 hybrid-ablation
runner or a Colab workflow for it.

## Work added or changed

### Local runtime repair

- Installed the missing audio/inference runtime dependencies in the local
  Python 3.10 environment during diagnosis.
- Corrected the project dependency pin to `librosa==0.11.0` in
  `requirements.txt`; `librosa==1.0.0` is not published for this environment.

### Research audit and safeguards

- Added `experiments/hybrid_audit.md`, documenting confirmed frontend details,
  deployment/historical artifacts and exact missing training artifacts.
- Added a blocked execution report and machine-readable metadata. No scores,
  confusion matrices, or improvement claims were created because valid
  training data is absent.
- Preserved the fixed output order: `normal`, `ambient`,
  `movable_synthetic`, `fixed_synthetic`. Cough remains an ambient confounder;
  synthetic obstruction labels are not clinical validation.

### Hybrid-ablation implementation

- Added `experiments/hybrid_ablation.py`, which defines controlled E0 CNN,
  E1 CNN+GRU, E2 CNN+temporal-attention and E3 CNN+GRU+attention models.
- Added checks for finite feature tensors, fixed label encoding, 32-band input,
  source-group overlap, output shape/probability behavior, and validation-loss
  checkpoint selection.
- Added actual-result export code for metrics, reports, confusion matrices,
  training curves, metadata and separate experimental checkpoints once the
  original data is supplied.
- Added the Colab notebook `notebooks/airway_hybrid_ablation.ipynb` for Drive
  mounting/upload, GPU detection, artifact validation, training and ZIP export.

### Repository hygiene

- Updated `.gitignore` to exclude generated hybrid checkpoints, plots,
  result files and a local `prepared_data/` directory.
- Did not modify `app.py`, `engine.py`, the deployed TFLite model, existing
  assets, historical results, deployment configuration, or source-credit data.

## Current experiment status

Implementation and local static/import checks are complete. Actual E0-E3
training remains blocked until the original `prepared_source_split.npz` and
`window_manifest.csv` are provided. The saved `exports/ready_run/signals.npz`
file is a demo timeline and was deliberately not repurposed as training data.
