# Airway Signal Lab — Streamlit deployment

**SIMULATION. Obstruction audio is SYNTHETIC. Not clinically validated.**

This repository contains the running presentation app, actual trained TFLite model, saved real source clips, synthetic surrogate clips and measured reference logs. No manual data entry is needed. The 3D scene illustrates known scenario labels; it is not reconstructed anatomy or a validated airflow simulation.

## New measured hybrid-model experiment

The repository now also contains a reproducible, source-separated **bundled reconstruction ablation**. It compares four models using the same frozen train/validation/test source groups, seed, features and validation-loss checkpoint selection:

| Variant | Architecture | Held-out accuracy | Held-out macro F1 |
| --- | --- | ---: | ---: |
| E0 | CNN | 76.39% | 0.7560 |
| E1 | CNN + GRU | 79.17% | 0.7871 |
| E2 | CNN + temporal attention | **98.61%** | **0.9861** |
| E3 | CNN + GRU + temporal attention | 88.19% | 0.8759 |

The result is a controlled engineering ablation, not a clinical result or a numerical replacement for the historical 345-window TFLite evaluation: it uses eight bundled ICBHI breathing sources, with controlled derived ambient and obstruction conditions. The experiment artifacts are directly available:

- [Full measured results and scope](experiments/bundled_reconstruction/RESULTS.md)
- [Split and feature validation report](experiments/bundled_reconstruction/VALIDATION_REPORT.md)
- [Cleaning and data-synthesis pipeline](experiments/bundled_reconstruction/pipeline_overview.png)
- [E0--E3 comparison chart](experiments/bundled_reconstruction/ablation_comparison.png)
- [E2 held-out confusion matrix](experiments/bundled_reconstruction/confusion_matrix_E2.png)
- [Google Colab workflow](notebooks/bundled_reconstruction_colab.ipynb)

A fresh CPU repeat (seed 20261026, batch size 32) found E3 highest at 93.75% accuracy and 0.9365 macro F1; the table above is the earlier run. Rankings differ across the two configurations, so neither establishes a reliably superior architecture. See [Phase 9 results, diagnostics and verification](experiments/phase9_repeat/REPORT.md).

To reproduce into a fresh directory, run `python experiments/build_bundled_reconstruction.py --output prepared_data/repeat`, then `python experiments/hybrid_ablation.py --data prepared_data/repeat/prepared_source_split.npz --manifest prepared_data/repeat/window_manifest.csv --output prepared_data/repeat/ablation --seed 20261026 --epochs 35 --batch-size 32`. Run `python experiments/validate_bundled_reconstruction.py --experiment-dir prepared_data/repeat`. Existing datasets and nonempty run directories are protected against overwriting.

Deploy through Streamlit Community Cloud using repository branch `main`, entrypoint `app.py`, and Python **3.13**. No secrets are needed by this app. Hosting uses the official LiteRT interpreter to execute the trained model; TensorFlow training and the full prepared dataset remain in the separate local project package. Local startup: `python -m pip install -r requirements.txt` followed by `python -m streamlit run app.py`.

The initial screen replays a measured Windows desktop run. Select **Run actual TFLite model** to measure a fresh run on the hosting server. Both use host CPU, not ESP32. Noise type, requested SNR and movable onset are adjustable. Complete-run downloads are generated per session and downloaded as ZIP files; the hosting server is not a durable store for experiments.

RESULTS.md includes actual errors and limitations. Six held-out cough clips do not establish reliability. The missing original R5 specification was not replaced by a claim of compliance: PROPOSED_SYNTHESIS.md explicitly defines a provisional surrogate recipe. Cough remains a real ambient confounder and is never synthesized.

Source-specific licences and journal citations are in DATASETS.md and source_credits. ESC-50 is non-commercial CC BY-NC 3.0. Preserve original credits; a mirror's licence does not replace original terms. This is an educational research presentation. The full prepared-data archive is not uploaded to this deployment.

The checked-in `demo_assets.zip` contains the model, real audio, saved run and original source licences. The app checks its SHA-256 and unpacks it automatically on first startup; no manual data entry or download is needed.

The machine-readable `packages.txt` accepts package names only; its template-like dependency declaration is covered by this checklist.

# Originality checklist

- Template-like: README structure, deployment commands and presentation narration.
- Rewrite the introduction, dataset-selection rationale, baseline justification and interpretation in your own voice.
- [STUDENT TO ADD: our own observation from run X / bench test Y]
- Run the university plagiarism checker and disclose AI assistance under its policy.
