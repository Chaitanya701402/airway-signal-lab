# Airway Signal Lab — Streamlit deployment

**SIMULATION. Obstruction audio is SYNTHETIC. Not clinically validated.**

This repository contains the running presentation app, actual trained TFLite model, saved real source clips, synthetic surrogate clips and measured reference logs. No manual data entry is needed. The 3D scene illustrates known scenario labels; it is not reconstructed anatomy or a validated airflow simulation.

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
