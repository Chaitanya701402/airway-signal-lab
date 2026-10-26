# Phase 9: verified repeat and inference diagnostics

Executed 2026-10-26 on Windows CPU, Python 3.13.5, TensorFlow 2.21.0, NumPy 2.1.3.

## Engineering decisions

The supplied submission directory had no Git repository. Work was performed in a new clone of `Chaitanya701402/airway-signal-lab`, starting from main at bd0c931. Existing submission files were preserved. Main already contained measured reconstruction results, superseding the earlier blocked-experiment assessment.

The feasible experiment was a fresh four-model repeat on the existing eight-source reconstruction. The reconstructed features and labels exactly matched all six arrays supplied locally. Source split sizes are 288/144/144 (4/2/2 source participants). The builder fits its scalar mean and standard deviation only on training features; training moments were also checked, though moments alone do not prove provenance.

The builder now emits embedded groups and per-row feature hashes. The runner rejects malformed one-hot labels, empty or inconsistent features, missing manifests, mismatched row counts or label order, group leakage, incorrect feature hashes, mismatched embedded groups and exact duplicates across splits. Legacy manifests remain usable with an explicit unverified feature-binding status. Hash checks do not authenticate source identities or detect near-duplicate audio.

The app now exposes class prediction counts, output concentration, normalized entropy, top-two margin and median/p95 model latency. Fresh inference exports retain these diagnostics in metadata. They describe outputs and timing, not calibrated certainty or measured accuracy. Diagnostics normalize small probability-sum quantization errors without altering the predictions used by alarms. No deployed model or alarm threshold was replaced.

New experiment outputs preserve predictions, training histories, all four confusion matrices, dataset/manifest hashes, software versions and run settings. Nonempty run directories are rejected to protect old evidence. The notebook now uses the scripts already in the clone and a separate output directory.

## Executed experiment

Seed 20261026; 35 epochs maximum; batch size 32; Adam 0.001; validation-loss checkpoints; patience 8. All models completed 35 epochs and checkpoint reload/evaluation. No test-set tuning was performed in this task.

| Variant | Accuracy | Macro F1 | Selected epoch | Parameters |
| --- | ---: | ---: | ---: | ---: |
| E0 CNN | 87.50% | 0.8667 | 35 | 7,344 |
| E1 CNN + GRU | 88.19% | 0.8750 | 33 | 13,104 |
| E2 CNN + attention | 92.36% | 0.9218 | 33 | 7,369 |
| E3 CNN + GRU + attention | 93.75% | 0.9365 | 33 | 13,137 |

E3 exceeded the contemporaneous E0 by 0.06984 macro F1. The earlier saved run favored E2 (0.9861). This repeat changed seed and batch size (earlier documented command: 16), so it is not an isolated seed-sensitivity test. Runtime/library differences may also matter. These are correlated synthetic transformations of eight breathing sources; only two participants contribute test windows. The shared test split has now been inspected repeatedly and is not an untouched final evaluation set. Neither statistical superiority nor clinical generalization is established. Historical 88.99% accuracy / 0.8816 macro F1 is a different evaluation and is not a valid comparator.

## Verification

- `python -m unittest discover -s tests -v`: 15 tests passed, including adversarial malformed-data and diagnostic checks.
- All four architecture forward passes: finite (2,4) probability outputs summing to one.
- Streamlit AppTest initial render: zero exceptions, including the diagnostics expander.
- Real deployed TFLite scenario: 113 windows; model SHA-256 `73ad9be43d03e8fbd38c0f4e32140f1890ae570ac92778263ad9f48966800e30`. Used the existing TensorFlow Lite fallback because ai-edge-litert was unavailable locally. Its deprecation warning does not establish LiteRT deployment compatibility.
- Existing reconstruction validator passed; zero cross-split exact duplicates; train moments consistent with scalar standardization.
- All four accuracy/macro-F1 values independently recomputed from saved predictions and matched reported metrics.
- Fresh feature and label arrays exactly equal the supplied reconstruction arrays.
- No failed tests. TensorFlow emitted a nonfatal dataset-attribute compatibility warning during training.

Run application/architecture smoke checks with `python tests/smoke_application.py`. It executes actual inference and requires the app and TensorFlow dependencies. Unit tests need NumPy only.

## Evidence and reproduction

This directory retains aggregate results, per-class reports and confusion matrices, histories, software/run metadata, data-builder metadata, smoke report and a CSV of all held-out predictions. Row indices refer to split-local test order in the generated manifest. The run metadata records the original local paths; reproduce with the README commands using a fresh output directory. Checkpoints, feature arrays and generated figures remain local in `../phase9_local` relative to the repository, rather than being added to the deployment repository. Existing bundled public recordings were used locally; no recordings or personal session files were added to the commit.

## Remaining work

Use more independent recordings and a new final held-out set, run a preregistered multi-seed protocol with identical hyperparameters, evaluate realistic ambient/cough sources and gain/noise shifts, and assess group-level uncertainty. Reproduce the historical full-corpus baseline only when its data and protocol are available. Source identity and near-duplicate auditing need stronger external provenance. No clinical or ESP32 validation was attempted. The manuscript's existing results were preserved; this report supplies a separate, qualified engineering finding for later inclusion.
