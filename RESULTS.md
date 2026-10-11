# Executed prototype results

**SIMULATION. Obstruction audio is SYNTHETIC. Not clinically validated.**

This report is generated from the saved training, TFLite evaluation and scenario logs. It is a small custom demonstration, not the official ICBHI/ESC benchmark, clinical validation, or an ESP32 measurement. No test-set-driven model revision was made after examining these results.

## Data actually used

| Original dataset | Prepared real audio segments |
|---|---:|
| icbhi | 166 |
| esc50 | 120 |
| coswara | 191 |
| musan | 78 |

Total: 555 prepared real segments. 52 preparation candidates were rejected: 15 no >=2s cycle annotated without crackle/wheeze, 35 quality missing/bad, 2 silent source.
ICBHI contributes annotated cycles, so its prepared segment count exceeds its selected recording count. Coswara cough/counting quality 1 or 2 is retained; bad or missing quality is rejected. The separate data archive includes the actual audio and manifest. No cough is synthesized.

ICBHI and Coswara participants stay within one split. ESC original folds 1–3/4/5 define train/validation/test. MUSAN is separated by recording; speaker/artist disjointness is not verified. Derived synthetic examples keep their source group. This is a deterministic custom split, not an official benchmark protocol.

## Training and model

The model has 7,344 parameters; its float32 TFLite file is 33,696 bytes. Training ran 35 epochs in 65.71 seconds on the host CPU. Early stopping restores the weights with the lowest validation loss. Seed: 20261011. Runtime versions are supplied separately.

Input: mono 16 kHz, two-second windows, 32 log-mel bands, FFT 512 and frame hop 256. Global mean/std come only from training features. The frontend is shared between training and inference. The CNN uses three convolution layers (12/20/24 filters), pooling, a 24-unit dense layer and four-way softmax. Adam learning rate 0.001 and inverse-class-frequency training weights are prototype choices.

| Split | Normal | Ambient (includes real cough) | Movable SYNTHETIC | Fixed SYNTHETIC |
|---|---:|---:|---:|---:|
| train | 342 | 825 | 342 | 342 |
| validation | 84 | 213 | 84 | 84 |
| test | 72 | 129 | 72 | 72 |

## E3: held-out windows and cough events

Window accuracy: **88.99% across 345 windows**. Windows from the same source are correlated; this is not a count of independent people. Macro F1: 0.8816.

| Class | Precision | Recall | F1 | Windows |
|---|---:|---:|---:|---:|
| normal | 0.8913 | 0.5694 | 0.6949 | 72 |
| ambient | 0.8212 | 0.9612 | 0.8857 | 129 |
| movable_synthetic | 0.9211 | 0.9722 | 0.9459 | 72 |
| fixed_synthetic | 1.0000 | 1.0000 | 1.0000 | 72 |

Confusion matrix: rows are true labels; columns are predictions in normal / ambient / movable SYNTHETIC / fixed SYNTHETIC order.

```text
 41   25    6    0
  5  124    0    0
  0    2   70    0
  0    0    0   72
```

Dedicated real-cough false-alarm result: model **0/6 (0.0%)**; baseline **0/6 (0.0%)**. Held-out real cough clips evaluated independently with alarm state reset at each clip; original clips <2 seconds zero-padded and flagged. Six cough clips are far too few to support a general reliability claim.

TFLite/Keras maximum absolute probability difference: 0.00000378. E3 mean timed TFLite invocation: 0.134 ms (**host CPU, not ESP32**). Feature extraction and audio decoding are excluded.

## E4: complete saved scenario

Scenario length: 30.168 seconds. Requested global background SNR is 30 dB. This is a mixture input, not a measured physiological value. Background mixing can affect local SNR differently in different events. Source clips were chosen by first eligible held-out ID without inspecting predictions.

Model false-alarm episodes: 0; baseline: 0. Model cough FAR: 0/1; baseline: 0/1. Unevaluable cough events: 0.

| SYNTHETIC event | Onset (s) | Model state/result | Model latency (s) | Baseline result | Baseline latency (s) | Difference (s) |
|---|---:|---|---:|---|---:|---:|
| movable SYNTHETIC | 18.168 | detected | 5.832 | missed | Unavailable | Unavailable |
| fixed SYNTHETIC | 24.168 | already_active_at_onset | Unavailable | detected | 4.582 | Unavailable |

Audio-timeline algorithmic latency: first new alarm window-end timestamp minus obstruction onset. Excludes processing and rendering; not end-to-end or ESP32 latency. Missing and already-active alarms have no numeric latency. Positive difference means the baseline was later on the audio timeline.

Mean invocation: 0.172 ms; 95th percentile: 0.320 ms (**host CPU, not ESP32**). These are saved-run measurements and will vary when rerun. No ESP32 RAM, power or timing figure has been measured or estimated.

For each detector, fraction of real cough events with an alarm already active on entry or active in a cough-overlapping analysis window that contains no obstruction. Includes boundary windows; events without an eligible window are excluded from the shared denominator.

## What these results do not establish

Normal recall is only 56.9% in this held-out set; many normal-labelled windows were predicted as ambient and six as synthetic movable obstruction. High synthetic-class scores can reflect the deliberately imposed low-pass/gain patterns rather than real obstruction physiology. The model is not noise-augmented during training. The E4 movable alarm arrives late, the baseline misses it, and the fixed event starts with an already-active model alarm. Therefore this run does not produce a jointly measurable latency advantage. The report preserves those limitations instead of selecting a different demonstration clip.

The original File 2 model and R5 rules were unavailable. The supplied prototype uses `PROPOSED_SYNTHESIS.md`, explicitly not verified against R5. Alarm threshold 0.7, three consecutive windows, one-second refractory period, RMS threshold 0.006 and three-second baseline timeout are declared input settings; they are not validated clinical thresholds.

## Journal reference

Acharya, J., & Basu, A. (2020). Deep Neural Network for Respiratory Sound Classification in Wearable Devices Enabled by Patient Specific Model Tuning. *IEEE Transactions on Biomedical Circuits and Systems*, 14(3), 535–544. https://doi.org/10.1109/TBCAS.2020.2981172. Selected through Consensus. The paper supports discussion of respiratory sound features and embedded classification; its labels and model differ from this synthetic obstruction demonstration. No published performance is reused as our result. Original dataset citations are in `DATASETS.md`.

## Originality checklist

- Template-like: methods, report structure, comparison wording and automatic tables.
- Rewrite the dataset-selection rationale, baseline justification and discussion of errors in your own voice. The numerical tables are generated measurements; do not invent personal observations.
- [STUDENT TO ADD: our own observation from run X / bench test Y]
- Run your university's plagiarism checker and disclose AI assistance according to its policy. No plagiarism or AI-detection percentage is promised.
