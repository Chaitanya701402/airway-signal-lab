# Bundled reconstruction ablation

## What was trained

This is a new, small reconstruction experiment created from the eight bundled
ICBHI breathing clips, not a reproduction of the historical prepared-data
experiment in `RESULTS.md`. Participants 105, 111, 125 and 131 are training
sources; 147 and 153 are validation sources; 207 and 216 are test sources.
Every breath window and every derivative of it remains in its assigned split.

For each source, 18 two-second waveform crops produced a normal example, a
synthetic additive-noise ambient example, a movable synthetic surrogate, and a
fixed synthetic surrogate. Log-Mel features use 32 bands, FFT 512, hop 256 and
train-only global normalization. The final split sizes are 288 / 144 / 144.

The ambient and obstruction labels are generated transformations. These results
measure discrimination of this constrained reconstruction and cannot be
compared numerically with the deployed model's 345-window evaluation, used as
clinical evidence, or used to replace the deployed TFLite model.

## Held-out results

| Variant | Architecture | Accuracy | Macro F1 | Parameters | Mean Keras-only latency |
|---|---|---:|---:|---:|---:|
| E0 | CNN | 76.39% | 0.7560 | 7,344 | 8.41 ms |
| E1 | CNN + GRU | 79.17% | 0.7871 | 13,104 | 57.57 ms |
| E2 | CNN + temporal attention | **98.61%** | **0.9861** | 7,369 | 20.99 ms |
| E3 | CNN + GRU + temporal attention | 88.19% | 0.8759 | 13,137 | 53.36 ms |

E2 is the best single-seed result on the fixed held-out participant split. Its
macro-F1 exceeds E0 by 0.2301, while adding only 25 parameters. The held-out
set has 144 correlated windows from two participants, so this apparent gain is
an experiment finding to investigate with more source recordings, not a robust
claim of superiority.

E2's class recalls are normal 36/36, ambient 36/36, movable synthetic 35/36
and fixed synthetic 35/36. See `ablation/confusion_matrix_E2.png` and
`ablation/training_curves.png` for the actual measured figures. The full
per-model reports, confusion matrices, checkpoints and run metadata remain in
`ablation/`.

## Presentation wording

Say: "On a new eight-source bundled reconstruction split, temporal attention
gave the highest held-out macro F1. The split is source-separated, but the data
are small and the ambient/obstruction classes are synthetic transformations, so
we treat it as an ablation result rather than clinical performance."
