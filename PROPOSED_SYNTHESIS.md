# Proposed synthetic obstruction recipe

This recipe makes two **synthetic signal classes** for an educational software demonstration. The original R5 instructions were absent. This document records the assumptions used to finish the demonstration; it does not claim compliance with that missing specification.

The signal classes are not validated examples of a real movable or fixed airway obstruction. Their names describe the intended scenario in the animation. Recognising these transformations does not establish clinical diagnostic ability.

## Source preparation

Use ICBHI 2017 cycles annotated with neither crackles nor wheezes and lasting at least two seconds. This annotation is not proof of health or normal respiratory physiology. Use the recording's participant ID for the train/validation/test grouping before any transformation. All derivatives of a participant stay in the same split.

Convert recordings to mono and 16 kHz using the same preprocessing implementation used by the app. Level-adjust real source segments toward a digital RMS of 0.04, with peak protection. This is a software amplitude convention, not a calibrated sound-pressure measurement.

Training uses two-second windows. Only real breathing windows are used to construct the synthetic obstruction classes. **Cough waveforms are never synthesized.** Real coughs are retained inside ambient, with a separate cough tag. Short real cough clips may be padded with silence for a complete model window; this is recorded explicitly.

## Synthetic movable-obstruction surrogate

Apply a third-order low-pass filter with a cutoff drawn uniformly from 700–1,600 Hz. Apply an amplitude envelope based on a cubed sinusoidal modulation at 2.5–5.5 Hz, with a randomly chosen phase. The envelope gain varies from 0.035 to 0.785.

These arbitrary signal-processing ranges create fluctuating attenuation that the software can learn. They are **not** measured airway dimensions, movement frequencies, obstruction severity or physiological parameter estimates.

## Synthetic fixed-obstruction surrogate

Apply a third-order low-pass filter with a cutoff drawn uniformly from 350–900 Hz. Apply a constant gain drawn uniformly from 0.025–0.08. This deliberately produces a consistently attenuated signal with a changed spectrum.

A model may identify this class mainly from its low level and filtering signature. Real quiet breathing, microphone distance, sensor contact failure or other recording changes may resemble it. A high score on this synthetic class would not resolve those confounders.

## Reproducibility and evaluation

The fixed random seed is an input setting, not a result. Each generated training window records its parent source, group, split and sampled transformation parameters in `experiment/window_manifest.json`. The app's synthetic demo clips record their parameters in `config.ready.json`.

All four source classes receive the documented real-source preparation; the training windows receive a common random gain range of 0.7–1.3 before transformation. No artificial coughs are generated. The current training procedure does not claim robust noise augmentation: adjustable added noise in the live demo is a separate stress test whose measured results may be worse.

Use validation loss for training checkpoint selection. Report the held-out test once after training and retain its errors. The presented timeline selects source clips deterministically before consulting their predictions. Do not replace them with easier clips to imply better performance.

The fixed-timeout comparator uses audio RMS as an activity proxy. Cough or background sound can reset it. It is not a validated breath detector. The threshold, dwell and timeout are declared software inputs, with no claim that they are clinically suitable.

## Originality checklist

- Template-like: recipe structure, implementation description and reproducibility steps.
- Rewrite the motivation for the synthetic transformations, baseline choice and interpretation of actual errors in your own voice.
- [STUDENT TO ADD: our own observation from run X / bench test Y]
- Explain why this restricted demonstration cannot validate real obstruction detection. Do not invent observations from a bench or patient study.
- Run your university's plagiarism checker and disclose AI assistance per policy. No plagiarism or AI-detection percentage is promised.
