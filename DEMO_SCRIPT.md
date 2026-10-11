# FILE 4 — two-minute presentation script

Read the permanent screen banner aloud once: **SIMULATION. Obstruction audio is SYNTHETIC. Not clinically validated.** The following time marks are a suggested speaking schedule, not measured audio event timings. The supplied app already loads the executed prototype model run and saved audio. Double-click `Start_demo.cmd`, then select Play. Open `RESULTS.md` before rehearsal; measured values are already filled from logs.

| Presentation time | Action and suggested words |
|---|---|
| 0:00–0:20 | Show the scene and selected saved-audio timeline. “This is a software demonstration of our audio classifier and alarm logic. The real breathing, noise and cough clips have source records. The two obstruction classes use synthetic audio. The animated airway is an illustration, not a validated airflow calculation.” |
| 0:20–0:40 | Start normal breathing playback and point to waveform, scrolling spectrogram and four model output bars. “These panels share the same playback timeline. The probability bars come from the loaded model. The visual scene helps us explain the event; it does not measure airway shape.” |
| 0:40–1:00 | Move to the real cough event. “Cough belongs to ambient. We use it as a difficult negative example. We count an obstruction alarm that is active on entry or during an eligible cough-overlapping window. Both detectors use the same evaluable cough events.” State the observed state only after seeing the log. |
| 1:00–1:20 | Select an available real ambient/speech clip and adjust the scenario SNR. “This control changes the audio mixture. We rerun the model on that mixture and inspect its actual output. We do not assume it will remain correct.” Pause after a change if inference needs time. |
| 1:20–1:40 | Move the synthetic movable-obstruction onset, then advance to synthetic fixed obstruction. Show the model alarm lane and fixed-timeout baseline lane. “The lanes compare the same saved timeline and declared rules. The displayed difference is computed from logged alarm times; an absent detection remains undetected.” |
| 1:40–2:00 | Open the actual result summary. “Inference duration is measured on the host CPU, not ESP32. Our demo result is [STUDENT TO ADD: our own observation from run X / bench test Y]. We still need board measurements and appropriate external validation. This presentation does not establish clinical performance.” |

Before presenting, rehearse using a saved run and note its model hash, data-manifest identity, scenario settings and run ID. Keep an exported animation available as a fallback. The exported animation must come from that run's audio and logs. If any required asset is missing, show the clearly labelled layout preview and say that inference has not run; do not read placeholder bars or animation states as predictions.

Automatically recorded in `RESULTS.md` and the app: accepted/rejected source clip counts, evaluated cough-event denominator, false-alarm count, model and baseline cough false-alarm rates, detection latency or undetected status, model-versus-baseline difference when both are detected, and per-window host inference duration. Detection latency is calculated on the audio timeline; it excludes processing and rendering time. Do not promise that the model beats the baseline.

## Originality checklist

- Template-like sections: the speaking schedule, introductory explanation and closing limitations.
- Rewrite the introduction and final interpretation in your own voice. Replace the observation placeholder only after examining your saved run. Add **[STUDENT TO ADD: our own observation from run X / bench test Y]** for any new unsupported claim.
- Cite research and dataset sources in the accompanying report. Run your university's plagiarism checker on submitted writing and disclose AI assistance according to its policy.
