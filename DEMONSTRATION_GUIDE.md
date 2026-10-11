# Demonstration for your professor

Open https://airway-signal-lab.streamlit.app/ before the presentation and let the model finish its first run. Keep the laptop volume on. The audio play buttons and animated trace controls are separate.

## Two minute walkthrough

1. Start with Clean in Noise type. Point to the permanent simulation banner. Explain that the breathing sources are real held-out ICBHI participants, while both obstruction conditions are generated signal transformations.
2. Open Breathing source participant and show the eight options. Pick a different participant. Selection automatically runs the actual TFLite model and replaces the audio and trace. The source caption identifies the recording and the saved normal-recall denominator.
3. Play Normal breathing, then Real cough, then one SYNTHETIC obstruction tab. Explain that cough is an ambient confounder, not a separate output class. The cough comes from a separate Coswara donor; this is a constructed timeline.
4. Press Play trace. Show the waveform, spectral features, four probabilities, model alarm and fixed-timeout comparator. Inference timing is host CPU only. Explain that a probability is a model output, not confirmed clinical certainty.
5. Choose speech or music, set the desired SNR and press Run actual TFLite model. Discuss the resulting misses or false alarms honestly. Changed controls display a warning until the model runs again.
6. Open Research evidence and failure analysis. State that the original held-out window accuracy was 88.99%, but loudness equalisation reduced it to 64.93%. The paper studies this weakness rather than claiming clinical reliability.

## Questions you may be asked

**Can I try a different patient?** Yes. Eight held-out breathing-source participants are selectable. They are not eight confirmed obstruction cases.

**Where are the obstruction recordings from?** They are explicitly synthetic low-pass and gain/modulation derivatives. No real obstruction dataset was collected for this project.

**How many patients were evaluated?** The held-out ICBHI subset has eight breathing participants. The 345 windows include ambient sources and synthetic derivatives. The 312 scenarios reuse sources; neither number is an independent patient count.

**Are coughs detected as obstruction?** E3 recorded 0/6 cough clips with an alarm. The expanded E4 grid recorded alarms during 12/312 repeated real-cough events, using only six distinct cough clips. This sample is too small for a population reliability claim.

**What is new?** The proposed contribution is the reproducible audit combining participant diagnostics, amplitude controls, real-cough confounders and state-aware alarm accounting. CNN airway monitoring already exists; this is not the first such CNN or a new physiological simulator.

**Is it running on an ESP32?** No. Inference runs on the hosting server or laptop CPU. No ESP32 timing, power or memory result is claimed.

**Why does performance change between participants?** The public subset contains different source recordings and devices, and normal recall varies. The results expose this variation. The experiment does not isolate its cause.

## Backup on this Windows laptop

Extract the portable ZIP. Use the same Python environment already used for the project, then double-click `start_demo.cmd`. It opens a local Streamlit server; visit the displayed localhost address. The package includes source audio, model, registry and research evidence and needs no audio downloads. Python and the app dependencies must already be installed for an offline launch.

If the hosted page still shows the previous interface, refresh once and look for Breathing source participant near the top. Use the local backup if internet access is unreliable.

## Originality checklist

The narration and answers are presentation templates. [STUDENT TO ADD: our own observation from run X / bench test Y]. Describe only behaviour you personally observed when rehearsing. Add your own reasoning about limitations, run the university plagiarism checker on written work and disclose AI assistance according to policy.