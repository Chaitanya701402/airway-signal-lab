# Google Colab demonstration

Use the notebook `notebooks/bundled_reconstruction_colab.ipynb` to demonstrate
the process live. It clones the public repository to obtain the bundled
participant payloads and hybrid runner, uploads the local reconstruction
builder, then runs these stages in order:

1. Decode and validate compressed float32 participant payloads.
2. Assign participants to train, validation and test before generating windows.
3. Crop two-second waveform windows and build normal, additive-noise ambient,
   movable synthetic and fixed synthetic examples.
4. Extract 32-band log-Mel features and fit normalization on training features.
5. Write an auditable source-group manifest, train E0--E3 and inspect the
   confusion matrix and learning curves.

Use the notebook's displayed split map and generated `dataset_metadata.json`
when presenting. Do not describe the generated ambient or obstruction signals
as patient recordings.
