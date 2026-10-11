# FILE 4 — separate dataset and source register

Status: **ICBHI 2017, ESC-50, MUSAN and Coswara subsets were downloaded and used for the executable prototype.** Actual retained/rejected counts are in `RESULTS.md` and the separate prepared-data archive. Its `data_manifest.csv` lists every prepared recording, source hash, split, quality decision and duration. COUGHVID and UrbanSound8K are supported alternatives, not data used in this run.

The four model classes are normal breathing, **synthetic movable obstruction**, **synthetic fixed obstruction**, and ambient. Real cough is an ambient hard-negative confounder. Never synthesize cough. These source datasets do not establish movable/fixed obstruction labels. R5 was not supplied. The automatically generated obstruction clips use the explicitly provisional recipe in `PROPOSED_SYNTHESIS.md`; they are not clinical obstruction recordings or verified R5 outputs.

## Source register

| Source and intended role | Original location | Licence / verification | Citation |
|---|---|---|---|
| ICBHI 2017: annotated real respiratory recordings | [Original challenge database](https://bhichallenge.med.auth.gr/node/50) | Original site states freely available for research. A named standard data licence was **not verified**; record the original terms before redistribution. Do not inherit a mirror's licence claim. | Rocha, B. M., et al. (2019). An open access database for the evaluation of respiratory sound classification algorithms. *Physiological Measurement*, 40, 035001. [DOI](https://doi.org/10.1088/1361-6579/ab03ea). |
| ESC-50: environmental ambient clips | [Author's repository](https://github.com/karolpiczak/ESC-50) | **CC BY-NC 3.0; non-commercial.** The ESC-10 subset has a different, CC BY licence. Preserve the repository's clip attributions. | Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification. *ACM Multimedia*, 1015–1018. [DOI](https://doi.org/10.1145/2733373.2806390). |
| UrbanSound8K: optional urban ambient clips | [Official dataset](https://urbansounddataset.weebly.com/urbansound8k.html), [conditions](https://urbansounddataset.weebly.com/) | **CC BY-NC 3.0; non-commercial.** Preserve `FREESOUNDCREDITS.txt`. | Salamon, J., Jacoby, C., & Bello, J. P. (2014). A Dataset and Taxonomy for Urban Sound Research. *ACM Multimedia*. [Official citation](https://urbansounddataset.weebly.com/). |
| MUSAN: speech, music and noise | [OpenSLR 17](https://www.openslr.org/17/) | **CC BY 4.0**, as stated by the original resource. Retain attribution metadata. | Snyder, D., Chen, G., & Povey, D. (2015). MUSAN: A Music, Speech, and Noise Corpus. [arXiv:1510.08484](https://arxiv.org/abs/1510.08484). This is a dataset reference, not a journal article. |
| COUGHVID v3: real cough confounders | [Original Zenodo v3](https://zenodo.org/records/7024894) | Dataset licence **pending confirmation from the original record**: the available text did not expose the Rights value. Do not substitute the journal article's licence for the audio licence. | Orlandic, L., Teijeiro, T., & Atienza, D. (2021). The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms. *Scientific Data*, 8, 156. [DOI](https://doi.org/10.1038/s41597-021-00937-4). Dataset DOI: [10.5281/zenodo.7024894](https://doi.org/10.5281/zenodo.7024894). |
| Coswara: alternative real cough confounders | [Original IISc repository](https://github.com/iiscleap/Coswara-Data) | **CC BY 4.0**, explicitly stated by the repository. | Bhattacharya, D., et al. (2023). Coswara: A respiratory sounds and symptoms dataset for remote screening of SARS-CoV-2 infection. *Scientific Data*, 10, 397. [DOI](https://doi.org/10.1038/s41597-023-02266-0). |

Kaggle mirrors may be a download convenience, but source identity, citation and licence must remain tied to the original dataset. The automatic acquisition uses original GitHub repositories for ESC-50 and Coswara, the Kaggle `vbookshelf/respiratory-sound-database` convenience mirror for ICBHI, and the Hugging Face `FluidInference/musan` and `corypaik/musan` mirrors for MUSAN. Exact URLs/revisions and actual hashes are retained in `download_manifest.json`. The original ICBHI HTTPS certificate was expired at acquisition; certificate verification was not disabled. Mirror licence labels do not override original terms.

## Dataset checks and label limits

- **ICBHI:** full release expectation is 920 audio recordings. Cycle annotations describe the presence of crackles and wheezes. Neither the absence of those sounds nor a disease label proves normal airway function. The executed prototype selects cycles with both flags zero and duration at least two seconds, at most four cycles per selected recording. It uses a subject-separated custom split, not the official benchmark split. The utility intentionally leaves imported ICBHI target classes blank. Verify patient separation; retain the original split and corrected recording filenames.
- **ESC-50:** full expectation is 2,000 WAV recordings, with source-defined five folds. Retain these source groups. The utility excludes its breathing category from ambient; real coughing clips remain ambient with a cough tag. Its authors warn that some original clips have class-dependent preprocessing artefacts.
- **UrbanSound8K:** full expectation is 8,732 WAV excerpts in ten predefined folds. Sample rates, channel counts and bit depths differ. Preserve source grouping; do not randomly split related excerpts. Do not describe a custom subset evaluation as the official benchmark.
- **MUSAN:** choose labelled speech, music or noise material as needed. The full archive is large. The automatic subset downloader verifies published Hugging Face LFS SHA-256 values for mirror payloads and retains original attribution files. These verify mirror integrity, not equivalence to an independently verified original corpus hash.
- **Coswara:** use deep/shallow real cough recordings and its manual quality annotations: 0 = bad, 1 = good, 2 = excellent. The executed pipeline automatically joins the official annotations and accepts only quality 1 or 2. Bad or missing scores are rejected. Participant groups stay within one split. The generic import utility alone does not perform this join; use `build_complete_demo.py` for the complete pipeline.

The full-corpus counts above are **published source expectations**. The executed subset downloaded 72 ICBHI WAVs with annotations, 120 ESC-50 WAVs, 228 Coswara WAVs (cough and counting), and 78 MUSAN WAVs (30 noise, 24 speech, 24 music). Prepared counts differ after filtering and segmentation; see the actual manifest. Actual file, retained and rejected counts are written by the utility. Imports with missing source labels/folds, unreadable audio or empty decoded files are rejected and logged; an existing filename alone is not proof of usable audio. Preserve original ICBHI train/test information with your File 2 preparation records: the generic import manifest leaves its split field unresolved.

## COUGHVID acquisition, decoding and quality policy

The v3 original archive is `public_dataset_v3.zip`. The published integrity value is **MD5 `7c6cdf184748a2600538c331ba0ba718`**. The script checks it before extraction and also records a SHA-256. Safe extraction rejects paths outside the extraction folder and rejects archive links/devices. The script refuses to overwrite an existing extraction; use `import` to inspect an existing folder. Do not select the utility's `prepared` outputs as original source data. Keep the destination outside an imported source directory.

Original cough audio may use WEBM or OGG containers with Opus audio. The utility calls the installed [FFmpeg](https://ffmpeg.org/ffmpeg.html) executable and produces mono PCM16 WAV at the **explicit File 2 sample rate**. It preserves originals, original/converted hashes, duration and conversion decisions. Resampling must match training preprocessing and does not create new measurement information.

The strict default policy requires a finite `cough_detected` score greater than 0.8 and available expert quality assessments in the good/ok categories. A poor/bad/no-cough assessment rejects the clip, including conflicting positive/negative expert assessments. Missing metadata, missing scores and unknown quality vocabulary are rejected and logged. This conservative quality policy is a project choice; it is not an official guarantee of cough validity. A deliberate `--allow-unannotated-cough` option retains automatically screened unannotated coughs in a separately labelled **expert quality unknown** group.

The `status` and `status_SSL` fields are not used to define the four classes. The v3 source says `status_SSL` comes from semi-supervised labelling. Do not describe those labels as independent clinical diagnoses. Review the metadata schema of the exact release before changing filters. Only public data are handled; COUGHVID's private test set is not included in its public archive.

## Preparation commands

Run commands from this package directory. Replace `FILE2_SAMPLE_RATE` with the integer sample rate recorded by the training pipeline; there is no assumed default.

```text
python prepare_data.py fetch --dataset esc50 --data-dir data
python prepare_data.py fetch --dataset coughvid --data-dir data --sample-rate FILE2_SAMPLE_RATE
python prepare_data.py import --dataset icbhi --source PATH_TO_EXTRACTED_ICBHI --data-dir data
python prepare_data.py import --dataset urbansound8k --source PATH_TO_EXTRACTED_URBANSOUND8K --data-dir data
python prepare_data.py import --dataset musan --source PATH_TO_EXTRACTED_MUSAN --data-dir data
python prepare_data.py import --dataset coswara --source PATH_TO_EXTRACTED_COSWARA --data-dir data
```

Select only the command/dataset you need. `--convert --sample-rate FILE2_SAMPLE_RATE` optionally decodes other imported audio. Output filenames preserve the source extension before `.wav`, so `example.ogg` becomes `example.ogg.wav`. Record the exact source release/commit with `--source-version YOUR_SOURCE_REVISION`; do not infer a revision from a mirror filename. `--allow-subset` explicitly permits an intentional incomplete ICBHI, ESC-50 or UrbanSound8K source and the report retains the actual count. Fetching and dependency installation require internet; playback from prepared audio and the local trained model can run offline.

For each selected dataset, the utility creates `data/manifests/DATASET/data_manifest.csv`, `preparation_report.json`, and an originality sidecar. The CSV contains real file paths and hashes, source identity, labels or unresolved labels, cough tags and quality decisions. The JSON includes counts and every rejection reason. These generated machine-readable records are not fabricated results. Do not move the raw data without updating its recorded paths.

## Dedicated cough result in E3 and E4

E4 uses the following continuous-timeline metric. E3 uses the same event-level active-alarm principle but resets both detectors at each held-out cough clip and zero-pads clips shorter than the analysis window, with padding flagged. These different boundary/state conditions are explicitly reported; do not merge their rates:

**Cough false-alarm rate = evaluable real cough events with an obstruction alarm active on entry or active in a cough-overlapping, obstruction-free analysis window / evaluable real cough events.**

Report the numerator and denominator, alarm attribution interval, threshold/dwell policy, and separate model/baseline rates. File 4 includes boundary windows that overlap a real cough but contain no obstruction; a cough need not fill a complete analysis window. An alarm already active when the cough begins also counts as a cough-event false alarm, so this rate measures whether the presentation remains in alarm during cough, not only whether cough caused a fresh alarm edge. Events with no eligible analysis window are excluded using a shared denominator for the model and baseline. Keep this same definition in E3 or explicitly distinguish different metrics. Report cough overlapping synthetic obstruction separately rather than counting a correct obstruction alarm as a cough false alarm. A window-level fraction is a different statistic and must be named separately. No valid denominator means unavailable, not zero. The example scenario contains too few coughs to estimate generalisation; repeated controlled evaluation requires held-out cough events and a recorded test protocol.

## Journal research anchors and scope

Use the Rocha et al. *Physiological Measurement* paper for the origin and limitations of respiratory-sound data, and the Orlandic et al. *Scientific Data* paper for cough capture, metadata and filtering. A relevant embedded-classification reference is Acharya, J., & Basu, A. (2020), **Deep Neural Network for Respiratory Sound Classification in Wearable Devices Enabled by Patient Specific Model Tuning**, *IEEE Transactions on Biomedical Circuits and Systems*, [DOI 10.1109/TBCAS.2020.2981172](https://doi.org/10.1109/TBCAS.2020.2981172). Its task and validation do not establish accuracy for this project's synthetic obstruction classes, ESP32 timing or alarm policy. Do not reuse its performance numbers as project results.

Paragraph to write yourself: explain why your chosen real cough/noise examples are useful confounders, how train/test source overlap was excluded, and what remained uncertain after listening to the saved clips. **[STUDENT TO ADD: our own observation from run X / bench test Y]**

## Originality checklist

- Template-like sections: the source table, preparation commands, metric definition and manifest description.
- Rewrite the dataset-selection rationale, filtering tradeoffs and interpretation paragraphs in your own voice using your actual run records. Add **[STUDENT TO ADD: our own observation from run X / bench test Y]** where evidence is still missing.
- Cite the original datasets and papers; retain relevant licence/attribution files. Run your university's plagiarism checker on the written submission and disclose AI assistance according to its policy.
