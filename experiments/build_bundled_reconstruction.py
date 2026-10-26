"""Build a small source-separated reconstruction dataset from bundled participant clips.

This is a constrained educational experiment, not the original training corpus.
Every derivative of a breathing source stays in one split.  Ambient examples are
synthetic additive-noise mixtures of the same source, and both obstruction
classes are synthetic signal transformations; the output must never be reported
as clinical performance or as a reproduction of RESULTS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import base64
import csv
import json
import math
import zlib
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = ("normal", "ambient", "movable_synthetic", "fixed_synthetic")
SPLITS = {"105": "train", "111": "train", "125": "train", "131": "train",
          "147": "val", "153": "val", "207": "test", "216": "test"}
SEED = 20261026
SR, WINDOW = 16000, 32000


def decode(record: dict, key: str) -> np.ndarray:
    item = record[key]
    raw = zlib.decompress(base64.b64decode(item["zlib_base64_float32"]))
    value = np.frombuffer(raw, dtype="<f4").copy()
    if len(value) != item["sample_count"] or not np.isfinite(value).all():
        raise ValueError(f"Invalid bundled {key} payload")
    return value


def crop_loop(audio: np.ndarray, start: int) -> np.ndarray:
    repeated = np.tile(audio, math.ceil((start + WINDOW) / len(audio)))
    return repeated[start:start + WINDOW].astype(np.float32)


def mel(audio: np.ndarray) -> np.ndarray:
    value = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=512,
        hop_length=256, n_mels=32, fmin=50, fmax=7800, power=2,
        center=False, window="hann", pad_mode="constant", norm="slaney", htk=False)
    return librosa.power_to_db(value, ref=1.0, amin=1e-10, top_db=80.0).astype(np.float32)


def make_examples(breath: np.ndarray, participant: str, rng: np.random.Generator):
    examples = []
    for index in range(18):
        start = int(rng.integers(0, max(1, len(breath))))
        base = crop_loop(breath, start)
        base *= float(rng.uniform(0.75, 1.20))
        peak = max(float(np.max(np.abs(base))), 1e-6)
        base = base * min(0.92 / peak, 1.0)
        examples.append(("normal", base, index))
        noise = rng.normal(0, float(rng.uniform(0.004, 0.018)), WINDOW).astype(np.float32)
        examples.append(("ambient", np.clip(base + noise, -0.95, 0.95), index))
        cutoff = float(rng.uniform(700, 1600)); hz = float(rng.uniform(2.5, 5.5)); phase = float(rng.uniform(0, 2*np.pi))
        movable = sosfilt(butter(3, cutoff, fs=SR, output="sos"), base)
        t = np.arange(WINDOW, dtype=np.float32) / SR
        envelope = .035 + .75 * (.5 + .5 * np.sin(2*np.pi*hz*t + phase)) ** 3
        examples.append(("movable_synthetic", (movable * envelope).astype(np.float32), index))
        cutoff = float(rng.uniform(350, 900)); gain = float(rng.uniform(.025, .08))
        fixed = sosfilt(butter(3, cutoff, fs=SR, output="sos"), base) * gain
        examples.append(("fixed_synthetic", fixed.astype(np.float32), index))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "bundled_reconstruction")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    registry = json.loads((ROOT / "participants.json").read_text(encoding="utf-8"))
    rows, split_values = [], {name: [] for name in ("train", "val", "test")}
    for entry in registry["participants"]:
        participant = entry["id"]
        record = json.loads((ROOT / entry["path"]).read_text(encoding="utf-8"))
        split = SPLITS[participant]
        for label, waveform, index in make_examples(decode(record, "breath"), participant, rng):
            split_values[split].append((mel(waveform), CLASS_ORDER.index(label)))
            rows.append({"split": split, "group_id": f"icbhi:{participant}", "class_name": label,
                         "window_index": index, "source": entry["breath_source"],
                         "derivation": "bundled reconstruction; synthetic additive noise or documented DSP surrogate"})
    train_features = np.stack([x for x, _ in split_values["train"]])
    mean, std = float(train_features.mean()), float(train_features.std())
    output = args.output
    if (output / "prepared_source_split.npz").exists():
        raise ValueError("Dataset already exists; choose a new --output directory.")
    output.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for split, values in split_values.items():
        arrays[f"X_{split}"] = (np.stack([x for x, _ in values]) - mean) / max(std, 1e-6)
        arrays[f"y_{split}"] = np.asarray([y for _, y in values], dtype=np.int64)
    for split in split_values:
        selected = [row for row in rows if row["split"] == split]
        arrays[f"groups_{split}"] = np.asarray([row["group_id"] for row in selected])
        for index, row in enumerate(selected):
            value = np.array(arrays[f"X_{split}"][index], dtype="<f4", copy=True)
            value[value == 0] = 0
            row["feature_sha256"] = hashlib.sha256(value.tobytes()).hexdigest()
    np.savez_compressed(output / "prepared_source_split.npz", **arrays)
    with (output / "window_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (output / "dataset_metadata.json").write_text(json.dumps({
        "status": "prepared", "seed": SEED,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy": np.__version__, "librosa": librosa.__version__, "class_order": CLASS_ORDER,
        "splits": SPLITS, "counts": {key: len(value) for key, value in split_values.items()},
        "normalization": {"type": "global_scalar", "fit": "training features only", "mean": mean, "std": std},
        "scope": "Small bundled reconstruction using eight ICBHI breathing sources. Ambient and obstruction examples are generated transformations; results are not clinical validation and are not comparable to the historical full prepared-data evaluation."}, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
