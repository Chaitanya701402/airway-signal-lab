"""Reproducible CNN/GRU/attention ablation runner for the research package.

This module deliberately does not import or alter the Streamlit deployment model.
It expects prepared, source-split log-Mel arrays supplied by the original research
package; it never creates a split from already derived windows.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np

CLASS_ORDER = ("normal", "ambient", "movable_synthetic", "fixed_synthetic")
SEED = 20261011


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)


def require_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required. In Colab, run the notebook environment cell first.") from exc


def _one_hot(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == 4:
        return labels.astype(np.float32)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be integer class indices or one-hot arrays with four columns.")
    if np.any((labels < 0) | (labels >= 4)):
        raise ValueError("Labels must use the fixed indices 0..3 in CLASS_ORDER.")
    return np.eye(4, dtype=np.float32)[labels]


def _features(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 3:
        value = value[..., None]
    if value.ndim != 4 or value.shape[-1] != 1:
        raise ValueError(f"{name} must be (N, mels, frames) or (N, mels, frames, 1), got {value.shape}.")
    if value.shape[1] != 32:
        raise ValueError(f"{name} must contain 32 Mel bands, got {value.shape[1]}.")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return value


def load_prepared_npz(path: str | Path) -> dict:
    """Load a deliberately narrow, documented prepared-array interchange format.

    Required keys: X_train, y_train, X_val, y_val, X_test, y_test.
    Optional group arrays are checked for source leakage; the runner does not infer
    groups from window order. Features must already be normalized using train only,
    with the supplied manifest recording the normalization provenance.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prepared dataset archive not found: {path}")
    required = ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"Prepared archive lacks required arrays: {', '.join(missing)}")
        data = {key: archive[key] for key in archive.files}
    for split in ("train", "val", "test"):
        data[f"X_{split}"] = _features(data[f"X_{split}"], f"X_{split}")
        data[f"y_{split}"] = _one_hot(data[f"y_{split}"])
        if len(data[f"X_{split}"]) != len(data[f"y_{split}"]):
            raise ValueError(f"X_{split} and y_{split} have different lengths.")
    return data


def validate_provenance(data: dict, manifest_path: str | Path | None) -> dict:
    """Validate the explicit source split manifest if available; otherwise report a block."""
    report = {"manifest": None, "group_integrity": "unverified", "normalization": "unverified"}
    if manifest_path is None:
        return report
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    required = {"split", "group_id", "class_name"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Manifest must include split, group_id, and class_name columns.")
    bad = [r for r in rows if r["split"] not in {"train", "val", "test"} or r["class_name"] not in CLASS_ORDER or not r["group_id"]]
    if bad:
        raise ValueError("Manifest contains invalid split, group_id, or class_name values.")
    memberships: dict[str, set[str]] = {}
    for row in rows:
        memberships.setdefault(row["group_id"], set()).add(row["split"])
    leaked = sorted(group for group, splits in memberships.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"Source-group leakage in manifest: {leaked[:10]}")
    report.update({"manifest": str(path), "manifest_rows": len(rows), "groups": len(memberships),
                   "group_integrity": "verified", "normalization": "must be documented as train-only by manifest/run metadata"})
    return report


def frequency_mean_layer():
    """A registered serializable layer that retains the temporal feature axis."""
    tf = require_tensorflow()
    @tf.keras.utils.register_keras_serializable(package="airway")
    class FrequencyMean(tf.keras.layers.Layer):
        def call(self, values):
            return tf.reduce_mean(values, axis=1)
    return FrequencyMean(name="frequency_mean")


def temporal_attention(sequence, name: str):
    """Learn a scalar score per frame and normalize weights across time (axis=1)."""
    tf = require_tensorflow()
    @tf.keras.utils.register_keras_serializable(package="airway")
    class AttentionPooling(tf.keras.layers.Layer):
        def build(self, input_shape):
            self.score = tf.keras.layers.Dense(1, name="score")
        def call(self, values):
            weights = tf.nn.softmax(self.score(values), axis=1)
            return tf.reduce_sum(values * weights, axis=1)
    return AttentionPooling(name=f"{name}_context")(sequence)


def cnn_feature_extractor(inputs):
    tf = require_tensorflow()
    x = inputs
    for i, filters in enumerate((12, 20, 24), start=1):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"conv{i}")(x)
        x = tf.keras.layers.MaxPool2D((2, 2), name=f"pool{i}")(x)
    return x


def build_model(variant: str, input_shape: tuple[int, int, int], dropout: float = 0.2):
    """Build E0..E3. E0 is a documented reimplementation: original source is absent."""
    tf = require_tensorflow()
    if variant not in {"E0", "E1", "E2", "E3"}:
        raise ValueError("variant must be E0, E1, E2, or E3")
    inputs = tf.keras.Input(input_shape, name="logmel")
    feature_map = cnn_feature_extractor(inputs)
    if variant == "E0":
        representation = tf.keras.layers.GlobalAveragePooling2D(name="cnn_gap")(feature_map)
    else:
        # Average only over frequency, retaining the (downsampled) temporal axis.
        sequence = frequency_mean_layer()(feature_map)
        if variant in {"E1", "E3"}:
            sequence = tf.keras.layers.GRU(32, return_sequences=True, name="gru")(sequence)
        representation = (tf.keras.layers.GlobalAveragePooling1D(name="temporal_mean")(sequence)
                          if variant == "E1" else temporal_attention(sequence, "attention"))
    x = tf.keras.layers.Dropout(dropout, name="dropout")(representation)
    x = tf.keras.layers.Dense(24, activation="relu", name="dense24")(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax", name="class_probability")(x)
    return tf.keras.Model(inputs, outputs, name=f"airway_{variant.lower()}")


def smoke_models(input_shape=(32, 124, 1)) -> dict:
    tf = require_tensorflow()
    results = {}
    batch = np.zeros((2, *input_shape), dtype=np.float32)
    for variant in ("E0", "E1", "E2", "E3"):
        model = build_model(variant, input_shape)
        output = model(batch, training=False).numpy()
        assert output.shape == (2, 4), (variant, output.shape)
        assert np.isfinite(output).all() and np.allclose(output.sum(axis=1), 1, atol=1e-5)
        results[variant] = {"parameters": int(model.count_params()), "output_shape": list(output.shape)}
        tf.keras.backend.clear_session()
    return results


def class_weights(y: np.ndarray) -> dict[int, float]:
    indices = np.argmax(y, axis=1)
    counts = np.bincount(indices, minlength=4)
    if np.any(counts == 0):
        raise ValueError("Every class must be represented in training data.")
    return {i: float(len(indices) / (4 * count)) for i, count in enumerate(counts)}


def evaluate(model, x: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    probabilities = model.predict(x, verbose=0)
    truth, predicted = np.argmax(y, axis=1), np.argmax(probabilities, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=range(4), zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": float(np.mean(precision)), "macro_recall": float(np.mean(recall)), "macro_f1": float(np.mean(f1)),
        "per_class": {CLASS_ORDER[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                                        "f1": float(f1[i]), "support": int(support[i])} for i in range(4)},
        "classification_report": classification_report(truth, predicted, labels=range(4), target_names=CLASS_ORDER, output_dict=True, zero_division=0),
    }, confusion_matrix(truth, predicted, labels=range(4))


def _plot_matrix(matrix: np.ndarray, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
           xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for row in range(4):
        for col in range(4): ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _plot_histories(histories: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, history in histories.items():
        axes[0].plot(history["loss"], label=name); axes[0].plot(history["val_loss"], linestyle="--")
        axes[1].plot(history["accuracy"], label=name); axes[1].plot(history["val_accuracy"], linestyle="--")
    axes[0].set(title="Loss (solid train / dashed validation)", xlabel="Epoch"); axes[1].set(title="Accuracy", xlabel="Epoch")
    axes[0].legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def run(data_path: str | Path, output_dir: str | Path, manifest: str | Path | None = None, seed: int = SEED,
        epochs: int = 35, batch_size: int = 32) -> dict:
    tf = require_tensorflow(); set_seed(seed)
    data = load_prepared_npz(data_path); provenance = validate_provenance(data, manifest)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    weights = class_weights(data["y_train"])
    rows, reports, histories, matrices = [], {}, {}, {}
    for variant in ("E0", "E1", "E2", "E3"):
        tf.keras.backend.clear_session(); set_seed(seed)
        model = build_model(variant, tuple(data["X_train"].shape[1:]))
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
        checkpoint = output / f"{variant}_best.keras"
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                     tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True)]
        start = time.perf_counter()
        history = model.fit(data["X_train"], data["y_train"], validation_data=(data["X_val"], data["y_val"]),
                            class_weight=weights, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=callbacks)
        duration = time.perf_counter() - start
        loaded = tf.keras.models.load_model(checkpoint)
        metrics, matrix = evaluate(loaded, data["X_test"], data["y_test"])
        elapsed = []
        for sample in data["X_test"][: min(100, len(data["X_test"]))]:
            begin = time.perf_counter_ns(); loaded(sample[None, ...], training=False); elapsed.append((time.perf_counter_ns() - begin) / 1e6)
        best = int(np.argmin(history.history["val_loss"])) + 1
        result = {**metrics, "parameter_count": int(loaded.count_params()), "training_duration_s": duration,
                  "best_val_loss": float(min(history.history["val_loss"])), "selected_epoch": best,
                  "inference_mean_ms": float(np.mean(elapsed)), "inference_scope": "Keras model only; feature extraction excluded; current Colab device"}
        reports[variant] = result; matrices[variant] = matrix.tolist(); histories[variant] = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        rows.append({"variant": variant, **{key: result[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "parameter_count", "training_duration_s", "best_val_loss", "selected_epoch", "inference_mean_ms")}})
    with (output / "ablation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (output / "detailed_reports.json").write_text(json.dumps({"class_order": CLASS_ORDER, "reports": reports, "confusion_matrices": matrices}, indent=2), encoding="utf-8")
    _plot_matrix(np.array(matrices["E0"]), output / "confusion_matrix_E0.png", "E0 test confusion matrix")
    _plot_matrix(np.array(matrices["E3"]), output / "confusion_matrix_E3.png", "E3 test confusion matrix")
    _plot_histories(histories, output / "training_curves.png")
    metadata = {"status": "completed", "seed": seed, "class_order": CLASS_ORDER, "data_path": str(data_path),
                "provenance": provenance, "feature_protocol": "Prepared 16 kHz mono 2 s windows; 32-band log-Mel; FFT 512; hop 256. Train-only normalization must be documented by supplied artifact.",
                "historical_reference": "7,344-parameter CNN and 0.8816 macro F1 are historical, not reproduced claims; E0 is a reimplementation because original training source is absent.",
                "scientific_limits": "Synthetic obstruction surrogates are not clinical labels; ambient includes real cough; held-out custom split is not an official ICBHI score."}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "README.md").write_text("# Hybrid ablation output\n\nGenerated by `experiments/hybrid_ablation.py`. Checkpoints use validation loss; held-out test is evaluated once per fixed configuration. Synthetic obstruction results are not clinical validation.\n", encoding="utf-8")
    return {"rows": rows, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Prepared source-split .npz archive")
    parser.add_argument("--output", default="experiments/hybrid_ablation")
    parser.add_argument("--manifest", help="CSV source split manifest with split/group_id/class_name")
    parser.add_argument("--seed", type=int, default=SEED); parser.add_argument("--epochs", type=int, default=35); parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true", help="Run architecture forward-pass checks only")
    args = parser.parse_args()
    if args.smoke: print(json.dumps(smoke_models(), indent=2)); return
    print(json.dumps(run(args.data, args.output, args.manifest, args.seed, args.epochs, args.batch_size), indent=2))


if __name__ == "__main__": main()
