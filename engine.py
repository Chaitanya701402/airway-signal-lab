"""Offline audio replay and real TFLite inference for the File 4 demonstration.

No model, medical labels, coughs, or obstruction recordings are generated here.
The configured frontend must be copied from the model's training/export pipeline.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
import os
os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(__file__).resolve().parent / '.runtime_cache' / 'numba'))
from typing import Any

import numpy as np

CLASS_NAMES = ("normal", "ambient", "movable_synthetic", "fixed_synthetic")
SCENARIO_KINDS = ("normal", "cough", "ambient", "movable", "fixed")
OBSTRUCTION_KINDS = {"movable", "fixed"}


def _required(mapping: dict, keys: tuple | list, where: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{where} must be a JSON object.")
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"Missing {where} settings: {', '.join(missing)}")


def _number(value: Any, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a JSON number, not a string or placeholder.")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"Invalid {name}: {value}")
    return result


def load_config(path: str | Path) -> dict:
    """Read explicit settings and reject unconfirmed or incompatible metadata."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8-sig") as handle:
        config = json.load(handle)
    _required(config, ["model_path", "class_order", "sample_rate", "window_s", "hop_s",
                       "input_layout", "output_type", "frontend_confirmed_matches_file2",
                       "feature", "scenario", "alarm", "baseline"], "configuration")
    if config["frontend_confirmed_matches_file2"] is not True:
        raise ValueError("Copy and verify the File 2 frontend settings, then set "
                         "frontend_confirmed_matches_file2 to true. Example settings cannot run.")
    if (not isinstance(config["class_order"], list)
            or any(not isinstance(label, str) for label in config["class_order"])
            or sorted(config["class_order"]) != sorted(CLASS_NAMES)):
        raise ValueError("class_order must give the model's exact order of normal, ambient, "
                         "movable_synthetic, fixed_synthetic. Cough belongs to ambient.")
    if config["input_layout"] not in {"NMF", "NMFC", "NFM", "NFMC"}:
        raise ValueError("input_layout must be NMF, NMFC, NFM, or NFMC (M=feature bins, F=frames).")
    if config["output_type"] not in {"probabilities", "logits"}:
        raise ValueError("output_type must be probabilities or logits.")
    sr = _number(config["sample_rate"], "sample_rate", 1)
    if sr != int(sr):
        raise ValueError("sample_rate must be an integer.")
    window_s = _number(config["window_s"], "window_s", 1 / sr)
    hop_s = _number(config["hop_s"], "hop_s", 1 / sr)
    if hop_s > window_s:
        raise ValueError("hop_s cannot exceed window_s; unanalysed timeline gaps are not supported.")
    feature = config["feature"]
    _required(feature, ["type", "n_fft", "hop_length", "n_mels", "fmin", "fmax", "power",
                        "center", "window", "pad_mode", "mel_norm", "htk", "log", "normalization"], "feature")
    if feature["type"] not in {"logmel", "mfcc"}:
        raise ValueError("feature.type must be logmel or mfcc.")
    for key in ("center", "htk"):
        if not isinstance(feature[key], bool):
            raise ValueError(f"feature.{key} must be true or false, not a string.")
    if _number(feature["power"], "feature.power", 0) == 0:
        raise ValueError("feature.power must be positive.")
    if feature["type"] == "mfcc":
        _required(feature, ["n_mfcc", "dct_type", "dct_norm", "lifter"], "MFCC")
    _required(feature["log"], ["type", "reference", "amin", "top_db"], "log scaling")
    if feature["log"]["type"] != "db":
        raise ValueError("This frontend supports explicit power-to-dB only. Use the File 2 frontend "
                         "implementation if its training transform differs.")
    if _number(feature["log"]["reference"], "log.reference", 0) == 0:
        raise ValueError("log.reference must be positive.")
    if _number(feature["log"]["amin"], "log.amin", 0) == 0:
        raise ValueError("log.amin must be positive.")
    if feature["log"]["top_db"] is not None:
        _number(feature["log"]["top_db"], "log.top_db", 0)
    if not 0 <= _number(feature["fmin"], "feature.fmin") < _number(feature["fmax"], "feature.fmax") <= sr / 2:
        raise ValueError("Feature frequencies must satisfy 0 <= fmin < fmax <= sample_rate/2.")
    for key in ("n_fft", "hop_length", "n_mels"):
        val = _number(feature[key], f"feature.{key}", 1)
        if val != int(val):
            raise ValueError(f"feature.{key} must be an integer.")
    if round(window_s * sr) < feature["n_fft"]:
        raise ValueError("Analysis windows must contain at least n_fft samples.")
    norm = feature["normalization"]
    _required(norm, ["type"], "normalization")
    if norm.get("type") not in {"none", "per_window", "global"}:
        raise ValueError("normalization.type must be none, per_window, or global.")
    if norm["type"] == "per_window":
        _required(norm, ["epsilon"], "per-window normalization")
        if _number(norm["epsilon"], "normalization.epsilon", 0) == 0:
            raise ValueError("normalization.epsilon must be positive.")
    if norm["type"] == "global":
        _required(norm, ["mean", "std"], "training-set normalization")
        mean, std = np.asarray(norm["mean"], dtype=float), np.asarray(norm["std"], dtype=float)
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Global normalization must be finite and standard deviation positive.")
    scenario = config["scenario"]
    if (not isinstance(scenario, list) or any(not isinstance(event, dict) for event in scenario)
            or [event.get("kind") for event in scenario] != list(SCENARIO_KINDS)):
        raise ValueError("scenario must contain normal, cough, ambient, movable, fixed in that order.")
    for event in scenario:
        _required(event, ["path", "synthetic"], f"scenario {event['kind']}")
        if not isinstance(event["path"], str) or not event["path"].strip():
            raise ValueError(f"Scenario {event['kind']} requires a saved-audio path.")
        if "repeat" in event and not isinstance(event["repeat"], bool):
            raise ValueError("Scenario repeat must be true or false, not a string.")
        if event["kind"] in OBSTRUCTION_KINDS:
            provenance = event.get("r5_provenance", "")
            placeholder = (not isinstance(provenance, str) or not provenance.strip()
                           or any(marker in provenance.upper() for marker in
                                  ("STUDENT TO ADD", "TO BE FILLED", "TODO", "PLACEHOLDER")))
            if event["synthetic"] is not True or placeholder:
                raise ValueError("Both obstruction files must declare synthetic:true and a nonempty "
                                 "r5_provenance describing the production recipe and any unresolved R5 requirements.")
        elif event["synthetic"] is not False:
            raise ValueError("Normal, cough, and ambient audio must be real recordings (synthetic:false).")
        if event["kind"] == "cough" and event.get("repeat", False):
            raise ValueError("Cough clips cannot be repeated or synthesized by this demo.")
        if "duration_s" in event:
            _number(event["duration_s"], "event duration_s", 1 / sr)
    alarm = config["alarm"]
    _required(alarm, ["probability_threshold", "consecutive_windows", "refractory_s"], "alarm")
    threshold = _number(alarm["probability_threshold"], "alarm threshold", 0)
    if threshold > 1:
        raise ValueError("Alarm probability threshold must be in [0,1].")
    windows = _number(alarm["consecutive_windows"], "consecutive_windows", 1)
    if windows != int(windows):
        raise ValueError("consecutive_windows must be an integer.")
    _number(alarm["refractory_s"], "refractory_s", 0)
    baseline = config["baseline"]
    _required(baseline, ["rms_threshold", "timeout_s", "description"], "baseline")
    _number(baseline["rms_threshold"], "rms_threshold", 0)
    if _number(baseline["timeout_s"], "timeout_s", 0) == 0:
        raise ValueError("baseline.timeout_s must be positive.")
    if not isinstance(baseline["description"], str) or not baseline["description"].strip():
        raise ValueError("Explain the baseline inactivity definition in baseline.description.")
    if "UNVERIFIED EXAMPLE" in baseline["description"].upper():
        raise ValueError("Replace the unverified example baseline description and confirm its RMS "
                         "threshold and timeout against your experiment protocol.")
    config["_config_dir"] = str(config_path.parent)
    config["_config_path"] = str(config_path)
    return config


def _path(config: dict, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config["_config_dir"]) / path).resolve()


def _load_audio(path: Path, target_sr: int) -> np.ndarray:
    import librosa
    import soundfile as sf
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved audio: {path}")
    audio, source_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    # This explicit mono average and soxr_hq resampling must match File 2 preprocessing.
    audio = audio.mean(axis=1)
    if source_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr, res_type="soxr_hq")
    if not np.any(audio):
        raise ValueError(f"All-zero audio is not accepted as a saved scenario recording: {path}")
    return np.asarray(audio, dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble_scenario(config: dict, movable_onset_s: float | None = None) -> tuple:
    """Concatenate saved recordings; repeat only where explicitly permitted."""
    sr = int(config["sample_rate"])
    pieces, events, source_records = [], [], []
    offset = 0
    for spec in config["scenario"]:
        source = _path(config, spec["path"])
        audio = _load_audio(source, sr)
        duration = spec.get("duration_s", len(audio) / sr)
        if spec["kind"] == "ambient" and movable_onset_s is not None:
            duration = _number(movable_onset_s, "movable_onset_s", 0) - offset / sr
        count = round(_number(duration, "segment duration", 1 / sr) * sr)
        if count > len(audio):
            if not spec.get("repeat", False):
                raise ValueError(f"{spec['kind']} needs {count / sr:.3f}s but its saved clip has "
                                 f"{len(audio) / sr:.3f}s. Supply longer audio or explicitly permit repeat.")
            audio = np.tile(audio, math.ceil(count / len(audio)))
        audio = audio[:count]
        events.append({"kind": spec["kind"], "start_s": offset / sr, "end_s": (offset + count) / sr,
                       "synthetic": spec["synthetic"], "source_path": str(source),
                       "r5_provenance": spec.get("r5_provenance")})
        source_records.append({"kind": spec["kind"], "path": str(source), "sha256": _sha256(source),
                               "used_samples": count, "repeat_permitted": bool(spec.get("repeat", False))})
        pieces.append(audio)
        offset += count
    return np.concatenate(pieces), events, source_records


def mix_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, dict]:
    """Set global full-timeline RMS SNR; use common attenuation to prevent clipping."""
    requested = _number(snr_db, "snr_db")
    noise = np.tile(noise, math.ceil(len(audio) / len(noise)))[:len(audio)]
    signal_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    noise_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
    if signal_rms <= 0 or noise_rms <= 0:
        raise ValueError("SNR mixing requires nonzero signal and noise RMS.")
    noise_gain = signal_rms / (10 ** (requested / 20) * noise_rms)
    scaled_noise = noise * noise_gain
    mixed = audio + scaled_noise
    attenuation = min(1.0, 0.999 / max(float(np.max(np.abs(mixed))), 1e-12))
    achieved = 20 * np.log10(signal_rms / float(np.sqrt(np.mean(scaled_noise.astype(np.float64) ** 2))))
    return (mixed * attenuation).astype(np.float32), {
        "requested_snr_db": requested, "achieved_snr_db": float(achieved),
        "noise_gain": float(noise_gain), "common_attenuation": float(attenuation),
        "definition": "Full-timeline signal RMS divided by added-noise RMS; not segment-local SNR."}


def extract_features(audio: np.ndarray, sr: int, feature: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return display features before normalization and the normalized model features."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=int(feature["n_fft"]), hop_length=int(feature["hop_length"]),
        n_mels=int(feature["n_mels"]), fmin=float(feature["fmin"]), fmax=float(feature["fmax"]),
        power=float(feature["power"]), center=bool(feature["center"]), window=feature["window"],
        pad_mode=feature["pad_mode"], norm=feature["mel_norm"], htk=bool(feature["htk"]))
    log = feature["log"]
    display = librosa.power_to_db(mel, ref=float(log["reference"]), amin=float(log["amin"]),
                                 top_db=log["top_db"])
    if feature["type"] == "mfcc":
        display = librosa.feature.mfcc(S=display, n_mfcc=int(feature["n_mfcc"]),
                                      dct_type=int(feature["dct_type"]), norm=feature["dct_norm"],
                                      lifter=float(feature["lifter"]))
    normalized = display.copy()
    norm = feature["normalization"]
    if norm["type"] == "per_window":
        normalized = (normalized - normalized.mean()) / max(float(normalized.std()), float(norm["epsilon"]))
    elif norm["type"] == "global":
        mean, std = np.asarray(norm["mean"]), np.asarray(norm["std"])
        if mean.ndim == 1:
            mean = mean[:, None]
        if std.ndim == 1:
            std = std[:, None]
        normalized = (normalized - mean) / std
    if not np.isfinite(normalized).all():
        raise ValueError("Feature extraction produced non-finite values.")
    return display.astype(np.float32), normalized.astype(np.float32)


def _quant_parameters(detail: dict, ndim: int) -> tuple:
    params = detail.get("quantization_parameters", {})
    scales = np.asarray(params.get("scales", []), dtype=np.float32)
    zeros = np.asarray(params.get("zero_points", []), dtype=np.float32)
    if (scales.size == 0 or np.any(scales <= 0) or zeros.size != scales.size
            or not np.isfinite(scales).all() or not np.isfinite(zeros).all()):
        raise ValueError("Integer model tensor lacks valid quantization parameters.")
    if scales.size > 1:
        axis = params.get("quantized_dimension")
        if not isinstance(axis, (int, np.integer)) or not 0 <= int(axis) < ndim:
            raise ValueError("Per-axis quantization dimension does not match the tensor rank.")
        shape = [1] * ndim
        shape[int(axis)] = len(scales)
        scales, zeros = scales.reshape(shape), zeros.reshape(shape)
    return scales, zeros


def encode_tensor(values: np.ndarray, detail: dict) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError("Model input contains non-finite feature values.")
    dtype = np.dtype(detail["dtype"])
    if np.issubdtype(dtype, np.integer):
        scales, zeros = _quant_parameters(detail, values.ndim)
        limits = np.iinfo(dtype)
        return np.clip(np.rint(values / scales + zeros), limits.min, limits.max).astype(dtype)
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(f"Unsupported input dtype: {dtype}")
    return values.astype(dtype)


def decode_tensor(values: np.ndarray, detail: dict) -> np.ndarray:
    if np.issubdtype(np.dtype(detail["dtype"]), np.integer):
        scales, zeros = _quant_parameters(detail, values.ndim)
        return (values.astype(np.float32) - zeros) * scales
    return values.astype(np.float32)


class TFLiteModel:
    def __init__(self, path: Path, config: dict):
        if not path.is_file():
            raise FileNotFoundError(f"Missing File 2 trained model: {path}")
        try:
            from ai_edge_litert.interpreter import Interpreter
            self.runtime = "ai_edge_litert"
        except ImportError:
            try:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
                self.runtime = "tensorflow.lite"
            except ImportError as error:
                raise ImportError("Install ai-edge-litert or TensorFlow to run the saved TFLite model.") from error
        self.interpreter = Interpreter(model_path=str(path), num_threads=1)
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("This demo requires a stateless single-input, single-output TFLite classifier.")
        self.input, self.output = inputs[0], outputs[0]
        self.layout, self.output_type = config["input_layout"], config["output_type"]

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, float]:
        values = features.T if self.layout in {"NFM", "NFMC"} else features
        values = values[None, ...]
        if self.layout.endswith("C"):
            values = values[..., None]
        if tuple(values.shape) != tuple(self.input["shape"]):
            raise ValueError(f"Frontend shape {values.shape} differs from trained model input "
                             f"{tuple(self.input['shape'])}. Copy File 2 settings; do not reshape blindly.")
        self.interpreter.set_tensor(self.input["index"], encode_tensor(values, self.input))
        start = time.perf_counter_ns()
        self.interpreter.invoke()
        inference_ms = (time.perf_counter_ns() - start) / 1e6
        prediction = decode_tensor(self.interpreter.get_tensor(self.output["index"]), self.output).reshape(-1)
        if prediction.size != 4 or not np.isfinite(prediction).all():
            raise ValueError("Model must return exactly four finite class outputs.")
        if self.output_type == "logits":
            prediction = np.exp(prediction - prediction.max())
            prediction /= prediction.sum()
        else:
            if np.any(prediction < -1e-7) or np.any(prediction > 1 + 1e-7) or not np.isclose(prediction.sum(), 1, atol=0.02):
                raise ValueError("Output declared probabilities is not a normalized four-class distribution.")
            # Preserve dequantized outputs, including small quantization sum error; do not invent calibration.
        return prediction.astype(np.float32), float(inference_ms)


def model_alarm_trace(probabilities: np.ndarray, times: np.ndarray, labels: list, settings: dict) -> np.ndarray:
    """Hysteresis in time: consecutive threshold windows, then refractory after release."""
    indices = [labels.index("movable_synthetic"), labels.index("fixed_synthetic")]
    scores = probabilities[:, indices].sum(axis=1)
    active, count, next_allowed = False, 0, -math.inf
    trace = []
    for score, timestamp in zip(scores, times):
        qualifying = score >= settings["probability_threshold"]
        count = count + 1 if qualifying else 0
        if active and not qualifying:
            active = False
            next_allowed = float(timestamp) + settings["refractory_s"]
        if not active and count >= settings["consecutive_windows"] and timestamp >= next_allowed:
            active = True
        trace.append(active)
    return np.asarray(trace, dtype=bool)


def baseline_alarm_trace(rms_values: np.ndarray, times: np.ndarray, settings: dict) -> np.ndarray:
    """Fixed timeout since the most recent high-RMS analysis window; start at t=0.

    This is an amplitude inactivity comparator, not a validated breath detector.
    Noise and cough can reset the timer; the display must retain that explanation.
    """
    last_activity = 0.0
    trace = []
    for rms, timestamp in zip(rms_values, times):
        if rms >= settings["rms_threshold"]:
            last_activity = float(timestamp)
        trace.append(float(timestamp) - last_activity >= settings["timeout_s"])
    return np.asarray(trace, dtype=bool)


def rising_edges(trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=bool)
    return trace & ~np.r_[False, trace[:-1]] if len(trace) else trace.copy()


def _event_at(timestamp: float, events: list) -> str:
    # A window ending at a boundary contains the preceding event, not the next event.
    for event in events:
        if event["start_s"] < timestamp <= event["end_s"]:
            return event["kind"]
    return "outside_timeline"


def compute_metrics(times: np.ndarray, window_s: float, model_alarm: np.ndarray,
                    baseline_alarm: np.ndarray, events: list, inference_ms: np.ndarray) -> dict:
    """Count alarm episodes and explicitly censor unavailable event latencies."""
    times = np.asarray(times, dtype=float)
    model_alarm, baseline_alarm = np.asarray(model_alarm, bool), np.asarray(baseline_alarm, bool)
    model_edges, baseline_edges = rising_edges(model_alarm), rising_edges(baseline_alarm)
    kinds = np.asarray([_event_at(float(t), events) for t in times])
    non_obstruction = ~np.isin(kinds, list(OBSTRUCTION_KINDS))
    metrics = {
        "false_alarm_episodes": int(np.sum(model_edges & non_obstruction)),
        "baseline_false_alarm_episodes": int(np.sum(baseline_edges & non_obstruction)),
        "cough_false_alarm_events": 0, "baseline_cough_false_alarm_events": 0,
        "cough_evaluable_events": 0, "cough_unevaluable_events": 0,
        "inference_mean_ms": float(np.mean(inference_ms)) if len(inference_ms) else None,
        "inference_p95_ms": float(np.percentile(inference_ms, 95)) if len(inference_ms) else None,
        "detection_events": [],
        "false_alarm_definition": "New alarm episodes at analysis-window end times in non-obstruction events.",
        "cough_far_definition": "For each detector, fraction of real cough events with an alarm already active "
                                "on entry or active in a cough-overlapping analysis window that contains no "
                                "obstruction. Includes boundary windows; events without an eligible window "
                                "are excluded from the shared denominator.",
        "latency_definition": "Audio-timeline algorithmic latency: first new alarm window-end timestamp minus "
                              "obstruction onset. Excludes processing and rendering; not end-to-end or ESP32 "
                              "latency. Missing and already-active alarms have no numeric latency. Positive "
                              "difference means the baseline was later on the audio timeline.",
    }
    for event in events:
        start, end = event["start_s"], event["end_s"]
        if event["kind"] == "cough":
            cough_windows = (times > start + 1e-9) & (times - window_s < end - 1e-9)
            for other in events:
                if other["kind"] in OBSTRUCTION_KINDS:
                    contaminated = (times > other["start_s"] + 1e-9) & (times - window_s < other["end_s"] - 1e-9)
                    cough_windows &= ~contaminated
            if cough_windows.any():
                metrics["cough_evaluable_events"] += 1
                previous = np.flatnonzero(times <= start + 1e-9)
                for key, states in (("cough_false_alarm_events", model_alarm),
                                    ("baseline_cough_false_alarm_events", baseline_alarm)):
                    active_on_entry = bool(len(previous) and states[previous[-1]])
                    metrics[key] += int(active_on_entry or bool(np.any(states & cough_windows)))
            else:
                metrics["cough_unevaluable_events"] += 1
        if event["kind"] not in OBSTRUCTION_KINDS:
            continue
        eligible = (times > start + 1e-9) & (times <= end + 1e-9)
        record = {"kind": event["kind"], "onset_s": float(start), "event_duration_s": float(end - start),
                  "observation_horizon_s": float(times[eligible][-1] - start) if eligible.any() else 0.0}
        previous = np.flatnonzero(times <= start + 1e-9)
        for name, states, edges in (("model", model_alarm, model_edges),
                                    ("baseline", baseline_alarm, baseline_edges)):
            hits = np.flatnonzero(edges & eligible)
            if not eligible.any():
                status, latency = "no_evaluable_windows", None
            elif len(previous) and states[previous[-1]]:
                status, latency = "already_active_at_onset", None
            elif len(hits):
                status, latency = "detected", float(times[hits[0]] - start)
            else:
                status, latency = "missed", None
            record[f"{name}_status"] = status
            record[f"{name}_latency_s"] = latency
        model_latency, base_latency = record["model_latency_s"], record["baseline_latency_s"]
        record["latency_difference_s"] = (base_latency - model_latency
                                           if base_latency is not None and model_latency is not None else None)
        metrics["detection_events"].append(record)
    denominator = metrics["cough_evaluable_events"]
    metrics["cough_false_alarm_rate"] = (metrics["cough_false_alarm_events"] / denominator
                                        if denominator else None)
    metrics["baseline_cough_false_alarm_rate"] = (metrics["baseline_cough_false_alarm_events"] / denominator
                                                 if denominator else None)
    return metrics


def run_scenario(config_path: str | Path, snr_db: float | None = None,
                 noise_path: str | Path | None = None, movable_onset_s: float | None = None) -> dict:
    config = load_config(config_path)
    sr = int(config["sample_rate"])
    model_path = _path(config, config["model_path"])
    model = TFLiteModel(model_path, config)
    audio, events, source_records = assemble_scenario(config, movable_onset_s=movable_onset_s)
    noise_spec = config.get("noise") or {}
    selected_noise = noise_path if noise_path is not None else noise_spec.get("path")
    selected_snr = snr_db if snr_db is not None else noise_spec.get("snr_db")
    mixing = None
    if selected_noise:
        if selected_snr is None:
            raise ValueError("A selected noise recording requires an explicit SNR.")
        resolved_noise = _path(config, selected_noise)
        noise = _load_audio(resolved_noise, sr)
        audio, mixing = mix_noise(audio, noise, selected_snr)
        source_records.append({"kind": "added_noise", "path": str(resolved_noise), "sha256": _sha256(resolved_noise)})
    elif selected_snr is not None:
        raise ValueError("SNR was set but no real saved noise recording was selected.")
    window = round(float(config["window_s"]) * sr)
    hop = round(float(config["hop_s"]) * sr)
    starts = np.arange(0, len(audio) - window + 1, hop, dtype=int)
    if len(starts) == 0:
        raise ValueError("Scenario is shorter than one complete model window.")
    times = (starts + window) / sr
    features, probabilities, inference_ms, rms_values = [], [], [], []
    for start in starts:
        chunk = audio[start:start + window]
        display, model_features = extract_features(chunk, sr, config["feature"])
        prediction, elapsed = model.predict(model_features)
        features.append(display)
        probabilities.append(prediction)
        inference_ms.append(elapsed)
        rms_values.append(float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))))
    probabilities, inference_ms, rms_values = map(np.asarray, (probabilities, inference_ms, rms_values))
    from diagnostics import inference_diagnostics
    labels = list(config["class_order"])
    diagnostics = inference_diagnostics(probabilities, inference_ms, labels)
    model_alarm = model_alarm_trace(probabilities, times, labels, config["alarm"])
    baseline_alarm = baseline_alarm_trace(rms_values, times, config["baseline"])
    metrics = compute_metrics(times, window / sr, model_alarm, baseline_alarm, events, inference_ms)
    model_edges, baseline_edges = rising_edges(model_alarm), rising_edges(baseline_alarm)
    rows = []
    for i, timestamp in enumerate(times):
        row = {"time_s": float(timestamp), "window_start_s": float(starts[i] / sr),
               "event_kind": _event_at(float(timestamp), events), "rms": float(rms_values[i]),
               "model_alarm": bool(model_alarm[i]), "model_alarm_onset": bool(model_edges[i]),
               "baseline_alarm": bool(baseline_alarm[i]), "baseline_alarm_onset": bool(baseline_edges[i]),
               "inference_ms": float(inference_ms[i])}
        row.update({f"p_{label}": float(probabilities[i, j]) for j, label in enumerate(labels)})
        rows.append(row)
    return {"audio": audio, "sr": sr, "times": times, "probabilities": probabilities,
            "features": np.stack(features), "inference_ms": inference_ms,
            "model_alarm": model_alarm, "baseline_alarm": baseline_alarm, "events": events,
            "labels": labels, "metrics": metrics, "log_rows": rows, "config": config,
            "metadata": {"diagnostics": diagnostics, "runtime": model.runtime, "inference_device": "host CPU, not ESP32",
                         "detection_latency_scope": "Audio-timeline algorithmic latency; excludes processing and rendering; not end-to-end or ESP32 latency.",
                         "inference_timing_scope": "TFLite invoke only; includes first invocation; excludes feature extraction and display",
                         "model_sha256": _sha256(model_path), "sources": source_records, "mixing": mixing,
                         "window_s_actual": window / sr, "hop_s_actual": hop / sr,
                         "unanalysed_tail_s": (len(audio) - int(starts[-1]) - window) / sr,
                         "preprocessing": "Stereo channels averaged to mono; librosa soxr_hq resampling when needed.",
                         "baseline_description": config["baseline"]["description"]}}


# Originality checklist: configuration and signal-processing structure are template-like.
# [STUDENT TO ADD: our own observation from run X / bench test Y]
# Rewrite interpretation of measured results and baseline limitations in your own voice.
# Run the university plagiarism checker and disclose AI assistance according to policy.
