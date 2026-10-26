"""Descriptive diagnostics for model probabilities, never clinical confidence."""
from __future__ import annotations
import numpy as np


def inference_diagnostics(probabilities, inference_ms, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    timing = np.asarray(inference_ms, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
        raise ValueError("Expected a nonempty windows-by-classes probability array.")
    if len(labels) != p.shape[1] or len(set(labels)) != len(labels):
        raise ValueError("Labels must be unique and match probability columns.")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(axis=1), 1, atol=0.02):
        raise ValueError("Probabilities must be finite, bounded, and sum to one.")
    if timing.shape != (len(p),) or not np.isfinite(timing).all() or np.any(timing < 0):
        raise ValueError("Provide one finite nonnegative timing per window.")
    # Quantized probability outputs may have small sum errors; normalize only for diagnostics.
    p = p / p.sum(axis=1, keepdims=True)
    ordered = np.sort(p, axis=1)
    entropy = -np.sum(p * np.log(np.maximum(p, np.finfo(float).tiny)), axis=1) / np.log(p.shape[1])
    counts = np.bincount(p.argmax(axis=1), minlength=len(labels))
    return {
        "windows": len(p),
        "predicted_class_counts": dict(zip(labels, map(int, counts))),
        "mean_max_probability": float(ordered[:, -1].mean()),
        "mean_top_two_margin": float((ordered[:, -1] - ordered[:, -2]).mean()),
        "mean_normalized_entropy": float(entropy.mean()),
        "inference_median_ms": float(np.median(timing)),
        "inference_p95_ms": float(np.percentile(timing, 95)),
        "interpretation": "Describes output concentration, not calibrated confidence or accuracy. Correlated windows are not independent observations.",
    }
