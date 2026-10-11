"""Matplotlib animation fallback and PNG/GIF export from real inference logs.

Run with --config for inference, or --run to replay a previously saved run.
No demonstration probabilities, waveforms, or numerical results are fabricated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BANNER = "SIMULATION. Obstruction audio is SYNTHETIC. Not clinically validated."


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Unsupported metadata type: {type(obj).__name__}")


def save_run(run: dict, directory: str | Path) -> Path:
    """Preserve only actual run outputs; NPZ loads without Python pickles."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    keys = ("audio", "times", "probabilities", "features", "inference_ms", "model_alarm", "baseline_alarm")
    np.savez_compressed(directory / "signals.npz", **{k: np.asarray(run[k]) for k in keys})
    metadata = {k: v for k, v in run.items() if k not in keys}
    metadata["Originality checklist"] = [
        "Template-like: automatically generated metadata and export layout.",
        "Rewrite your explanation of these results in your own voice.",
        "[STUDENT TO ADD: our own observation from run X / bench test Y]",
        "Run your university's plagiarism checker and disclose AI assistance per policy.",
    ]
    (directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_safe, allow_nan=False), encoding="utf-8"
    )
    from scipy.io import wavfile
    wavfile.write(directory / "scenario.wav", int(run["sr"]), np.asarray(run["audio"], dtype=np.float32))
    return directory


def load_run(directory: str | Path) -> dict:
    directory = Path(directory)
    run = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    with np.load(directory / "signals.npz", allow_pickle=False) as values:
        run.update({k: values[k] for k in values.files})
    if len(run["times"]) == 0:
        raise ValueError("The run has no inference windows.")
    return run


def build_animation(run: dict, fps: float = 6.0, max_frames: int | None = None):
    """Return (figure, FuncAnimation, update) for local or Colab display."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if fps <= 0:
        raise ValueError("fps must be positive")
    times = np.asarray(run["times"], dtype=float)
    audio = np.asarray(run["audio"])
    sr = int(run["sr"])
    if len(times) == 0:
        raise ValueError("No model output exists to animate.")
    duration = len(audio) / sr
    timeline = np.r_[np.arange(0, duration, 1.0 / fps), duration]
    if max_frames and len(timeline) > max_frames:
        timeline = np.linspace(0, duration, max_frames)
    labels = [str(x).replace("_synthetic", "\nSYNTHETIC") for x in run["labels"]]
    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(12, 9), constrained_layout=True)
        grid = fig.add_gridspec(5, 2, height_ratios=[0.55, 1.3, 1.3, 0.85, 0.7])
        title = fig.add_subplot(grid[0, :]); title.axis("off")
        title.text(.5, .95, BANNER, ha="center", va="top", color="#f4ce86", fontsize=11)
        status = title.text(.5, .30, "", ha="center", va="center", fontsize=10)
        wave = fig.add_subplot(grid[1, :]); spec = fig.add_subplot(grid[2, 0])
        bars = fig.add_subplot(grid[2, 1]); lane = fig.add_subplot(grid[3, :])
        summary = fig.add_subplot(grid[4, :]); summary.axis("off")
        metrics = run["metrics"]
        count = metrics.get("cough_false_alarm_events")
        total = metrics.get("cough_evaluable_events")
        cough_text = f"{count}/{total}" if total else "not evaluable"
        baseline_count = metrics.get("baseline_cough_false_alarm_events")
        baseline_cough_text = f"{baseline_count}/{total}" if total and baseline_count is not None else "not evaluable"
        summary.text(0, .95, f"Whole-run summaries · Model false-alarm episodes: {metrics.get('false_alarm_episodes', 'unavailable')} · Baseline: {metrics.get('baseline_false_alarm_episodes', 'unavailable')}\nReal-cough false-alarm events · Model: {cough_text} · Baseline: {baseline_cough_text} (E4 demo; not E3)", va="top", fontsize=9)
        latency_text = summary.text(0, .24, "", va="center", fontsize=9)
    values = np.asarray(run["features"])
    fmin, fmax = float(np.min(values)), float(np.max(values))
    if fmin == fmax:
        fmax = fmin + 1.0  # display scale only, never a reported model value

    @plt.style.context("dark_background")
    def update(clock_s):
        index = int(np.searchsorted(times, clock_s, side="right") - 1)
        for axis in (wave, spec, bars, lane):
            axis.clear()
        active = next((e for e in run["events"] if e["start_s"] <= clock_s < e["end_s"]), run["events"][-1])
        kind = active["kind"]
        detection = next((x for x in run["metrics"].get("detection_events", []) if x["kind"] == kind), None)
        if detection:
            def latency(name):
                value = detection.get(f"{name}_latency_s")
                return f"{value:.3f} s" if value is not None else detection.get(f"{name}_status", "unavailable")
            delta = detection.get("latency_difference_s")
            difference = f"{delta:.3f} s" if delta is not None else "not jointly measurable"
            latency_text.set_text(f"Whole-run {kind} SYNTHETIC event result · Model: {latency('model')} · Baseline: {latency('baseline')}\nBaseline minus model latency: {difference} · Audio timeline only; excludes processing / rendering")
        else:
            latency_text.set_text("Audio-timeline latency is evaluated per SYNTHETIC obstruction event; excludes processing / rendering.")
        if kind in {"movable", "fixed"}:
            kind += " obstruction · SYNTHETIC"
        elif kind == "cough":
            kind = "real cough · ambient-class confounder"
        start = max(0, clock_s - 6)
        end = min(duration, max(clock_s, 0.05))
        a, b = int(start * sr), min(len(audio), max(int(end * sr), 1))
        stride = max(1, (b - a) // 2500)
        wave.plot(np.arange(a, b, stride) / sr, audio[a:b:stride], color="#78d9e5", linewidth=.8)
        wave.set(xlabel="Scenario time (s)", ylabel="Digital amplitude", xlim=(start, max(start + .05, end)))
        wave.set_title("Saved scenario audio · scrolling view", loc="left", fontsize=10)
        lane.set(xlabel="Scenario time (s)", yticks=[0, 1], yticklabels=["Fixed-timeout baseline", "Model alarm"], xlim=(0, duration), ylim=(-.4, 1.8))
        for event in run["events"]:
            if event["kind"] in {"movable", "fixed"}:
                lane.axvspan(event["start_s"], event["end_s"], color="#dfaa68", alpha=.13)
        lane.axvline(clock_s, color="#ffffff", linewidth=.8)
        if index >= 0:
            idx = min(index, len(times) - 1)
            elapsed = times[:idx + 1]
            for key, y, color in [("model_alarm", 1, "#78d9e5"), ("baseline_alarm", 0, "#dda2dc")]:
                states = np.asarray(run[key][:idx + 1], dtype=bool)
                lane.scatter(elapsed[states], np.full(np.count_nonzero(states), y), marker="|", s=70, color=color)
            feature = np.asarray(run["features"][idx])
            spec.imshow(feature, origin="lower", aspect="auto", vmin=fmin, vmax=fmax, cmap="magma")
            spec.set(xlabel="Feature frame within input window", ylabel="Feature bin / coefficient")
            spec.set_title("Frontend features before normalization", loc="left", fontsize=10)
            bars.barh(labels, run["probabilities"][idx], color="#78d9e5")
            bars.set(xlim=(0, 1), xlabel="Model probability")
            bars.invert_yaxis()
            for row, p in enumerate(run["probabilities"][idx]):
                bars.text(min(float(p) + .015, .87), row, f"{float(p):.3f}", va="center", fontsize=8)
            status.set_text(f"t={clock_s:.2f} s · {kind}\nModel alarm: {bool(run['model_alarm'][idx])} · Baseline: {bool(run['baseline_alarm'][idx])} · Inference: {run['inference_ms'][idx]:.3f} ms (host CPU, not ESP32)")
        else:
            for axis in (spec, bars):
                axis.text(.5, .5, "Awaiting first complete audio window", transform=axis.transAxes, ha="center", fontsize=9)
                axis.set_xticks([]); axis.set_yticks([])
            status.set_text(f"t={clock_s:.2f} s · {kind} · Awaiting first complete window")
        # Tick text is created lazily at draw time, after the style context exits.
        for axis in (wave, spec, bars, lane):
            axis.tick_params(axis="both", colors="#ecf3ff", labelcolor="#ecf3ff")
        return []

    animation = FuncAnimation(fig, update, frames=timeline, interval=1000 / fps, blit=False, repeat=False)
    return fig, animation, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path)
    src.add_argument("--run", type=Path, help="Directory previously saved by save_run")
    parser.add_argument("--output", type=Path, help="GIF animation path (no audio)")
    parser.add_argument("--png", type=Path, help="Still image export path")
    parser.add_argument("--frame-time", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save-run", type=Path, help="Preserve actual run arrays and metadata")
    args = parser.parse_args()
    if not args.output and not args.png and not args.save_run:
        parser.error("Choose --output, --png or --save-run")
    if args.config:
        from engine import run_scenario
        run = run_scenario(args.config)
    else:
        run = load_run(args.run)
    if args.save_run:
        save_run(run, args.save_run)
    if args.output or args.png:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.animation import PillowWriter
        import matplotlib.pyplot as plt
        fig, animation, update = build_animation(run, args.fps, args.max_frames)
        if args.output:
            if args.output.suffix.lower() != ".gif":
                parser.error("--output must end in .gif")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            animation.save(str(args.output), writer=PillowWriter(fps=args.fps))
        if args.png:
            if not 0 <= args.frame_time <= len(run["audio"]) / run["sr"]:
                parser.error("--frame-time lies outside the saved scenario")
            args.png.parent.mkdir(parents=True, exist_ok=True)
            update(args.frame_time)
            fig.savefig(args.png, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
    print("Exported actual run data. Inference timings are host CPU, not ESP32.")


if __name__ == "__main__":
    main()

# Originality checklist: The export layout and command structure are template-like.
# Rewrite the methods, comparison rationale and result interpretation yourself.
# [STUDENT TO ADD: our own observation from run X / bench test Y]
# Run your university's plagiarism checker and disclose AI assistance per policy.
