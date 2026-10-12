"""FILE 4: saved-audio TFLite demonstration; no fabricated model outputs."""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from engine import run_scenario
from visualization import render_airway


ROOT = Path(__file__).resolve().parent

@st.cache_resource
def unpack_bundled_assets():
    """Verify and unpack the repository's own versioned, offline asset archive."""
    import hashlib
    import zipfile
    archive_path = ROOT / "demo_assets.zip"
    expected = "bb7e0a778f545ea582ad7bf4e9940586ea8c18e27caa212a1f621b378767b78e"
    marker = ROOT / ".assets-ready"
    if marker.exists() and marker.read_text() == expected:
        return
    data = archive_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("The saved demo asset archive failed its checksum check.")
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            target = (ROOT / item.filename).resolve()
            if not target.is_relative_to(ROOT.resolve()):
                raise ValueError("Invalid bundled asset path")
        archive.extractall(ROOT)
    marker.write_text(expected)

KINDS = ("normal", "cough", "ambient", "movable", "fixed")
TITLES = {
    "normal": "Normal breathing",
    "cough": "Cough (real recording)",
    "ambient": "Background / speech",
    "movable": "Movable obstruction (synthetic)",
    "fixed": "Fixed obstruction (synthetic)",
}
COLORS = {"normal": "#5ce4c1", "cough": "#ffbd69", "ambient": "#7facff", "movable": "#ff9188", "fixed": "#e88bd9"}
CLASS_TITLES = {
    "normal": "Normal",
    "ambient": "Ambient (includes cough)",
    "movable_synthetic": "Movable · SYNTHETIC",
    "fixed_synthetic": "Fixed · SYNTHETIC",
}


def style_figure(fig: go.Figure, height: int, *, margin_l: int = 48) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=margin_l, r=18, t=15, b=35),
        paper_bgcolor="#111c2b", plot_bgcolor="#111c2b",
        font=dict(color="#c5d3e4", family="Arial", size=11),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#24334a", zeroline=False)
    fig.update_yaxes(gridcolor="#24334a", zeroline=False)
    return fig


def get_event(events: list[dict], cursor: float) -> dict:
    for event in events:
        if float(event["start_s"]) <= cursor < float(event["end_s"]):
            return event
    return events[-1]


def display_number(value: object, suffix: str = "", decimals: int = 2) -> str:
    if value is None:
        return "Unavailable"
    try:
        numeric = float(value)
        return f"{numeric:.{decimals}f}{suffix}" if np.isfinite(numeric) else "Not available"
    except (TypeError, ValueError):
        return str(value)


def csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


@st.cache_data(show_spinner=False)
def prepared_participant(entry: dict):
    from participants import participant_config
    _, config = participant_config(ROOT, entry)
    return config


def choose_config() -> tuple[Path, dict | None]:
    registry_path = ROOT / "participants.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        choices = {row["id"]: row for row in registry["participants"]}
        selected = st.sidebar.selectbox("Breathing recording", list(choices), format_func=lambda value: f"Participant {value}", key="participant_choice")
        entry = choices[selected]
        with st.sidebar.expander("About this recording"):
            st.caption(registry["scope"])
            st.write(f"Breathing source: {entry['breath_source']}")
            st.write(f"Separate cough source: {entry['cough_source']}")
            st.caption(f"Saved test-set normal recall for this source: {entry['normal_recall']:.1%} across {entry['normal_windows']} windows. This is not a clinical success rate or the current timeline result.")
            st.caption("The cough comes from a separate donor. Both obstruction sounds are synthetic.")
        st.session_state.setdefault("config_session_id", uuid.uuid4().hex)
        path = ROOT / f".participant-{st.session_state.config_session_id}-{selected}.json"
        return path, prepared_participant(entry)
    path = ROOT / "config.ready.json"
    st.sidebar.caption("Trained model and real presentation recordings are included.")
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        st.sidebar.error(f"Bundled configuration could not be read: {exc}")
        return path, None


def derive_onset(config_path: Path, config: dict) -> float | None:
    """Inspect clip metadata only; never guess timing for unavailable assets."""
    total = 0.0
    try:
        import soundfile as sf

        for event in config.get("scenario", []):
            if event["kind"] == "movable":
                return total
            if event.get("duration_s") is not None:
                total += float(event["duration_s"])
            else:
                audio_path = Path(event["path"])
                if not audio_path.is_absolute():
                    audio_path = config_path.parent / audio_path
                info = sf.info(audio_path)
                total += info.frames / info.samplerate
    except (ImportError, KeyError, TypeError, ValueError, OSError, RuntimeError):
        pass
    return None


def empty_state(config_path: Path, config: dict | None) -> None:
    left, right = st.columns([1.8, 1], gap="large")
    with left:
        st.markdown("### Explore the visual stage")
        kind = st.selectbox("Illustration preview — no inference", KINDS, format_func=TITLES.get)
        components.html(render_airway(kind, 0.0), height=380)
        st.caption("Illustrative 3D airway cutaway. Scene motion follows the chosen scenario; it is not a measured airflow or anatomy simulation.")
    with right:
        st.markdown("### Waiting for your File 2 assets")
        st.markdown("Connect the trained **four-output TFLite model**, its exact preprocessing settings, saved real breathing/cough/noise clips, and your R5-approved **synthetic** obstruction clips.")
        st.info("Model probabilities, alarms, latency, and inference timings: **TO BE FILLED FROM MY RUN**")
        if config is not None and not config.get("frontend_confirmed_matches_file2", False):
            st.warning("The example configuration is intentionally unverified. Copy the exact frontend and alarm settings from File 2 before enabling it.")
        st.caption(f"Configuration: {config_path}")
    st.markdown("### One continuous demonstration")
    st.markdown(" → ".join(f"**{TITLES[kind]}**" for kind in KINDS))
    st.caption("Cough remains inside the ambient class. The class labels must be supplied in the model's actual output order.")


def waveform_figure(result: dict, cursor: float, history: float) -> go.Figure:
    audio = np.asarray(result["audio"])
    sr = int(result["sr"])
    start = max(0.0, cursor - history)
    stop = min(len(audio), max(1, int(cursor * sr)))
    first = min(stop, int(start * sr))
    stride = max(1, (stop - first) // 2000)
    indices = np.arange(first, stop, stride)
    fig = go.Figure(go.Scatter(x=indices / sr, y=audio[indices], mode="lines", line=dict(color="#5ce4c1", width=1)))
    for event in result["events"]:
        if float(event["end_s"]) > start and float(event["start_s"]) <= cursor:
            fig.add_vrect(x0=max(start, event["start_s"]), x1=min(cursor, event["end_s"]), fillcolor=COLORS[event["kind"]], opacity=0.07, line_width=0)
    fig.update_xaxes(title="Saved-audio time (s)", range=[start, max(cursor, start + 0.01)])
    fig.update_yaxes(title="Amplitude")
    return style_figure(fig, 190)


def feature_figure(result: dict, index: int, config: dict) -> go.Figure:
    feature_type = config["feature"]["type"]
    if index < 0:
        fig = go.Figure()
        fig.add_annotation(text="Waiting for the first complete model window", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    else:
        matrix = np.asarray(result["features"])[index]
        if matrix.ndim != 2:
            raise ValueError(f"Expected channels × frames features; got {matrix.shape}.")
        end = float(result["times"][index])
        feature = config["feature"]
        offset = 0.0 if feature["center"] else float(feature["n_fft"]) / (2.0 * float(result["sr"]))
        frame_times = end - float(config["window_s"]) + offset + np.arange(matrix.shape[1]) * float(feature["hop_length"]) / float(result["sr"])
        fig = go.Figure(go.Heatmap(z=matrix, x=frame_times, y=np.arange(matrix.shape[0]), colorscale="Magma", colorbar=dict(title="Value", thickness=10)))
        fig.update_xaxes(title="Window time (s)")
        fig.update_yaxes(title="Mel band" if feature_type == "logmel" else "MFCC coefficient")
    return style_figure(fig, 230)


def probability_figure(result: dict, index: int) -> go.Figure:
    labels = result["labels"]
    values = np.asarray(result["probabilities"])[index] if index >= 0 else [None] * len(labels)
    color_for_class = {"normal": COLORS["normal"], "ambient": COLORS["ambient"], "movable_synthetic": COLORS["movable"], "fixed_synthetic": COLORS["fixed"]}
    fig = go.Figure(go.Bar(
        x=values, y=[CLASS_TITLES[label] for label in labels], orientation="h",
        marker_color=[color_for_class[label] for label in labels],
        text=[f"{v:.1%}" if v is not None else "Awaiting window" for v in values],
        textposition="auto", hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title="Model probability")
    fig.update_yaxes(autorange="reversed")
    return style_figure(fig, 230, margin_l=178)


def alarm_figure(result: dict, cursor: float) -> go.Figure:
    times = np.asarray(result["times"])
    visible = times <= cursor
    fig = go.Figure()
    for offset, key, color in ((1.4, "model_alarm", "#ff9188"), (0.0, "baseline_alarm", "#7facff")):
        active = np.asarray(result[key], dtype=float)[visible]
        fig.add_trace(go.Scatter(x=times[visible], y=active * 0.7 + offset, line=dict(color=color, width=3, shape="hv"), mode="lines", hovertemplate="Time %{x:.3f}s<extra></extra>"))
    for event in result["events"]:
        fig.add_vrect(x0=event["start_s"], x1=event["end_s"], fillcolor=COLORS[event["kind"]], opacity=0.06, line_width=0)
    fig.add_vline(x=cursor, line_color="#ecf3ff", line_dash="dot")
    fig.update_xaxes(range=[0, float(result["events"][-1]["end_s"])], title="Scenario time (s)")
    fig.update_yaxes(tickvals=[0.0, 0.7, 1.4, 2.1], ticktext=["Baseline · clear", "Baseline · alarm", "Model · clear", "Model · alarm"], range=[-0.2, 2.35])
    return style_figure(fig, 190, margin_l=110)


def scenario_audio_bytes(result: dict, event: dict) -> bytes:
    """Encode only this event's actual mixed samples, preserving their amplitude."""
    import soundfile as sf
    sr = int(result["sr"])
    audio = np.asarray(result["audio"])
    start = max(0, min(len(audio), round(float(event["start_s"]) * sr)))
    stop = max(start, min(len(audio), round(float(event["end_s"]) * sr)))
    buffer = io.BytesIO()
    sf.write(buffer, audio[start:stop], sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def render_scenario_audio(result: dict) -> None:
    st.markdown("### Hear the sounds")
    st.caption("Choose one part of the scenario and press play.")
    events = result["events"]
    selected = st.selectbox(
        "Sound to play", range(len(events)),
        format_func=lambda index: TITLES[events[index]["kind"]],
        key="scenario_audio_event",
    )
    event = events[selected]
    source = "Synthetic scenario audio" if event["kind"] in {"movable", "fixed"} else "Real source recording"
    st.caption(f'{source} · {float(event["start_s"]):.1f}–{float(event["end_s"]):.1f} s of the timeline')
    st.audio(scenario_audio_bytes(result, event), format="audio/wav")


def render_run(result: dict, config: dict) -> None:
    duration = float(result["events"][-1]["end_s"])
    st.session_state.setdefault("cursor", 0.0)
    st.session_state.setdefault("playing", False)
    st.session_state.setdefault("speed", 1.0)
    render_scenario_audio(result)
    st.markdown("### Watch the model respond")
    controls = st.columns([1, 1, 3])
    if controls[0].button("Pause" if st.session_state.playing else "Play timeline", use_container_width=True):
        st.session_state.playing = not st.session_state.playing
        st.session_state.position_slider = st.session_state.cursor
        st.session_state.wall_anchor = time.monotonic()
        st.session_state.trace_anchor = st.session_state.cursor
        st.rerun()
    if controls[1].button("Start over", use_container_width=True):
        st.session_state.cursor = 0.0
        st.session_state.position_slider = 0.0
        st.session_state.trace_anchor = 0.0
        st.session_state.wall_anchor = time.monotonic()
        st.rerun()
    controls[2].caption("The timeline shows recorded model results. Use the slider below to inspect any moment.")

    @st.fragment(run_every=0.5 if st.session_state.playing else None)
    def live_panels() -> None:
        if st.session_state.playing:
            elapsed = (time.monotonic() - st.session_state.wall_anchor) * st.session_state.speed
            st.session_state.cursor = min(duration, st.session_state.trace_anchor + elapsed)
            if st.session_state.cursor >= duration:
                st.session_state.playing = False
                st.session_state.position_slider = duration
                st.rerun()
        cursor = float(st.session_state.cursor)
        if not st.session_state.playing:
            if "position_slider" not in st.session_state:
                st.session_state.position_slider = min(cursor, duration)
            cursor = st.slider("Trace position (seconds)", 0.0, duration, step=min(0.05, duration / 100), key="position_slider")
            st.session_state.cursor = cursor
        event = get_event(result["events"], cursor)
        index = int(np.searchsorted(result["times"], cursor, side="right") - 1)
        first, second = st.columns([1.1, 1.4], gap="medium")
        with first:
            st.markdown(f"### {TITLES[event['kind']]}")
            components.html(render_airway(event["kind"], cursor), height=355)
            st.caption("Illustrative airway view · drag to rotate")
            st.progress(min(1.0, cursor / duration), text=f"Trace {cursor:.2f} s / {duration:.2f} s")
        with second:
            state_columns = st.columns(3)
            model_state = bool(result["model_alarm"][index]) if index >= 0 else None
            baseline_state = bool(result["baseline_alarm"][index]) if index >= 0 else None
            state_columns[0].metric("Model alarm", "ALARM" if model_state else "CLEAR" if model_state is not None else "Awaiting window")
            state_columns[1].metric("Fixed-timeout baseline", "ALARM" if baseline_state else "CLEAR" if baseline_state is not None else "Awaiting window")
            inference = result["inference_ms"][index] if index >= 0 else None
            state_columns[2].metric("Inference / window", display_number(inference, " ms"))
            st.caption("Inference timing: host CPU, not ESP32")
            st.plotly_chart(probability_figure(result, index), use_container_width=True, config={"displayModeBar": False}, key="probabilities")
        st.plotly_chart(waveform_figure(result, cursor, float(st.session_state.get("history_s", 8.0))), use_container_width=True, config={"displayModeBar": False}, key="waveform")
        lower_left, lower_right = st.columns([1, 1])
        with lower_left:
            st.markdown("**Sound features**")
            st.plotly_chart(feature_figure(result, index, config), use_container_width=True, config={"displayModeBar": False}, key="spectrogram")
        with lower_right:
            st.markdown("**Alarm timeline**")
            st.plotly_chart(alarm_figure(result, cursor), use_container_width=True, config={"displayModeBar": False}, key="alarms")
            detections = [row for row in result["metrics"]["detection_events"] if float(row["onset_s"]) <= cursor]
            if detections:
                row = detections[-1]
                st.caption(f"Complete-run result for {TITLES[row['kind']]}: model {row['model_status']}; baseline {row['baseline_status']}.")
                st.metric("Baseline latency − model latency", display_number(row.get("latency_difference_s"), " s") if row.get("latency_difference_s") is not None else "Not jointly measurable")
            else:
                st.caption("Latency is evaluated from each configured obstruction onset. Positive difference means the model alarm occurred earlier.")
            st.caption("Timeline delay; processing time is excluded.")

    live_panels()
    st.divider()
    st.markdown("### Run summary")
    metrics = result["metrics"]
    boxes = st.columns(3)
    boxes[0].metric("False alarms", display_number(metrics.get("false_alarm_episodes"), decimals=0))
    boxes[1].metric("Cough alarms", f"{metrics.get('cough_false_alarm_events', '—')} / {metrics.get('cough_evaluable_events', '—')}")
    boxes[2].metric("Mean inference · host CPU", display_number(metrics.get("inference_mean_ms"), " ms"))
    with st.expander("Detailed event results"):
        detail = st.columns(2)
        detail[0].metric("Timeout baseline false alarms", display_number(metrics.get("baseline_false_alarm_episodes"), decimals=0))
        cough_rate = metrics.get("cough_false_alarm_rate")
        detail[1].metric("Cough false-alarm rate", f"{float(cough_rate):.1%}" if cough_rate is not None else "Not evaluable")
        st.caption(metrics.get("cough_far_definition", "See the recorded metric definition in the exported summary."))
        st.caption(metrics.get("latency_definition", "Latency is measured on the audio timeline; processing and rendering are excluded."))
        st.dataframe(metrics["detection_events"], hide_index=True, use_container_width=True)
        st.caption("A missing latency means no new qualifying alarm was measured. Previously active alarms are reported separately.")
        st.caption(result.get("metadata", {}).get("execution_host", "Fresh run measured on the current hosting CPU."))
    with st.expander("Listen to the actual mixed audio and inspect provenance"):
        st.audio(np.asarray(result["audio"]), sample_rate=int(result["sr"]))
        st.caption("The audio player has independent playback controls; it is not synchronized with the trace playback control above.")
        st.json({"events": result["events"], "class_order": result["labels"], "baseline": config["baseline"], "alarm": config["alarm"]})
    downloads = st.columns(2)
    downloads[0].download_button("Download per-window trace CSV", csv_bytes(result["log_rows"]), "demo_trace.csv", "text/csv", use_container_width=True)
    summary = dict(metrics)
    summary["Originality checklist"] = [
        "Automatically generated metric names are template-like; rewrite the interpretation in your own voice.",
        "[STUDENT TO ADD: our own observation from run X / bench test Y]",
        "Add your own reasoning and run the university plagiarism checker; disclose AI assistance per policy.",
    ]
    downloads[1].download_button("Download measured summary JSON", json.dumps(summary, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)), "demo_metrics.json", "application/json", use_container_width=True)
    with st.expander("Download this exact run for offline replay"):
        st.caption("Includes the mixed audio, features, predictions and measured logs for the displayed run.")
        if st.button("Prepare complete run download", use_container_width=True):
            import tempfile, zipfile
            from export_demo import save_run
            with tempfile.TemporaryDirectory(prefix="airway-export-") as temp:
                saved = save_run(result, Path(temp) / "run")
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                    for file in saved.iterdir():
                        if file.is_file():
                            archive.write(file, "run/" + file.name)
                st.session_state.run_download = buffer.getvalue()
        if st.session_state.get("run_download"):
            st.download_button("Download complete run ZIP", st.session_state.run_download, "airway-run.zip", "application/zip", use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Airway Signal Lab · Simulation", page_icon="◉", layout="wide")
    unpack_bundled_assets()
    st.markdown("""<style>
    .stApp {background:#0d1824;color:#eaf2f8;} [data-testid=stSidebar] {background:#142537;}
    .block-container {padding-top:3.5rem;max-width:1380px;} h1 {letter-spacing:-1px;font-weight:700;}
    .eyebrow {color:#68ddc7;letter-spacing:2px;font-size:12px;margin-top:20px;font-weight:700;}
    [data-testid=stMetric] {background:#142537;border:1px solid #2d4557;padding:14px;border-radius:10px;}
    [data-testid=stMetricValue] {font-size:1.35rem;white-space:normal;}
    [data-testid=stTextInput] input,[data-testid=stNumberInput] input {background:#1a3042;color:#eaf2f8;}
    [data-baseweb=select] > div {background:#1a3042;color:#eaf2f8;}
    .stApp p,.stApp label,.stApp h1,.stApp h2,.stApp h3 {color:#eaf2f8;}
    .stApp [data-testid=stCaptionContainer] p {color:#b7cad7;}
    [data-testid=stSidebar] label,[data-testid=stSidebar] p {color:#dce8f2;}
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">SOFTWARE SIMULATION</div>', unsafe_allow_html=True)
    st.title("Airway Signal Lab")
    st.markdown("Listen to the sounds, then watch how the model and timeout alarm respond.")
    st.sidebar.markdown("## Set up the scenario")
    st.sidebar.markdown("**1. Choose a breathing recording**")
    config_path, config = choose_config()
    ready_run = ROOT / "exports" / "ready_run"
    if not (ROOT / "participants.json").exists() and "result" not in st.session_state and "ready_load_attempted" not in st.session_state and (ready_run / "signals.npz").exists():
        st.session_state.ready_load_attempted = True
        try:
            from export_demo import load_run
            saved_result = load_run(ready_run)
            st.session_state.result = saved_result
            st.session_state.run_config = {k: v for k, v in saved_result["config"].items() if not k.startswith("_")}
            st.session_state.run_path = str(ROOT / "config.ready.json")
            st.session_state.run_overrides = {"noise_path": saved_result["config"].get("noise", {}).get("path"), "snr_db": saved_result["config"].get("noise", {}).get("snr_db"), "movable_onset_s": next(e["start_s"] for e in saved_result["events"] if e["kind"] == "movable")}
            st.session_state.cursor = float(saved_result["times"][0])
            st.session_state.playing = False
        except Exception as exc:
            st.warning(f"Saved demonstration could not be loaded: {exc}")
    overrides: dict = {}
    if config:
        st.sidebar.markdown("**2. Add background and set timing**")
        noise_spec = config.get("noise") or {}
        if not isinstance(noise_spec, dict):
            noise_spec = {}
            st.sidebar.warning("The optional noise setting must be a JSON object.")
        noise_options = config.get("noise_options") or []
        if not isinstance(noise_options, list):
            noise_options = []
        noise_options = [entry for entry in noise_options if isinstance(entry, dict) and isinstance(entry.get("label"), str) and isinstance(entry.get("path"), str)]
        if noise_options:
            noise_by_label = {"Clean · no added background": None, **{entry["label"]: entry["path"] for entry in noise_options}}
            noise_label = st.sidebar.selectbox("Background sound", list(noise_by_label))
            overrides["noise_path"] = noise_by_label[noise_label]
            if overrides["noise_path"] is None:
                config.pop("noise", None)
        elif noise_spec.get("path"):
            st.sidebar.caption(f"Noise file: {noise_spec['path']}")
        else:
            st.sidebar.caption("Add local noise_options or noise.path to enable adjustable SNR.")
        if overrides.get("noise_path") is not None or (not noise_options and noise_spec.get("path")):
            configured_snr = noise_spec.get("snr_db")
            if isinstance(configured_snr, (int, float)) and not isinstance(configured_snr, bool) and np.isfinite(configured_snr):
                overrides["snr_db"] = st.sidebar.number_input("Background level (SNR, dB)", value=float(configured_snr), min_value=-10.0, max_value=40.0, step=1.0, help="A lower SNR makes the background sound more prominent.")
            else:
                st.sidebar.warning("Set noise.snr_db in the configuration to choose a starting SNR.")
        onset = derive_onset(config_path, config)
        if onset is not None and np.isfinite(onset) and onset > 0:
            overrides["movable_onset_s"] = st.sidebar.number_input("Movable event starts at (s)", value=float(onset), min_value=max(0.25, float(onset) - 4.0), max_value=float(onset) + 15.0, step=0.25, help="Changes the ambient segment duration. The real cough is never synthesized or looped.")
        else:
            st.sidebar.caption("Onset control becomes available once clip durations can be read.")
    requested_run = st.sidebar.button("Run scenario", type="primary", disabled=config is None, use_container_width=True)
    participant_changed = config is not None and "participant_id" in config and str(config_path) != st.session_state.get("run_path")
    if requested_run or participant_changed:
        try:
            with st.spinner("Building the audio timeline and measuring model invocation on this host CPU…"):
                if "participant_id" in config:
                    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
                result = run_scenario(str(config_path), **overrides)
            st.session_state.result = result
            result.setdefault("metadata", {})["ui_scenario_overrides"] = dict(overrides)
            st.session_state.run_config = config
            st.session_state.run_path = str(config_path)
            st.session_state.run_overrides = overrides
            st.session_state.cursor = float(result["times"][0])
            st.session_state.playing = False
            st.session_state.pop("position_slider", None)
            st.session_state.pop("saved_run_directory", None)
            st.session_state.pop("run_download", None)
        except Exception as exc:
            st.session_state.pop("result", None)
            st.error(f"The simulation has not run: {exc}")
            st.info("Resolve the missing assets or configuration mismatch and run again. No fallback predictions are generated.")
    with st.sidebar.expander("Playback options"):
        st.number_input("Waveform history (s)", min_value=1.0, max_value=60.0, value=8.0, step=1.0, key="history_s")
        st.select_slider("Timeline speed", options=[0.5, 1.0, 2.0], value=1.0, key="speed", format_func=lambda value: f"{value:g}×")
    if "result" in st.session_state:
        if str(config_path) != st.session_state.get("run_path") or overrides != st.session_state.get("run_overrides") or config != st.session_state.get("run_config"):
            st.info("Settings changed. Select **Run scenario** to update the results.")
        render_run(st.session_state.result, st.session_state.run_config)
    else:
        empty_state(config_path, config)
    research_path = ROOT / "research_evidence.json"
    if research_path.exists():
        with st.expander("Study results and limitations"):
            evidence = json.loads(research_path.read_text(encoding="utf-8"))
            if config and config.get("synthetic_recipe_notice"):
                st.caption(config["synthetic_recipe_notice"])
            st.warning(evidence["scope"])
            st.write("Classification controls")
            st.dataframe([{"Evaluation": key, "Accuracy": f"{value['accuracy']:.2%}", "Macro F1": round(value["macro_f1"], 4)} for key, value in evidence["classification"].items()], hide_index=True, use_container_width=True)
            st.caption("RMS equalization is a test-only distribution shift. It is a sensitivity probe, not a revised clinical benchmark.")
            st.write("Normal recall by held-out respiratory source participant")
            st.dataframe([{"Source participant": row["participant"], "Normal windows": row["normal_windows"], "Normal recall": f"{row['normal_recall']:.1%}"} for row in evidence["participants"]], hide_index=True, use_container_width=True)
            scenarios = evidence["scenarios"]
            st.write(f"Repeated real-cough events with model alarms: {scenarios['cough_model']} / {scenarios['cough_n']}. Baseline: {scenarios['cough_baseline']} / {scenarios['cough_n']}.")
            st.caption("The scenario grid reuses six cough clips. These are not independent patient-level estimates.")
            for kind, detectors in scenarios["all_events"].items():
                st.write(f"{kind.capitalize()} SYNTHETIC event outcomes")
                st.json(detectors, expanded=True)
            st.caption("Already-active alarms have no new-event latency. Misses are retained. Host CPU invocation time excludes preprocessing and is not ESP32 timing.")
    e3_path = ROOT / "experiment" / "e3_results.json"
    if e3_path.exists():
        with st.expander("Test-set results"):
            e3 = json.loads(e3_path.read_text(encoding="utf-8"))
            cough = e3["cough_false_alarm_rate"]
            st.write(e3["scope"])
            st.metric("Held-out window accuracy · synthetic task", f"{e3['classification_report']['accuracy']:.1%}")
            st.write(f"Real cough false alarms: model {cough['model_numerator']} / {cough['denominator']}; baseline {cough['baseline_numerator']} / {cough['denominator']}.")
            st.caption(cough["definition"])
            st.caption("These are prototype measurements. Synthetic-class accuracy does not establish detection of real obstruction. See RESULTS.md for the full error table and limitations.")


if __name__ == "__main__":
    main()

# Originality checklist:
# Interface text and plot layout are templates; rewrite presentation interpretation.
# [STUDENT TO ADD: our own observation from run X / bench test Y]
# Add your own reasoning and run the university plagiarism checker; disclose AI assistance.
