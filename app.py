 """Streamlit interface for the AI Music Generator."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio import render_tokens_to_mp3
from generator import generate_tokens, load_artifacts
from preprocessing import extract_tokens

MODEL_DIR = ROOT / "models"
MIDI_DIR = ROOT / "data" / "midi"
GENERATED_DIR = ROOT / "generated"
MODEL_PATH = MODEL_DIR / "music_generator.keras"

st.set_page_config(
    page_title="AI Music Generator",
    page_icon="♫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#1e2030; --muted:#73768a; --purple:#7962d8; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
    .stApp { background:radial-gradient(circle at 90% 0%, #e8e2ff 0, transparent 28%), linear-gradient(135deg,#fbfaff 0%,#f6f4fb 100%); color:var(--ink); }
    .block-container { max-width:1120px; padding-top:2.5rem; padding-bottom:3rem; }
    h1, h2, h3 { font-family:'Space Grotesk', sans-serif !important; }
    .hero { padding:1.5rem 0 2rem; }
    .kicker { color:var(--purple); letter-spacing:.15em; font-size:.72rem; font-weight:700; }
    .hero h1 { font-size:clamp(2.7rem,5vw,4.6rem); letter-spacing:-.06em; line-height:1; margin:.8rem 0 1rem; }
    .hero h1 span { color:var(--purple); }
    .hero p { max-width:650px; color:var(--muted); font-size:1.05rem; line-height:1.7; }
    .control-card { padding:1.2rem 1.3rem; border:1px solid #e8e4f5; border-radius:18px; background:rgba(255,255,255,.82); box-shadow:0 15px 40px rgba(67,50,126,.07); line-height:1.65; }
    .stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:700; min-height:2.8rem; }
    .metric-box { padding:1rem; border-radius:14px; background:#fff; border:1px solid #ebe8f5; }
    .metric-box strong { display:block; color:var(--purple); font:700 1.35rem 'Space Grotesk'; }
    .metric-box small { color:var(--muted); }
    .status { padding:.9rem 1rem; border-radius:12px; background:#f1edff; color:#5b4ba9; line-height:1.7; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><div class="kicker">AI MUSIC GENERATION STUDIO</div>'
    '<h1>AI Music <span>Generator</span></h1>'
    '<p>Compose original music from a MIDI seed, preview it in your browser, and download the finished track as an MP3.</p></div>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file():
    st.error("The trained model is missing. Run `python train.py --epochs 12` before starting Streamlit.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_artifacts():
    """Load the model and its vocabulary once for the Streamlit process."""
    return load_artifacts(MODEL_DIR)


try:
    model, note_to_int, int_to_note, config = get_artifacts()
    sequence_length = int(config["sequence_length"])
    vocab_size = int(config["vocab_size"])
    if sequence_length < 1 or not note_to_int or not int_to_note:
        raise ValueError("The model configuration or token vocabulary is empty or invalid.")
except (KeyError, TypeError, ValueError, OSError) as exc:
    st.error(f"The model artifacts could not be loaded or are invalid: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"The model could not be loaded: {exc}")
    st.stop()

seed_options = sorted([*MIDI_DIR.glob("*.mid"), *MIDI_DIR.glob("*.midi")], key=lambda path: path.name.lower())
if not seed_options:
    st.warning(f"No MIDI seed files were found in `{MIDI_DIR.relative_to(ROOT)}`. Add a `.mid` or `.midi` seed file to generate a track.")
    st.stop()

with st.sidebar:
    st.markdown("### Generation studio")
    seed_path = st.selectbox(
        "Seed composition",
        seed_options,
        format_func=lambda path: path.stem,
        help="The model uses this composition as context for the generated continuation.",
    )
    length = st.slider("New notes to generate", min_value=50, max_value=300, value=100, step=10)
    temperature = st.slider(
        "Creativity (temperature)", min_value=0.5, max_value=1.5, value=0.8, step=0.1,
        help="Lower values tend to be more predictable; higher values tend to be more varied.",
    )
    tempo = st.slider("Tempo (BPM)", min_value=60, max_value=180, value=110, step=5)
    note_beats = st.select_slider("Note duration", options=[0.25, 0.5, 0.75, 1.0], value=0.5, format_func=lambda value: f"{value:g} beat")
    timbre = st.selectbox("Synth sound", ["Warm", "Soft", "Bright"])
    use_fixed_seed = st.toggle("Make this run repeatable", value=False)
    random_seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1, disabled=not use_fixed_seed)
    st.caption("The model is cached. Generated tracks are synthesized directly to MP3.")

left, right = st.columns([1.3, 0.7], gap="large")
with left:
    st.markdown("#### Create a new track")
    st.markdown(
        f'<div class="control-card"><b>Seed:</b> {seed_path.stem}<br>'
        f'<b>Arrangement:</b> {length} notes · {tempo} BPM · {timbre} synth.<br>'
        "The generated audio is saved and delivered as an MP3 file.</div>",
        unsafe_allow_html=True,
    )
    generate = st.button("♫  Generate MP3", use_container_width=True, type="primary")

    if generate:
        try:
            with st.spinner("Composing and rendering your MP3…"):
                seed_tokens = extract_tokens(seed_path)
                if not seed_tokens:
                    raise ValueError("The selected MIDI file did not contain readable note or chord events.")
                rng = np.random.default_rng(int(random_seed)) if use_fixed_seed else None
                generated_tokens = generate_tokens(
                    model, seed_tokens, note_to_int, int_to_note, sequence_length,
                    length, temperature, rng=rng,
                )
                mp3_data = render_tokens_to_mp3(
                    generated_tokens, tempo=tempo, beats_per_note=note_beats, timbre=timbre,
                )
                GENERATED_DIR.mkdir(parents=True, exist_ok=True)
                filename = f"generated_music_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp3"
                output_path = GENERATED_DIR / filename
                output_path.write_bytes(mp3_data)

            st.session_state["generated_path"] = str(output_path)
            st.session_state["generated_count"] = len(generated_tokens)
            st.session_state["generated_temperature"] = temperature
            st.session_state["generated_seed_name"] = seed_path.stem
            st.session_state["generated_tempo"] = tempo
            st.success("MP3 generated successfully.")
        except Exception as exc:
            st.error(f"MP3 generation failed: {exc}")

    generated_path = st.session_state.get("generated_path")
    if generated_path:
        output_path = Path(generated_path)
        if output_path.is_file():
            st.markdown("#### Your generated track")
            st.audio(output_path.read_bytes(), format="audio/mpeg")
            metric_cols = st.columns(3)
            metric_cols[0].markdown(
                f'<div class="metric-box"><strong>{st.session_state.get("generated_count", 0)}</strong><small>notes generated</small></div>',
                unsafe_allow_html=True,
            )
            metric_cols[1].markdown(
                f'<div class="metric-box"><strong>{st.session_state.get("generated_temperature", 0.8):.1f}</strong><small>creativity</small></div>',
                unsafe_allow_html=True,
            )
            metric_cols[2].markdown(
                f'<div class="metric-box"><strong>{st.session_state.get("generated_tempo", 110)}</strong><small>BPM</small></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Seed composition: {st.session_state.get('generated_seed_name', 'Unknown')}")
            st.download_button(
                "Download MP3", data=output_path.read_bytes(), file_name=output_path.name,
                mime="audio/mpeg", use_container_width=True,
            )
        else:
            st.info("The previous generated track is no longer available. Generate a new composition to continue.")
            st.session_state.pop("generated_path", None)

with right:
    st.markdown("#### Model snapshot")
    st.markdown(
        f'<div class="status"><b>LSTM model ready</b><br>{vocab_size:,} musical tokens'
        f'<br>Context window: {sequence_length} tokens<br>Available seed files: {len(seed_options)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("\n")
    st.markdown("**MP3 audio**")
    st.markdown("Generated notes and chords are rendered into a synthesized audio track, playable here and downloadable as MP3.")
    st.markdown("**Creative controls**")
    st.markdown("Adjust sampling temperature, tempo, note duration, and synth timbre. Enable a fixed seed to reproduce the same sampled note sequence.")
