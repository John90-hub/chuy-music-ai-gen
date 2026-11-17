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

from generator import generate_tokens, load_artifacts
from preprocessing import extract_tokens, write_tokens_to_midi

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
    '<p>Create original symbolic musical sequences with an LSTM neural network. '
    'Choose a MIDI seed, tune the generation, and download the result to edit in your DAW.</p></div>',
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
    st.warning(f"No MIDI seed files were found in `{MIDI_DIR.relative_to(ROOT)}`. Add a `.mid` or `.midi` file and restart the app.")
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
        "Creativity (temperature)",
        min_value=0.5,
        max_value=1.5,
        value=0.8,
        step=0.1,
        help="Lower values tend to be more predictable; higher values tend to be more varied.",
    )
    use_fixed_seed = st.toggle("Make this run repeatable", value=False, help="Use the same random seed to reproduce a run with the same model and inputs.")
    random_seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1, disabled=not use_fixed_seed)
    st.caption("The model is loaded once and is not retrained during generation.")

left, right = st.columns([1.3, 0.7], gap="large")
with left:
    st.markdown("#### Create a new sequence")
    st.markdown(
        f'<div class="control-card"><b>Selected seed:</b> {seed_path.stem}<br>'
        f'<b>Generation:</b> {length} new notes at temperature {temperature:.1f}. '
        "Each composition is saved as a downloadable MIDI file.</div>",
        unsafe_allow_html=True,
    )
    generate = st.button("♫  Generate Music", use_container_width=True, type="primary")

    if generate:
        try:
            with st.spinner("Composing a new sequence…"):
                seed_tokens = extract_tokens(seed_path)
                if not seed_tokens:
                    raise ValueError("The selected MIDI file did not contain readable note or chord events.")
                rng = np.random.default_rng(int(random_seed)) if use_fixed_seed else None
                generated_tokens = generate_tokens(
                    model,
                    seed_tokens,
                    note_to_int,
                    int_to_note,
                    sequence_length,
                    length,
                    temperature,
                    rng=rng,
                )
                GENERATED_DIR.mkdir(parents=True, exist_ok=True)
                filename = f"generated_music_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mid"
                output_path = write_tokens_to_midi(generated_tokens, GENERATED_DIR / filename)

            st.session_state["generated_path"] = str(output_path)
            st.session_state["generated_count"] = len(generated_tokens)
            st.session_state["generated_temperature"] = temperature
            st.session_state["generated_seed_name"] = seed_path.stem
            st.success("Generation completed successfully.")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

    generated_path = st.session_state.get("generated_path")
    if generated_path:
        output_path = Path(generated_path)
        if output_path.is_file():
            st.markdown("#### Your generated MIDI")
            metric_cols = st.columns(3)
            metric_cols[0].markdown(
                f'<div class="metric-box"><strong>{st.session_state.get("generated_count", 0)}</strong><small>new notes</small></div>',
                unsafe_allow_html=True,
            )
            metric_cols[1].markdown(
                f'<div class="metric-box"><strong>{st.session_state.get("generated_temperature", 0.8):.1f}</strong><small>temperature</small></div>',
                unsafe_allow_html=True,
            )
            metric_cols[2].markdown(
                f'<div class="metric-box"><strong>.mid</strong><small>playable format</small></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Seed composition: {st.session_state.get('generated_seed_name', 'Unknown')}")
            st.download_button(
                "Download MIDI",
                data=output_path.read_bytes(),
                file_name=output_path.name,
                mime="audio/midi",
                use_container_width=True,
            )
        else:
            st.info("The previous generated file is no longer available. Generate a new composition to continue.")
            st.session_state.pop("generated_path", None)

with right:
    st.markdown("#### Model snapshot")
    st.markdown(
        f'<div class="status"><b>LSTM model ready</b><br>{vocab_size:,} musical tokens'
        f'<br>Context window: {sequence_length} tokens<br>Available seeds: {len(seed_options)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("\n")
    st.markdown("**Temperature guide**")
    st.markdown("Lower settings generally create more predictable continuations. Higher settings can produce more surprising ideas, but may be less coherent.")
    st.markdown("**Reproducible runs**")
    st.markdown("Enable the fixed random seed in the sidebar to repeat a generation with the same seed file and settings.")
    st.markdown("**Output**")
    st.markdown("MIDI is editable and playable in MIDI-compatible music software.")
