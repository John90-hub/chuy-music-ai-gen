"""Streamlit demo for the AI Music Generator."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator import generate_tokens, load_artifacts
from preprocessing import extract_tokens, write_tokens_to_midi

MODEL_DIR = ROOT / "models"
MIDI_DIR = ROOT / "data" / "midi"
GENERATED_DIR = ROOT / "generated"

st.set_page_config(page_title="AI Music Generator", page_icon="♫", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#1e2030; --muted:#73768a; --purple:#7962d8; --lav:#eeeafe; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
.stApp { background: radial-gradient(circle at 90% 0%, #e8e2ff 0, transparent 28%), linear-gradient(135deg,#fbfaff 0%,#f6f4fb 100%); color:var(--ink); }
.block-container { max-width: 1120px; padding-top: 3rem; }
h1, h2, h3 { font-family:'Space Grotesk', sans-serif !important; }
.hero { padding:2rem 0 2.5rem; }
.kicker { color:var(--purple); letter-spacing:.15em; font-size:.72rem; font-weight:700; }
.hero h1 { font-size:clamp(2.7rem,5vw,4.6rem); letter-spacing:-.06em; line-height:1; margin:.8rem 0 1rem; }
.hero h1 span { color:var(--purple); }
.hero p { max-width:600px; color:var(--muted); font-size:1.05rem; line-height:1.7; }
.control-card { padding:1.25rem 1.3rem; border:1px solid #e8e4f5; border-radius:18px; background:rgba(255,255,255,.78); box-shadow:0 15px 40px rgba(67,50,126,.07); }
.stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:700; border:0; background:var(--purple); color:white; min-height:2.8rem; }
.stButton > button:hover, .stDownloadButton > button:hover { background:#624bc2; color:white; }
.metric-box { padding:1rem; border-radius:14px; background:#fff; border:1px solid #ebe8f5; }
.metric-box strong { display:block; color:var(--purple); font:700 1.35rem 'Space Grotesk'; }
.metric-box small { color:var(--muted); }
.status { padding:.85rem 1rem; border-radius:12px; background:#f1edff; color:#5b4ba9; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="kicker">AI MUSIC GENERATION STUDIO</div><h1>AI Music <span>Generator</span></h1><p>Generate original symbolic musical sequences with a compact LSTM neural network trained on classical MIDI compositions.</p></div>', unsafe_allow_html=True)

if not (MODEL_DIR / "music_generator.keras").exists():
    st.error("The trained model is not available yet. Run `python train.py --epochs 12` before starting Streamlit.")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts(MODEL_DIR)

try:
    model, note_to_int, int_to_note, config = get_artifacts()
except Exception as exc:
    st.error(f"The model could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("### Generation studio")
    length = st.slider("Generation length (notes)", min_value=50, max_value=300, value=100, step=10)
    temperature = st.slider("Temperature", min_value=0.5, max_value=1.5, value=0.8, step=0.1, help="Lower values are more predictable; higher values are more creative.")
    seed_options = sorted([*MIDI_DIR.glob("*.mid"), *MIDI_DIR.glob("*.midi")])
    seed_path = st.selectbox("Seed composition", seed_options, format_func=lambda path: path.stem)
    st.caption("The model is loaded once per session and never retrained during generation.")

left, right = st.columns([1.3, .7], gap="large")
with left:
    st.markdown("#### Create a new sequence")
    st.markdown('<div class="control-card">Choose a seed, shape the output with temperature, and generate a new MIDI file. Each run creates a unique downloadable artifact.</div>', unsafe_allow_html=True)
    generate = st.button("♫  Generate Music", use_container_width=True, type="primary")
    if generate:
        try:
            with st.spinner("Composing a new sequence..."):
                seed_tokens = extract_tokens(seed_path)
                generated_tokens = generate_tokens(model, seed_tokens, note_to_int, int_to_note, int(config["sequence_length"]), length, temperature)
                filename = f"generated_music_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid"
                output_path = write_tokens_to_midi(generated_tokens, GENERATED_DIR / filename)
            st.session_state["generated_path"] = str(output_path)
            st.session_state["generated_count"] = len(generated_tokens)
            st.session_state["generated_temperature"] = temperature
            st.success("Generation completed successfully.")
        except Exception as exc:
            st.error(f"Generation failed safely: {exc}")

    if "generated_path" in st.session_state:
        output_path = Path(st.session_state["generated_path"])
        st.markdown("#### Your generated MIDI")
        metric_cols = st.columns(3)
        metric_cols[0].markdown(f'<div class="metric-box"><strong>{st.session_state["generated_count"]}</strong><small>notes generated</small></div>', unsafe_allow_html=True)
        metric_cols[1].markdown(f'<div class="metric-box"><strong>{st.session_state["generated_temperature"]:.1f}</strong><small>temperature</small></div>', unsafe_allow_html=True)
        metric_cols[2].markdown(f'<div class="metric-box"><strong>.mid</strong><small>playable format</small></div>', unsafe_allow_html=True)
        st.download_button("Download MIDI", data=output_path.read_bytes(), file_name=output_path.name, mime="audio/midi", use_container_width=True)

with right:
    st.markdown("#### Model snapshot")
    st.markdown(f'<div class="status">LSTM model ready<br><small>{config["vocab_size"]} musical tokens · sequence length {config["sequence_length"]}</small></div>', unsafe_allow_html=True)
    st.markdown("\n")
    st.markdown("**Temperature guide**")
    st.markdown("Lower settings create safer, more predictable continuations. Higher settings broaden the sampling distribution for more surprising musical ideas.")
    st.markdown("**Output**")
    st.markdown("MIDI is the primary output because it is compact, editable, and playable in any MIDI-compatible digital audio workstation.")
