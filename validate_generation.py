from pathlib import Path

from music21 import converter

from generator import generate_tokens, load_artifacts
from preprocessing import extract_tokens, write_tokens_to_midi

ROOT = Path(__file__).resolve().parent
model, note_to_int, int_to_note, config = load_artifacts(ROOT / "models")
seed_path = sorted((ROOT / "data" / "midi").glob("*.mid"))[0]
seed_tokens = extract_tokens(seed_path)
generated = generate_tokens(model, seed_tokens, note_to_int, int_to_note, int(config["sequence_length"]), 100, temperature=0.8)
out = write_tokens_to_midi(generated, ROOT / "generated" / "generated_music_demo.mid")
parsed = converter.parse(str(out))
print(f"seed={seed_path.name}")
print(f"generated_tokens={len(generated)}")
print(f"output={out}")
print(f"parsed_elements={len(list(parsed.recurse().notes))}")
print(f"output_bytes={out.stat().st_size}")
