from pathlib import Path

import numpy as np
from music21 import chord, note, stream

from preprocessing import build_vocabulary, create_sequences, extract_tokens, write_tokens_to_midi


def make_midi(path: Path) -> Path:
    score = stream.Stream()
    score.append(note.Note("C4", quarterLength=1))
    score.append(chord.Chord(["E4", "G4"], quarterLength=1))
    score.append(note.Note("D4", quarterLength=1))
    score.write("midi", fp=str(path))
    return path


def test_midi_parsing_and_note_extraction(tmp_path):
    path = make_midi(tmp_path / "sample.mid")
    tokens = extract_tokens(path)
    assert path.exists()
    assert "C4" in tokens
    assert any(token.startswith("CHORD_") for token in tokens)


def test_vocabulary_and_sequence_creation():
    tokens = ["C4", "D4", "E4", "F4", "G4", "A4"]
    note_to_int, int_to_note = build_vocabulary(tokens)
    x, y = create_sequences(tokens, sequence_length=3, note_to_int=note_to_int)
    assert x.shape == (3, 3)
    assert y.shape == (3,)
    assert int_to_note[note_to_int["C4"]] == "C4"
    assert x.dtype == np.int32


def test_generated_midi_file_is_written(tmp_path):
    output = write_tokens_to_midi(["C4", "E4", "G4", "CHORD_C4.G4"], tmp_path / "generated.mid")
    assert output.exists()
    assert output.stat().st_size > 0
