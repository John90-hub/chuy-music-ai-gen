"""MIDI parsing and sequence preparation for the LSTM music generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from music21 import chord, converter, note, stream

DEFAULT_SEQUENCE_LENGTH = 32


def element_to_token(element) -> str:
    """Convert a music21 Note or Chord into a stable symbolic token."""
    if isinstance(element, note.Note):
        return element.pitch.nameWithOctave
    if isinstance(element, chord.Chord):
        pitches = ".".join(sorted(p.nameWithOctave for p in element.pitches))
        return f"CHORD_{pitches}"
    raise TypeError(f"Unsupported music element: {type(element).__name__}")


def extract_tokens(midi_path: str | Path) -> List[str]:
    """Parse a MIDI file and extract note/chord pitch tokens."""
    score = converter.parse(str(midi_path))
    tokens: List[str] = []
    for element in score.recurse().notes:
        try:
            tokens.append(element_to_token(element))
        except TypeError:
            continue
    return tokens


def load_token_corpus(midi_dir: str | Path) -> List[str]:
    """Load and concatenate token sequences from all MIDI files in a directory."""
    midi_dir = Path(midi_dir)
    paths = sorted({*midi_dir.glob("*.mid"), *midi_dir.glob("*.midi")})
    if not paths:
        raise FileNotFoundError(f"No MIDI files found in {midi_dir}")
    corpus: List[str] = []
    for path in paths:
        try:
            tokens = extract_tokens(path)
            if tokens:
                corpus.extend(tokens + ["<SEP>"])
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
    if len(corpus) < 3:
        raise ValueError("The MIDI corpus did not contain enough parseable musical events")
    return corpus


def build_vocabulary(tokens: Sequence[str]) -> Tuple[dict[str, int], dict[int, str]]:
    unique_tokens = sorted(set(tokens))
    note_to_int = {token: index for index, token in enumerate(unique_tokens)}
    int_to_note = {index: token for token, index in note_to_int.items()}
    return note_to_int, int_to_note


def create_sequences(tokens: Sequence[str], sequence_length: int, note_to_int: dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Create X windows and y next-token labels."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if len(tokens) <= sequence_length:
        raise ValueError("Not enough tokens for the requested sequence length")
    encoded = [note_to_int[token] for token in tokens]
    x = np.array([encoded[index:index + sequence_length] for index in range(len(encoded) - sequence_length)], dtype=np.int32)
    y = np.array([encoded[index + sequence_length] for index in range(len(encoded) - sequence_length)], dtype=np.int32)
    return x, y


def save_vocabulary(note_to_int: dict[str, int], int_to_note: dict[int, str], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "note_to_int.json").write_text(json.dumps(note_to_int, indent=2), encoding="utf-8")
    (output_dir / "int_to_note.json").write_text(json.dumps({str(key): value for key, value in int_to_note.items()}, indent=2), encoding="utf-8")


def tokens_to_stream(tokens: Iterable[str], step: float = 0.5) -> stream.Stream:
    """Convert generated tokens to a playable monophonic/chordal music21 stream."""
    result = stream.Stream()
    offset = 0.0
    for token in tokens:
        if token == "<SEP>":
            offset += step
            continue
        try:
            if token.startswith("CHORD_"):
                pitches = token.removeprefix("CHORD_").split(".")
                element = chord.Chord(pitches)
            else:
                element = note.Note(token)
            element.duration.quarterLength = step
            result.insert(offset, element)
            offset += step
        except Exception:
            continue
    return result


def write_tokens_to_midi(tokens: Iterable[str], output_path: str | Path, step: float = 0.5) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens_to_stream(tokens, step=step).write("midi", fp=str(output_path))
    return output_path
