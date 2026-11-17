"""Inference and MIDI generation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import tensorflow as tf

from preprocessing import extract_tokens, write_tokens_to_midi


def sample_with_temperature(probabilities: Sequence[float], temperature: float = 1.0, rng: np.random.Generator | None = None) -> int:
    """Sample an index from a probability vector using temperature scaling."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or probs.size == 0 or not np.isfinite(probs).all() or probs.sum() <= 0:
        raise ValueError("probabilities must be a non-empty finite vector with positive sum")
    logits = np.log(np.maximum(probs, 1e-8)) / temperature
    logits -= logits.max()
    scaled = np.exp(logits)
    scaled /= scaled.sum()
    rng = rng or np.random.default_rng()
    return int(rng.choice(len(scaled), p=scaled))


def load_artifacts(model_dir: str | Path):
    model_dir = Path(model_dir)
    model = tf.keras.models.load_model(model_dir / "music_generator.keras")
    note_to_int = json.loads((model_dir / "note_to_int.json").read_text(encoding="utf-8"))
    int_to_note = {int(key): value for key, value in json.loads((model_dir / "int_to_note.json").read_text(encoding="utf-8")).items()}
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return model, note_to_int, int_to_note, config


def generate_tokens(model, seed_tokens: Sequence[str], note_to_int: dict[str, int], int_to_note: dict[int, str], sequence_length: int, length: int, temperature: float = 0.8, rng: np.random.Generator | None = None, predictor: Callable | None = None) -> List[str]:
    """Generate a continuation from a seed token sequence."""
    if length < 1:
        raise ValueError("length must be positive")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    known_seed = [token for token in seed_tokens if token in note_to_int]
    if not known_seed:
        raise ValueError("seed_tokens must contain at least one known token")
    generated = list(known_seed[-sequence_length:])
    rng = rng or np.random.default_rng()
    predict = predictor or model.predict
    for _ in range(length):
        window = generated[-sequence_length:]
        if len(window) < sequence_length:
            window = [known_seed[0]] * (sequence_length - len(window)) + window
        encoded = np.array([[note_to_int[token] for token in window]], dtype=np.int32)
        probabilities = np.asarray(predict(encoded, verbose=0))[0]
        next_index = sample_with_temperature(probabilities, temperature, rng)
        generated.append(int_to_note[next_index])
    return generated[-length:]


def generate_midi_from_seed(model_dir: str | Path, seed_midi: str | Path, output_path: str | Path, length: int = 100, temperature: float = 0.8, rng: np.random.Generator | None = None) -> Path:
    model, note_to_int, int_to_note, config = load_artifacts(model_dir)
    seed_tokens = extract_tokens(seed_midi)
    generated = generate_tokens(model, seed_tokens, note_to_int, int_to_note, int(config["sequence_length"]), length, temperature, rng)
    return write_tokens_to_midi(generated, output_path)
