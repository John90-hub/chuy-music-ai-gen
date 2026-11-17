"""Train the LSTM music generator and save inference artifacts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from model import build_model, save_config
from preprocessing import DEFAULT_SEQUENCE_LENGTH, build_vocabulary, create_sequences, load_token_corpus, save_vocabulary


def train(midi_dir: str | Path, model_dir: str | Path, sequence_length: int = DEFAULT_SEQUENCE_LENGTH, epochs: int = 12, batch_size: int = 32, seed: int = 42) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    midi_dir = Path(midi_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tokens = load_token_corpus(midi_dir)
    if len(tokens) <= sequence_length + 1:
        sequence_length = max(8, min(sequence_length, len(tokens) // 3))
    note_to_int, int_to_note = build_vocabulary(tokens)
    x, y = create_sequences(tokens, sequence_length, note_to_int)
    split = max(1, int(len(x) * 0.9))
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]
    if len(x_val) == 0:
        x_val, y_val = x_train[-1:], y_train[-1:]
    model = build_model(len(note_to_int), sequence_length)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_dir / "music_generator.keras", monitor="val_loss", save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early_stopping], verbose=2)
    model.save(model_dir / "music_generator.keras")
    save_vocabulary(note_to_int, int_to_note, model_dir)
    config = {
        "sequence_length": sequence_length,
        "vocab_size": len(note_to_int),
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "dataset_directory": str(midi_dir),
        "token_count": len(tokens),
        "training_sequences": len(x_train),
        "validation_sequences": len(x_val),
    }
    save_config(config, model_dir / "config.json")
    (model_dir / "history.json").write_text(json.dumps({key: [float(value) for value in values] for key, values in history.history.items()}, indent=2), encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AI Music Generator LSTM")
    parser.add_argument("--midi-dir", default="data/midi")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = train(**vars(args))
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
