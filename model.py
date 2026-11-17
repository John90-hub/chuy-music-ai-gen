"""LSTM model construction and persistence helpers."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def build_model(vocab_size: int, sequence_length: int, embedding_dim: int = 64, lstm_units: int = 96, dropout: float = 0.25) -> tf.keras.Model:
    """Build a practical stacked LSTM next-token classifier."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length,), dtype="int32"),
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
            tf.keras.layers.LSTM(lstm_units, return_sequences=True),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(vocab_size, activation="softmax"),
        ],
        name="music_generator_lstm",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_config(config: dict, output_path: str | Path) -> None:
    import json

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(config, indent=2), encoding="utf-8")
