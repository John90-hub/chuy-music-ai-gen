"""Lightweight polyphonic audio rendering and MP3 encoding for generated tokens."""

from __future__ import annotations

from typing import Sequence

import lameenc
import numpy as np
from music21 import pitch

SAMPLE_RATE = 22_050


def _token_frequencies(token: str) -> list[float]:
    """Convert a note or chord token into frequencies, ignoring invalid pitches."""
    if token == "<SEP>":
        return []
    raw_pitches = token.removeprefix("CHORD_").split(".") if token.startswith("CHORD_") else [token]
    frequencies: list[float] = []
    for value in raw_pitches:
        try:
            frequency = float(pitch.Pitch(value).frequency)
            if np.isfinite(frequency) and frequency > 0:
                frequencies.append(frequency)
        except (ValueError, TypeError):
            continue
    return frequencies


def _envelope(length: int) -> np.ndarray:
    """Create a short attack/release envelope to avoid clicks between notes."""
    if length <= 1:
        return np.ones(max(1, length), dtype=np.float32)
    attack = min(int(SAMPLE_RATE * 0.012), max(1, length // 4))
    release = min(int(SAMPLE_RATE * 0.045), max(1, length // 3))
    result = np.ones(length, dtype=np.float32)
    result[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False, dtype=np.float32)
    result[-release:] = np.linspace(1.0, 0.0, release, endpoint=True, dtype=np.float32)
    return result


def _render_note(frequencies: Sequence[float], duration: float, timbre: str) -> np.ndarray:
    samples = max(1, int(SAMPLE_RATE * duration))
    if not frequencies:
        return np.zeros(samples, dtype=np.float32)

    time = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
    sound = np.zeros(samples, dtype=np.float32)
    wave = timbre.lower()
    for frequency in frequencies:
        phase = 2.0 * np.pi * frequency * time
        if wave == "bright":
            voice = 0.68 * np.sin(phase) + 0.22 * np.sin(2.0 * phase) + 0.10 * np.sin(3.0 * phase)
        elif wave == "soft":
            voice = np.sin(phase) * np.exp(-2.2 * time / max(duration, 1e-3))
        else:  # Warm, rounded tone with a gentle second harmonic.
            voice = 0.84 * np.sin(phase) + 0.16 * np.sin(2.0 * phase)
        sound += voice.astype(np.float32, copy=False)

    sound /= max(1.0, len(frequencies) ** 0.5)
    sound *= _envelope(samples)
    return sound


def render_tokens_to_mp3(
    tokens: Sequence[str],
    *,
    tempo: int = 110,
    beats_per_note: float = 0.5,
    timbre: str = "Warm",
    bitrate: int = 128,
) -> bytes:
    """Synthesize note/chord tokens and return encoded MP3 bytes.

    This lightweight software synth is intended for previews and demos; it does
    not require a separate soundfont or system audio executable.
    """
    if tempo <= 0:
        raise ValueError("tempo must be positive")
    if beats_per_note <= 0:
        raise ValueError("beats_per_note must be positive")
    if timbre.lower() not in {"warm", "soft", "bright"}:
        raise ValueError("timbre must be Warm, Soft, or Bright")
    if not tokens:
        raise ValueError("at least one note token is required")

    duration = 60.0 / tempo * beats_per_note
    audio = np.concatenate([_render_note(_token_frequencies(token), duration, timbre) for token in tokens])
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio * (0.92 / max(peak, 0.92))
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()

    encoder = lameenc.Encoder()
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(1)
    encoder.set_bit_rate(bitrate)
    encoder.set_quality(5)
    return encoder.encode(pcm) + encoder.flush()
