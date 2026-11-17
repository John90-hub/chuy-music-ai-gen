"""Tests for the lightweight MP3 synthesizer."""

import pytest

from audio import render_tokens_to_mp3


def test_render_tokens_to_mp3_returns_audio_bytes():
    encoded = render_tokens_to_mp3(["C4", "CHORD_E4.G4"], tempo=120, timbre="Warm")
    assert isinstance(encoded, bytes)
    assert len(encoded) > 100


def test_render_tokens_to_mp3_rejects_invalid_settings():
    with pytest.raises(ValueError, match="tempo"):
        render_tokens_to_mp3(["C4"], tempo=0)
    with pytest.raises(ValueError, match="beats_per_note"):
        render_tokens_to_mp3(["C4"], beats_per_note=0)
    with pytest.raises(ValueError, match="timbre"):
        render_tokens_to_mp3(["C4"], timbre="Unknown")
    with pytest.raises(ValueError, match="at least one"):
        render_tokens_to_mp3([])
