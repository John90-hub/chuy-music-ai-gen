import numpy as np
import pytest

from generator import generate_tokens, sample_with_temperature
from model import build_model


def test_lstm_model_builds_and_predicts():
    model = build_model(vocab_size=7, sequence_length=4, embedding_dim=8, lstm_units=8)
    prediction = model.predict(np.array([[0, 1, 2, 3]], dtype=np.int32), verbose=0)
    assert prediction.shape == (1, 7)
    assert np.isclose(prediction.sum(), 1.0, atol=1e-5)


def test_temperature_sampling_is_valid_and_deterministic_with_rng():
    probabilities = [0.1, 0.2, 0.7]
    first = sample_with_temperature(probabilities, temperature=0.5, rng=np.random.default_rng(4))
    second = sample_with_temperature(probabilities, temperature=0.5, rng=np.random.default_rng(4))
    assert first == second
    assert 0 <= first < 3


def test_temperature_sampling_rejects_invalid_values():
    with pytest.raises(ValueError):
        sample_with_temperature([0.5, 0.5], temperature=0)
    with pytest.raises(ValueError):
        sample_with_temperature([], temperature=1.0)


def test_generation_uses_predictor_and_seed():
    note_to_int = {"C4": 0, "D4": 1, "E4": 2}
    int_to_note = {0: "C4", 1: "D4", 2: "E4"}

    def predictor(_encoded, verbose=0):
        return np.array([[0.0, 0.0, 1.0]])

    generated = generate_tokens(None, ["C4", "D4"], note_to_int, int_to_note, sequence_length=2, length=5, temperature=1.0, predictor=predictor)
    assert generated == ["E4"] * 5

    with pytest.raises(ValueError):
        generate_tokens(None, ["UNKNOWN"], note_to_int, int_to_note, sequence_length=2, length=2, predictor=predictor)
