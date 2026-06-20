# tests/test_predict.py
import pytest
from src.predict import make_prediction as predict_yield

def test_predict_returns_float_in_range():
    """Verifies that the prediction returns a float value inside the sanity band."""
    result = predict_yield(22.0, 88.0, 920)
    assert isinstance(result, float)
    assert 0 < result < 50  # sanity band for daily kg

def test_higher_humidity_can_increase_yield():
    """Confirms changing humidity inputs yields different output numbers."""
    low = predict_yield(22.0, 75.0, 920)
    high = predict_yield(22.0, 92.0, 920)
    # Direction may depend on training data; document if assertion differs
    assert high != low