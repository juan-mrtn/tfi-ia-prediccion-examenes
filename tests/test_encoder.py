"""Tests de PositionalBinaryEncoder."""

import numpy as np
import pandas as pd
import pytest

from src.encoder import PositionalBinaryEncoder, hour_to_bin, uniform_to_bin

BINNING = {"study_hours": "horas", "attendance": "uniforme"}


def test_hour_bins_follow_d07():
    expected = {0: 0, 1: 0, 2: 1, 5: 4, 8: 7, 9: 8, 10: 8, 11: 9, 12: 9, 13: 10, 24: 10}
    assert {h: hour_to_bin(h) for h in expected} == expected


def test_uniform_bins():
    assert [uniform_to_bin(v) for v in (0, 9, 10, 65, 99, 100)] == [0, 0, 1, 6, 9, 10]


def test_positional_codes():
    enc = PositionalBinaryEncoder
    assert enc.int_to_4bit_positional(0) == [0, 0, 0, 0]
    assert enc.int_to_4bit_positional(6) == [0, 1, 1, 0]
    assert enc.int_to_4bit_positional(10) == [1, 0, 1, 0]
    with pytest.raises(ValueError):
        enc.int_to_4bit_positional(11)


def test_fit_transform_shape_and_binary():
    df = pd.DataFrame({"study_hours": [0, 7, 13], "attendance": [56, 100, 5]})
    X, names = PositionalBinaryEncoder(BINNING).fit_transform(df, list(BINNING))
    assert X.shape == (3, 8) and names[:4] == [f"study_hours_b{i}" for i in (3, 2, 1, 0)]
    assert list(X[1]) == [0, 1, 1, 0, 1, 0, 1, 0]
    assert set(np.unique(X)).issubset({0, 1})
