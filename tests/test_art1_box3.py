"""Tests de la red ART 1 (Box 3, Lau 1992)."""

import numpy as np
import pytest

from src.art1 import CarpenterGrossbergNetwork

PATTERNS = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.int8)


def test_initial_weights():
    net = CarpenterGrossbergNetwork(4, max_categories=3)
    assert np.allclose(net.b, 1 / 5)
    assert (net.t == 1).all()


def test_fast_learning_is_and():
    net = CarpenterGrossbergNetwork(4, rho=0.5)
    net.evaluate_pattern(PATTERNS[1])
    net.evaluate_pattern(PATTERNS[0])
    assert list(net.t[0]) == [1, 1, 0, 0]
    assert net.b[0] == pytest.approx([1 / 2.5, 1 / 2.5, 0, 0])


def test_high_vigilance_creates_more_categories():
    low = CarpenterGrossbergNetwork(4, rho=0.1)
    high = CarpenterGrossbergNetwork(4, rho=0.9)
    low.fit_predict(PATTERNS)
    high.fit_predict(PATTERNS)
    assert high.num_active_clusters > low.num_active_clusters


def test_predict_does_not_change_weights():
    net = CarpenterGrossbergNetwork(4, rho=0.5)
    net.fit_predict(PATTERNS)
    b, t = net.b.copy(), net.t.copy()
    net.predict(np.array([[1, 0, 1, 0]], dtype=np.int8))
    assert np.array_equal(b, net.b) and np.array_equal(t, net.t)


def test_empty_vector_is_flagged():
    net = CarpenterGrossbergNetwork(4)
    assert net.evaluate_pattern(np.zeros(4, dtype=np.int8)) == (-1, 0, False)
    assert net.empty_inputs == 1


def test_invalid_rho():
    with pytest.raises(ValueError):
        CarpenterGrossbergNetwork(4, rho=1.5)
