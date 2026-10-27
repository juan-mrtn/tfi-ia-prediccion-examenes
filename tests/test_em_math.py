"""Tests de EMClusterer."""

import numpy as np
import pytest

from src.em_clustering import CATEGORIES, EMClusterer, academic_weights


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [8, 8], [0, 8]])
    return np.vstack([c + rng.normal(size=(150, 2)) for c in centers])


def test_bic_selects_true_number_of_groups(blobs):
    em = EMClusterer(n_init=2)
    assert em.find_optimal_k(blobs, range(2, 6)) == 3
    assert list(em.selection_.columns) == ["K", "log_likelihood", "BIC", "AIC", "convergio"]


def test_eigenvalues_sorted_and_sum_to_trace(blobs):
    em = EMClusterer(n_components=3, n_init=2).fit(blobs)
    for k, cov in enumerate(em.model.covariances_):
        values = em.eigenvalues_[k]
        assert np.all(np.diff(values) <= 0)
        assert values.sum() == pytest.approx(np.trace(cov))


def test_risk_assigns_valid_categories(blobs):
    em = EMClusterer(n_components=3, n_init=2).fit(blobs)
    cats = em.compute_spectral_risk(np.array([1.0, 1.0]))
    assert set(cats.values()) <= set(CATEGORIES)
    assert (em.risk_table_["alpha_k"] >= 0).all()


def test_predict_returns_probabilities(blobs):
    em = EMClusterer(n_components=3, n_init=2).fit(blobs)
    labels, resp = em.predict(blobs)
    assert labels.shape == (len(blobs),)
    assert np.allclose(resp.sum(axis=1), 1.0)


def test_metrics_keys(blobs):
    em = EMClusterer(n_components=3, n_init=2).fit(blobs)
    m = em.metrics(blobs)
    assert m["silhouette"] > 0.5
    assert {"BIC", "AIC", "log_likelihood", "entropia_responsabilidades"} <= set(m)


def test_academic_weights_sign():
    w = academic_weights(["study_hours", "internet_usage"])
    assert list(w) == [1.0, -1.0]


def test_risk_before_fit_raises():
    with pytest.raises(RuntimeError):
        EMClusterer().compute_spectral_risk(np.ones(2))
