"""Tests de ExamPredictor."""

import numpy as np

from src.art1 import CarpenterGrossbergNetwork
from src.predictor import ExamPredictor, score_to_category

X = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8)


def test_score_to_category_limits():
    assert [score_to_category(s) for s in (9, 8, 6, 5.9, 4, 3.9)] == [
        "Promoción", "Aprobado Regular", "Aprobado Regular",
        "Recuperatorio", "Recuperatorio", "Riesgo Crítico / Abandono"]


def test_calibrate_and_predict():
    net = CarpenterGrossbergNetwork(4, rho=0.5)
    labels = net.fit_predict(X)
    pred = ExamPredictor(net).calibrate(labels, np.array(["A", "A", "B"]),
                                        np.array([8.0, 9.0, 3.0]))
    out = pred.predict(X[[0, 2]])
    assert list(out["nota_esperada"]) == [8.5, 3.0]
    assert list(out["categoria_pronosticada"]) == ["A", "B"]
    assert not out["confianza_baja"].any()
