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
    # La categoría sale de la nota esperada; el perfil EM del nodo va aparte.
    assert list(out["categoria_pronosticada"]) == ["Promoción", "Riesgo Crítico / Abandono"]
    assert list(out["perfil_em_nodo"]) == ["A", "B"]
    assert not out["confianza_baja"].any()


def test_confianza_baja_usa_el_nodo_que_mas_cubre():
    # Con rho alto, [1, 1, 1, 0] no resuena con ningún nodo. El nodo 0 cubre 2 de sus
    # 3 unos y el nodo 1 solo 1: se usa el nodo 0 y no el más activado.
    net = CarpenterGrossbergNetwork(4, rho=0.9)
    labels = net.fit_predict(X)
    pred = ExamPredictor(net).calibrate(labels, np.array(["A", "A", "B"]),
                                        np.array([8.0, 9.0, 3.0]))
    out = pred.predict(np.array([[1, 1, 1, 0]], dtype=np.int8))
    assert out["confianza_baja"].iloc[0]
    assert out["nota_esperada"].iloc[0] == 8.5
