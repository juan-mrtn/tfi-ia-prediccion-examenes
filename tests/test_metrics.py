"""Tests de las métricas de validación y predicción."""

import numpy as np
import pytest

from src.metrics import (exactitud_categoria, mae, pureza, tasa_compresion, tasa_resets,
                         varianza_por_nodo)


def test_pureza_ejemplo_del_informe():
    # Un nodo con 8 "Regular" y 2 "Recuperatorio" aporta 8 de 10: pureza 80%.
    nodos = [0] * 10
    categorias = ["Regular"] * 8 + ["Recuperatorio"] * 2
    assert pureza(nodos, categorias) == pytest.approx(0.8)


def test_pureza_ignora_alumnos_sin_nodo():
    assert pureza([0, 0, -1], ["A", "A", "B"]) == pytest.approx(1.0)


def test_tasas():
    assert tasa_compresion(871, 8000) == pytest.approx(0.108875)
    assert tasa_resets(1600, 8000) == pytest.approx(0.2)


def test_mae():
    assert mae([7.0, 6.5], [7.2, 6.0]) == pytest.approx(0.35)


def test_exactitud_categoria():
    assert exactitud_categoria(["A", "B", "C", "A"], ["A", "B", "A", "B"]) == pytest.approx(0.5)


def test_varianza_por_nodo():
    # Nodo 0: notas 4 y 6 (varianza 1). Nodo 1: nota única (varianza 0). Media 0,5.
    assert varianza_por_nodo(np.array([0, 0, 1]), np.array([4.0, 6.0, 9.0])) == pytest.approx(0.5)
