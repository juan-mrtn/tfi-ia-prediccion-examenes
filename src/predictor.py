"""Fase 4: predicción del próximo examen (inferencia ART 1 sin aprendizaje)."""

from typing import Dict

import numpy as np
import pandas as pd

from src.art1 import CarpenterGrossbergNetwork


def score_to_category(score_0_10: float) -> str:
    """Categoría ordinal según la nota esperada en escala 0–10 (sección 1.3)."""
    if score_0_10 > 8:
        return "Promoción"
    if score_0_10 >= 6:
        return "Aprobado Regular"
    if score_0_10 >= 4:
        return "Recuperatorio"
    return "Riesgo Crítico / Abandono"


class ExamPredictor:
    """Calibra cada nodo ART 1 con la nota media y la categoría EM modal de sus miembros.

    La categoría pronosticada sale de la nota esperada (umbrales de score_to_category),
    así siempre coincide con la nota. El perfil EM del nodo se informa aparte.
    Exam Score nunca forma parte del vector de entrada X: solo calibra los nodos.
    """

    def __init__(self, network: CarpenterGrossbergNetwork):
        self.network = network
        self.node_expected_score: Dict[int, float] = {}
        self.node_category: Dict[int, str] = {}
        self.node_size: Dict[int, int] = {}
        self.global_score: float = float("nan")

    def calibrate(self, assigned_clusters: np.ndarray, em_categories: np.ndarray,
                  exam_scores_0_10: np.ndarray) -> "ExamPredictor":
        df = pd.DataFrame({"nodo": assigned_clusters, "cat": em_categories,
                           "nota": exam_scores_0_10})
        df = df[df["nodo"] >= 0]
        self.global_score = float(df["nota"].mean())
        for node, group in df.groupby("nodo"):
            self.node_expected_score[int(node)] = float(group["nota"].mean())
            self.node_category[int(node)] = group["cat"].mode().iloc[0]
            self.node_size[int(node)] = len(group)
        return self

    def _nodo_mas_parecido(self, x: np.ndarray) -> int:
        """Para un alumno que no resonó con ningún nodo (confianza baja).

        Elige el nodo cuyo prototipo cubre más unos del alumno (‖T ∩ X‖ / ‖X‖) y,
        si empatan, el que tiene más alumnos. Así se evita caer en un nodo de un
        solo alumno, que es el que más se activa por tener pocos unos.
        """
        if x.sum() == 0 or not self.node_size:
            return -1
        nodos = list(self.node_size)
        cubre = self.network.t[nodos] @ x
        mejor = max(range(len(nodos)), key=lambda i: (cubre[i], self.node_size[nodos[i]]))
        return nodos[mejor]

    def predict(self, X_bin: np.ndarray) -> pd.DataFrame:
        """Devuelve nodo, nota esperada, categoría, perfil EM del nodo y si la confianza es baja."""
        nodes, resonated = self.network.predict(X_bin)
        nodes = np.array([n if ok else self._nodo_mas_parecido(x)
                          for n, ok, x in zip(nodes, resonated, X_bin)])
        scores = np.round([self.node_expected_score.get(int(n), self.global_score)
                           for n in nodes], 2)
        # La categoría se calcula con la nota esperada (Promoción > 8, Regular 6–8, ...).
        cats = [score_to_category(s) for s in scores]
        perfiles = [self.node_category.get(int(n), c) for n, c in zip(nodes, cats)]
        return pd.DataFrame({"nodo_art1": nodes, "nota_esperada": scores,
                             "categoria_pronosticada": cats, "perfil_em_nodo": perfiles,
                             "confianza_baja": ~resonated})
