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

    Exam Score nunca forma parte del vector de entrada X: solo calibra los nodos.
    """

    def __init__(self, network: CarpenterGrossbergNetwork):
        self.network = network
        self.node_expected_score: Dict[int, float] = {}
        self.node_category: Dict[int, str] = {}
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
        return self

    def predict(self, X_bin: np.ndarray) -> pd.DataFrame:
        """Devuelve nodo, nota esperada, categoría y si la confianza es baja."""
        nodes, resonated = self.network.predict(X_bin)
        scores = [self.node_expected_score.get(int(n), self.global_score) for n in nodes]
        cats = [self.node_category.get(int(n), score_to_category(s))
                for n, s in zip(nodes, scores)]
        return pd.DataFrame({"nodo_art1": nodes, "nota_esperada": np.round(scores, 2),
                             "categoria_pronosticada": cats, "confianza_baja": ~resonated})
