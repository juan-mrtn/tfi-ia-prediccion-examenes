"""Fase 3: red ART 1 según Box 3 (Lau, 1992)."""

from typing import Tuple

import numpy as np


class CarpenterGrossbergNetwork:
    """Red Carpenter-Grossberg 1 con aprendizaje rápido (fast learning).

    Pesos iniciales: b_ij(0) = 1 / (1 + N) y t_ji(0) = 1.
    Vigilancia: ||T_j* ∩ X|| / ||X|| > rho. Si falla, reset del nodo y nueva búsqueda.
    Adaptación: t_j* = t_j* AND x y b_j* = t_j* / (0.5 + ||t_j*||).
    """

    def __init__(self, input_dim: int, max_categories: int = 500, rho: float = 0.75):
        if not 0.0 < rho <= 1.0:
            raise ValueError("rho debe estar en (0, 1].")
        self.input_dim = input_dim
        self.max_categories = max_categories
        self.rho = rho
        self.b = np.full((max_categories, input_dim), 1.0 / (1.0 + input_dim))
        self.t = np.ones((max_categories, input_dim), dtype=np.int8)
        self.num_active_clusters = 0
        self.total_resets = 0
        self.empty_inputs = 0

    def evaluate_pattern(self, x: np.ndarray, update: bool = True) -> Tuple[int, int, bool]:
        """Presenta un vector y devuelve (nodo ganador, resets, resonó).

        Con update=False no se modifican los pesos (modo inferencia). Si ningún nodo
        resuena en inferencia, devuelve el de mayor activación con resonó = False.
        Un vector sin unos (||X|| = 0) no puede evaluarse: devuelve (-1, 0, False).
        """
        norm_x = int(x.sum())
        if norm_x == 0:
            self.empty_inputs += 1
            return -1, 0, False
        m = self.num_active_clusters
        mu = self.b[:m] @ x
        overlap = self.t[:m] @ x
        resets = 0
        for j in np.argsort(-mu, kind="stable"):
            if overlap[j] / norm_x > self.rho:
                if update:
                    self.total_resets += resets
                    self._adapt(j, x)
                return int(j), resets, True
            resets += 1
        if not update:
            return (int(np.argmax(mu)) if m else -1), resets, False
        if m >= self.max_categories:
            raise RuntimeError("Se alcanzó la capacidad máxima de categorías en F2.")
        self.total_resets += resets
        self.num_active_clusters += 1
        self._adapt(m, x)
        return m, resets, True

    def _adapt(self, j: int, x: np.ndarray) -> None:
        self.t[j] = self.t[j] & x
        self.b[j] = self.t[j] / (0.5 + self.t[j].sum())

    def fit_predict(self, X_bin: np.ndarray, max_epochs: int = 5) -> np.ndarray:
        """Presenta los vectores en épocas sucesivas hasta que las asignaciones se estabilizan."""
        labels = np.full(len(X_bin), -1)
        for _ in range(max_epochs):
            self.total_resets = 0
            new = np.array([self.evaluate_pattern(x)[0] for x in X_bin])
            if np.array_equal(new, labels):
                break
            labels = new
        return labels

    def predict(self, X_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inferencia sin aprendizaje: devuelve (nodos, resonó)."""
        out = [self.evaluate_pattern(x, update=False) for x in X_bin]
        return np.array([o[0] for o in out]), np.array([o[2] for o in out])
