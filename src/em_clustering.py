"""Fase 1: clustering EM/GMM, selección de K por BIC/AIC y riesgo por eigenvectores."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

CATEGORIES = ["Promoción", "Aprobado Regular", "Recuperatorio", "Riesgo Crítico / Abandono"]
EPSILON = 1e-9


class EMClusterer:
    """Agrupa a los estudiantes con una mezcla de gaussianas (EM) sobre las variables sin adaptar.

    Decisión D-08: K se elige en [2, 7] por mínimo BIC; el AIC se reporta como control.
    reg_covar se suma a la diagonal de Σ_k para evitar covarianzas nulas cuando un grupo
    tiene muchos valores idénticos (por ejemplo attendance = 100 o study_hours = 0).
    """

    def __init__(self, n_components: Optional[int] = None, max_iter: int = 200,
                 n_init: int = 5, reg_covar: float = 1e-3, random_state: int = 42):
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.model: Optional[GaussianMixture] = None
        self.selection_: Optional[pd.DataFrame] = None
        self.eigenvalues_: Dict[int, np.ndarray] = {}
        self.eigenvectors_: Dict[int, np.ndarray] = {}
        self.risk_table_: Optional[pd.DataFrame] = None
        self.risk_categories_: Dict[int, str] = {}

    def _new_model(self, k: int) -> GaussianMixture:
        return GaussianMixture(n_components=k, covariance_type="full", max_iter=self.max_iter,
                               n_init=self.n_init, reg_covar=self.reg_covar,
                               random_state=self.random_state)

    def find_optimal_k(self, X_cont: np.ndarray, k_range: range = range(2, 8)) -> int:
        """Ajusta un GMM por cada K y elige el de menor BIC. Guarda la tabla en selection_."""
        rows = []
        for k in k_range:
            gmm = self._new_model(k).fit(X_cont)
            rows.append({"K": k,
                         "log_likelihood": gmm.score(X_cont) * len(X_cont),
                         "BIC": gmm.bic(X_cont),
                         "AIC": gmm.aic(X_cont),
                         "convergio": gmm.converged_})
        self.selection_ = pd.DataFrame(rows)
        best = int(self.selection_.loc[self.selection_["BIC"].idxmin(), "K"])
        self.n_components = best
        return best

    def fit(self, X_cont: np.ndarray) -> "EMClusterer":
        """Ajusta el GMM definitivo (elige K por BIC si no fue fijado) y descompone Σ_k."""
        if self.n_components is None:
            self.find_optimal_k(X_cont)
        self.model = self._new_model(self.n_components).fit(X_cont)
        for k, cov in enumerate(self.model.covariances_):
            values, vectors = np.linalg.eigh(cov)
            order = np.argsort(values)[::-1]
            self.eigenvalues_[k] = values[order]
            self.eigenvectors_[k] = vectors[:, order]
        return self

    def compute_spectral_risk(self, feature_weights: np.ndarray) -> Dict[int, str]:
        """Calcula R_k = V_k / (P_k + eps) * (1 - alpha_k) y asigna las 4 categorías por ranking.

        P_k = w·μ_k, alpha_k = w·v_k1 y V_k = Tr(Σ_k), con w normalizado.
        El signo de un autovector es arbitrario: se orienta v_k1 para que alpha_k >= 0.
        """
        if self.model is None:
            raise RuntimeError("Primero hay que ajustar el modelo con fit().")
        w = np.asarray(feature_weights, dtype=float)
        w = w / np.linalg.norm(w)
        rows = []
        for k in range(self.n_components):
            v1 = self.eigenvectors_[k][:, 0]
            alpha = float(w @ v1)
            if alpha < 0:
                v1, alpha = -v1, -alpha
            perf = float(w @ self.model.means_[k])
            volume = float(self.eigenvalues_[k].sum())
            risk = volume / (perf + EPSILON) * (1.0 - alpha)
            rows.append({"componente": k, "peso": float(self.model.weights_[k]),
                         "P_k": perf, "alpha_k": alpha, "V_k": volume, "R_k": risk})
        table = pd.DataFrame(rows)
        table["rank"] = table["R_k"].rank(method="first").astype(int)
        table["categoria"] = table["rank"].map(lambda r: CATEGORIES[min(r, 4) - 1])
        self.risk_table_ = table
        self.risk_categories_ = dict(zip(table["componente"], table["categoria"]))
        return self.risk_categories_

    def predict(self, X_cont: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve (labels, responsabilidades γ_ik) para cada instancia."""
        if self.model is None:
            raise RuntimeError("Primero hay que ajustar el modelo con fit().")
        return self.model.predict(X_cont), self.model.predict_proba(X_cont)

    def metrics(self, X_cont: np.ndarray, sample_size: int = 3000) -> Dict[str, float]:
        """Métricas de EM: log-likelihood, BIC, AIC, silhouette y entropía de responsabilidades."""
        labels, resp = self.predict(X_cont)
        entropy = float(-(resp * np.log(resp + EPSILON)).sum(axis=1).mean())
        n = min(sample_size, len(X_cont))
        sil = (silhouette_score(X_cont, labels, sample_size=n, random_state=self.random_state)
               if len(set(labels)) > 1 else float("nan"))
        return {"K": self.n_components,
                "log_likelihood": float(self.model.score(X_cont) * len(X_cont)),
                "BIC": float(self.model.bic(X_cont)),
                "AIC": float(self.model.aic(X_cont)),
                "silhouette": float(sil),
                "entropia_responsabilidades": entropy}


def academic_weights(feature_cols: List[str]) -> np.ndarray:
    """Vector w_acad: +1 para hábitos favorables, -1 para internet_usage."""
    negative = {"internet_usage"}
    return np.array([-1.0 if c in negative else 1.0 for c in feature_cols])
