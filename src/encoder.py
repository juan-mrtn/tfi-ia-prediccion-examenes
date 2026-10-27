"""Fase 2B: histograma de 10 bins y codificación binaria para la capa F1 de ART 1."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# D-07: bins no uniformes para las horas (0–1→0, 2→1, ..., 8→7, 9–10→8, 11–12→9, >12→10).
HOUR_BINS: Dict[int, int] = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7,
                             9: 8, 10: 8, 11: 9, 12: 9}
MAX_BIN = 10


def hour_to_bin(hours: int) -> int:
    """Convierte horas diarias en bin según D-07."""
    return HOUR_BINS.get(int(hours), MAX_BIN)


def uniform_to_bin(value: int) -> int:
    """Convierte un valor entero en [0, 100] en bin = min(valor // 10, 10)."""
    return min(int(value) // 10, MAX_BIN)


class PositionalBinaryEncoder:
    """Discretiza cada variable en bins 0..10 y los codifica en 4 bits posicionales (6 -> 0110)."""

    def __init__(self, binning: Dict[str, str], bits_per_feature: int = 4):
        self.binning = binning
        self.bits_per_feature = bits_per_feature

    @staticmethod
    def int_to_4bit_positional(bin_index: int) -> List[int]:
        """Codifica un entero 0..10 en binario de 4 bits (0 -> 0000, 10 -> 1010)."""
        if not 0 <= bin_index <= MAX_BIN:
            raise ValueError("El bin debe estar entre 0 y 10.")
        return [int(b) for b in format(bin_index, "04b")]

    def to_bins(self, df_scaled_int: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Devuelve el índice de bin (0..10) de cada variable."""
        out = pd.DataFrame(index=df_scaled_int.index)
        for col in feature_cols:
            rule = hour_to_bin if self.binning[col] == "horas" else uniform_to_bin
            out[col] = df_scaled_int[col].map(rule).astype(int)
        return out

    def fit_transform(self, df_scaled_int: pd.DataFrame,
                      feature_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Devuelve (X_binario, nombres de bits) con X en {0, 1}^(D * bits_por_variable)."""
        encode = self.int_to_4bit_positional
        bins = self.to_bins(df_scaled_int, feature_cols)
        blocks = [np.array([encode(b) for b in bins[col]], dtype=np.int8) for col in feature_cols]
        X = np.hstack(blocks)
        n = self.bits_per_feature
        names = [f"{col}_b{n - 1 - i}" for col in feature_cols for i in range(n)]
        return X, names
