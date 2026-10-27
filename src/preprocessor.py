"""Fase 2A: ingesta, chequeo de 24 hs, escala base 100 y separación train/test."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

METADATA_COLUMNS = ["variable", "rol", "tipo", "min", "max", "escalar_a_100", "binning"]


class DataPreprocessor:
    """Prepara el dataset crudo según las reglas de student_metadata.csv.

    Decisiones aplicadas (ver docs/Registro_de_decisiones.docx):
        D-03: solo las variables con escalar_a_100 = si se llevan a base 100.
        D-05: el filtro de 24 hs es un chequeo de calidad de datos.
        D-06: las horas no se escalan; se exportan con 3 dígitos (000 a 024).
    """

    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path
        self.metadata_df = self._load_metadata(metadata_path)
        self.invalid_rows_: int = 0

    @staticmethod
    def _load_metadata(path: str) -> pd.DataFrame:
        meta = pd.read_csv(path)
        missing = [c for c in METADATA_COLUMNS if c not in meta.columns]
        if missing:
            raise ValueError(f"Faltan columnas en el metadata: {missing}")
        return meta.set_index("variable")

    def columns_by_role(self, rol: str) -> List[str]:
        """Devuelve las variables del metadata con el rol indicado."""
        return list(self.metadata_df.index[self.metadata_df["rol"] == rol])

    @property
    def feature_cols(self) -> List[str]:
        return self.columns_by_role("entrada")

    @property
    def target_col(self) -> str:
        return self.columns_by_role("objetivo")[0]

    def load(self, data_path: str) -> pd.DataFrame:
        """Lee el CSV crudo y verifica que estén todas las variables del metadata."""
        df = pd.read_csv(data_path)
        missing = [c for c in self.metadata_df.index if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en el dataset: {missing}")
        if df[list(self.metadata_df.index)].isnull().any().any():
            raise ValueError("El dataset tiene valores vacíos.")
        return df

    def validate_hourly_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Descarta los registros con sleep_hours + max(study_hours, internet_usage) > 24."""
        total = df["sleep_hours"] + df[["study_hours", "internet_usage"]].max(axis=1)
        valid = total <= 24
        self.invalid_rows_ = int((~valid).sum())
        return df[valid].reset_index(drop=True)

    def scale_to_base_100_int(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convierte las variables de entrada a enteros en [min, max].

        Las marcadas con escalar_a_100 = si se transforman con
        valor * 100 / (Max - Min). Todas se redondean al entero más cercano.
        """
        out = df.copy()
        for col in self.feature_cols:
            row = self.metadata_df.loc[col]
            values = out[col].astype(float)
            upper = float(row["max"])
            if row["escalar_a_100"] == "si":
                values = values * 100.0 / (float(row["max"]) - float(row["min"]))
                upper = 100.0
            out[col] = np.clip(np.floor(values + 0.5), 0, upper).astype(int)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica el chequeo de 24 hs y el escalado; devuelve entradas + objetivo."""
        df = self.validate_hourly_consistency(df)
        df = self.scale_to_base_100_int(df)
        return df[self.feature_cols + [self.target_col]]

    @staticmethod
    def train_test_split(df: pd.DataFrame, test_size: float = 0.2,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separa el 80% para entrenamiento y el 20% reservado para prueba."""
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size debe estar entre 0 y 1.")
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(df))
        n_test = int(round(len(df) * test_size))
        test = df.iloc[idx[:n_test]].reset_index(drop=True)
        train = df.iloc[idx[n_test:]].reset_index(drop=True)
        return train, test

    def save_csv(self, df: pd.DataFrame, path: str, digits: Optional[int] = 3) -> None:
        """Exporta las entradas enteras con ceros a la izquierda (ej. 7 -> 007)."""
        out = df.copy()
        if digits:
            for col in self.feature_cols:
                if col in out.columns:
                    out[col] = out[col].map(lambda v: str(int(v)).zfill(digits))
        out.to_csv(path, index=False)
