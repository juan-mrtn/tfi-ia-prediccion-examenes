"""Genera un dataset SIMULADO de 10.000 alumnos con perfiles latentes (semilla fija).

El dataset de Kaggle parece sintético: las variables son casi independientes y uniformes, y el
EM no encuentra estructura (el BIC no tiene mínimo). Este script simula un comportamiento más
parecido al de una escuela real, con las mismas columnas y rangos para que el preprocesador
funcione igual. Los datos son simulados y así debe aclararse en el informe.

Con el aval de la docente (D-10), su salida reemplaza al original como dataset de trabajo en
data/raw/student_performance.csv. El original de Kaggle queda en
data/raw/student_performance_kaggle_original.csv.

Modelo generativo:
    1. Cada alumno pertenece a uno de 4 perfiles latentes (no se exporta como columna).
    2. Dentro del perfil tiene 3 factores latentes: compromiso (e), base académica (a) y
       distracción (d), con dispersión y correlación propias, así los perfiles se solapan.
    3. Las variables observadas dependen de esos factores más ruido, y se redondean a enteros.
    4. exam_score depende de las variables observadas, de la base académica y de ruido.

Uso:
    python scripts/generar_dataset_simulado.py [--n 10000] [--seed 2026] [--salida RUTA]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "student_performance.csv"
COLUMNS = ["study_hours", "attendance", "sleep_hours", "internet_usage",
           "assignments_completed", "previous_score", "exam_score", "placement_status"]

# Perfil: (proporción, media de compromiso e, base académica a, distracción d)
PROFILES = {
    "comprometido": (0.20, 1.20, 0.90, -0.80),
    "regular": (0.40, 0.15, 0.10, -0.10),
    "distraido": (0.25, -0.55, -0.05, 1.20),
    "desenganchado": (0.15, -1.40, -1.10, 0.40),
}
LATENT_SD = 0.55
LATENT_CORR_EA = 0.30  # compromiso y base académica se mueven juntos dentro del perfil


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _int(values: np.ndarray, low: int, high: int) -> np.ndarray:
    return np.clip(np.rint(values), low, high).astype(int)


def sample_latent(n: int, rng: np.random.Generator):
    """Devuelve el perfil de cada alumno y sus factores latentes (e, a, d)."""
    names = list(PROFILES)
    probs = np.array([PROFILES[p][0] for p in names])
    profile = rng.choice(len(names), size=n, p=probs / probs.sum())
    means = np.array([PROFILES[p][1:] for p in names])[profile]
    cov = LATENT_SD ** 2 * np.array([[1.0, LATENT_CORR_EA, 0.0],
                                     [LATENT_CORR_EA, 1.0, 0.0],
                                     [0.0, 0.0, 1.0]])
    latent = means + rng.multivariate_normal(np.zeros(3), cov, size=n)
    return np.array(names)[profile], latent[:, 0], latent[:, 1], latent[:, 2]


def generate(n: int = 10000, seed: int = 2026) -> pd.DataFrame:
    """Genera n alumnos simulados. Devuelve las columnas del dataset más el perfil latente."""
    rng = np.random.default_rng(seed)
    profile, e, a, d = sample_latent(n, rng)

    study = _int(4.6 + 1.2 * e - 0.5 * d + rng.normal(0, 1.8, n), 0, 12)
    internet = _int(4.8 + 1.8 * d - 0.3 * e + rng.normal(0, 1.5, n), 0, 12)
    # Quien estudia o navega muchas horas duerme menos.
    sleep = _int(7.3 + 0.2 * e - 0.4 * d
                 - 0.25 * np.maximum(study - 6, 0) - 0.20 * np.maximum(internet - 6, 0)
                 + rng.normal(0, 0.9, n), 4, 10)
    # Escala logística: muchos alumnos cerca del 100% pero sin una pila exacta en 100.
    attendance = _int(100 * _sigmoid(1.7 + 0.7 * e + 0.2 * a - 0.3 * d
                                     + rng.normal(0, 0.7, n)), 0, 100)
    task_prob = _sigmoid(0.6 + 1.0 * e - 0.3 * d + rng.normal(0, 0.5, n))
    tasks = rng.binomial(20, task_prob)
    previous = _int(66 + 8 * a + 1 * e + rng.normal(0, 9, n), 0, 100)

    exam = (63 + 0.35 * (previous - 66) + 1.6 * (study - 4) + 0.7 * (tasks - 12)
            + 0.10 * (attendance - 80) - 0.8 * (internet - 5) - 1.2 * np.abs(sleep - 7.5)
            + 3.0 * a + rng.normal(0, 10, n))
    exam = _int(exam, 0, 100)
    placed = rng.random(n) < _sigmoid((exam - 52) / 6)

    df = pd.DataFrame({
        "study_hours": study, "attendance": attendance, "sleep_hours": sleep,
        "internet_usage": internet, "assignments_completed": tasks,
        "previous_score": previous, "exam_score": exam,
        "placement_status": np.where(placed, "Placed", "Not Placed"),
    })
    df["perfil_latente"] = profile
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    df[COLUMNS].to_csv(args.salida, index=False)
    print(f"{len(df)} alumnos simulados guardados en {args.salida}")


if __name__ == "__main__":
    main()
