"""Valida el dataset simulado: rangos, correlaciones, histograma de notas y EM/BIC.

Uso:
    python scripts/validar_dataset_simulado.py [--seed 2026]
Escribe data/output/simulado_validacion_bic.csv con la tabla de K.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generar_dataset_simulado import generate  # noqa: E402
from src.em_clustering import EMClusterer  # noqa: E402

FEATURES = ["study_hours", "attendance", "internet_usage", "assignments_completed",
            "previous_score"]
CATEGORY_BINS = [-1, 40, 60, 80, 100]
CATEGORY_NAMES = ["Riesgo (<4)", "Recuperatorio (4-6)", "Regular (6-8)", "Promoción (>8)"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    df = generate(seed=args.seed)

    print("== Rangos ==")
    print(df.drop(columns=["placement_status", "perfil_latente"]).describe()
          .T[["mean", "std", "min", "max"]].round(1))
    total = df["sleep_hours"] + df[["study_hours", "internet_usage"]].max(axis=1)
    print(f"Filas que violan la regla de 24 hs: {(total > 24).sum()}")

    print("\n== Correlación con exam_score ==")
    num = df.drop(columns=["placement_status", "perfil_latente"])
    print(num.corr()["exam_score"].drop("exam_score").round(2).to_string())
    print("\n== Correlaciones entre entradas ==")
    print(num[FEATURES + ["sleep_hours"]].corr().round(2))

    print("\n== Histograma de exam_score (10 bins) ==")
    hist = pd.cut(df["exam_score"], bins=range(0, 101, 10), include_lowest=True)
    print(hist.value_counts(sort=False, normalize=True).mul(100).round(1).to_string())
    print(f"exam_score == 100: {(df['exam_score'] == 100).mean():.1%}")
    cats = pd.cut(df["exam_score"], CATEGORY_BINS, labels=CATEGORY_NAMES)
    print(cats.value_counts(normalize=True).mul(100).round(1).to_string())
    print(df["placement_status"].value_counts(normalize=True).round(3).to_string())

    print("\n== EM sobre las 5 entradas crudas (80% train) ==")
    idx = np.random.default_rng(42).permutation(len(df))
    train = df.iloc[idx[: int(len(df) * 0.8)]]
    X = train[FEATURES].to_numpy(dtype=float)
    em = EMClusterer(random_state=42)
    em.find_optimal_k(X, range(2, 9))
    print(em.selection_.round(0).to_string(index=False))
    em.selection_.to_csv(ROOT / "data" / "output" / "simulado_validacion_bic.csv", index=False)

    em4 = EMClusterer(n_components=4, random_state=42).fit(X)
    labels, _ = em4.predict(X)
    print(f"Métricas K=4: {({k: round(v, 3) for k, v in em4.metrics(X).items()})}")
    print(f"ARI clusters vs perfil latente: "
          f"{adjusted_rand_score(train['perfil_latente'], labels):.2f}")
    y = train["exam_score"].to_numpy(dtype=float)
    means = pd.Series(y).groupby(labels).transform("mean").to_numpy()
    print(f"Varianza de la nota explicada por los clusters: "
          f"{1 - ((y - means) ** 2).sum() / ((y - y.mean()) ** 2).sum():.0%}")
    print(pd.crosstab(labels, train["perfil_latente"]))
    print(train.groupby(labels)["exam_score"].agg(["mean", "std", "size"]).round(1))


if __name__ == "__main__":
    main()
