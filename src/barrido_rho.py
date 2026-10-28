"""Barrido del parámetro de vigilancia ρ de ART 1 (ISSUE-103).

Entrena la red con distintos valores de ρ y mide, para cada uno, cuántos grupos
crea, qué tan puros son respecto de las clases EM y qué tan bien predice la nota
del 20% de alumnos que la red no vio.

Uso (desde la carpeta raíz del proyecto):
    python -m src.barrido_rho

Resultado: data/output/barrido_rho_art1.csv
"""

import numpy as np
import pandas as pd

from src.art1 import CarpenterGrossbergNetwork
from src.em_clustering import EMClusterer, academic_weights
from src.metrics import binarizar, exactitud_categoria, mae, pureza, tasa_compresion, tasa_resets
from src.predictor import ExamPredictor, score_to_category
from src.preprocessor import DataPreprocessor

DATASET = "data/raw/student_performance.csv"
METADATA = "data/metadata/student_metadata.csv"
SALIDA = "data/output/barrido_rho_art1.csv"

# Valores de ρ a probar: de 0,10 a 0,90 con paso 0,05 (17 valores).
VALORES_RHO = np.round(np.arange(0.10, 0.91, 0.05), 2)

# Si la red necesita más nodos que esto, el ρ es demasiado alto y se descarta.
MAX_NODOS = 3000


def preparar_datos():
    """Lee el dataset, aplica el chequeo de 24 hs y lo separa en 80% / 20%."""
    prep = DataPreprocessor(METADATA)
    datos = prep.validate_hourly_consistency(prep.load(DATASET))
    entrenamiento, prueba = prep.train_test_split(datos)
    return prep, entrenamiento, prueba


def clases_em(prep, entrenamiento):
    """Agrupa a los alumnos de entrenamiento con EM (K = 4) y devuelve su categoría."""
    X = entrenamiento[prep.feature_cols].values.astype(float)
    em = EMClusterer(n_components=4).fit(X)
    categorias = em.compute_spectral_risk(academic_weights(prep.feature_cols))
    grupos, _ = em.predict(X)
    return np.array([categorias[g] for g in grupos])


def evaluar_rho(rho, X_train, X_test, cat_train, nota_train, nota_test):
    """Entrena ART 1 con un ρ y devuelve sus métricas, o None si crea demasiados nodos."""
    red = CarpenterGrossbergNetwork(X_train.shape[1], max_categories=MAX_NODOS, rho=rho)
    try:
        nodos = red.fit_predict(X_train)
    except RuntimeError:
        return None

    # Cada nodo aprende la nota media y la categoría EM de sus alumnos.
    predictor = ExamPredictor(red).calibrate(nodos, cat_train, nota_train)
    pred = predictor.predict(X_test)

    # La categoría real del alumno de prueba sale de su nota real.
    cat_real = np.array([score_to_category(n) for n in nota_test])

    n_train = len(X_train)
    return {
        "rho": rho,
        "clusters": red.num_active_clusters,
        "CR": round(tasa_compresion(red.num_active_clusters, n_train), 4),
        "RR": round(tasa_resets(red.total_resets, n_train), 2),
        "vacios_train": int((X_train.sum(axis=1) == 0).sum()),
        "pureza_EM": round(pureza(nodos, cat_train), 3),
        "MAE": round(mae(nota_test, pred["nota_esperada"]), 3),
        "acc_cat": round(exactitud_categoria(cat_real, pred["categoria_pronosticada"]), 3),
        "conf_baja": round(pred["confianza_baja"].mean(), 3),
    }


def main():
    prep, entrenamiento, prueba = preparar_datos()

    cat_train = clases_em(prep, entrenamiento)
    X_train = binarizar(prep, entrenamiento)
    X_test = binarizar(prep, prueba)

    # Las notas se pasan a escala 0–10. Solo se usan para calibrar y evaluar,
    # nunca como entrada de la red.
    nota_train = entrenamiento["exam_score"].values / 10
    nota_test = prueba["exam_score"].values / 10

    filas = []
    for rho in VALORES_RHO:
        resultado = evaluar_rho(rho, X_train, X_test, cat_train, nota_train, nota_test)
        if resultado is None:
            print(f"rho = {rho:.2f}: más de {MAX_NODOS} nodos, se descarta")
            continue
        print(f"rho = {rho:.2f}: {resultado['clusters']} nodos, "
              f"pureza {resultado['pureza_EM']:.1%}, MAE {resultado['MAE']:.2f}")
        filas.append(resultado)

    pd.DataFrame(filas).to_csv(SALIDA, index=False)
    print(f"Resultados guardados en {SALIDA}")


if __name__ == "__main__":
    main()
