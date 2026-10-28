"""Métricas de validación ex-post y predicción del próximo examen (ISSUE-303 y 304).

Reúne en un solo lugar las métricas de la red y de la predicción, y genera el CSV
de predicciones por alumno con el ρ elegido (D-13).

Uso (desde la carpeta raíz del proyecto):
    python -m src.metrics

Resultados:
    data/output/predicciones_student.csv   una fila por alumno del 20% reservado
    data/output/metricas_student.csv       las métricas del modelo en una fila
"""

import numpy as np
import pandas as pd

from src.art1 import CarpenterGrossbergNetwork
from src.em_clustering import EMClusterer, academic_weights
from src.encoder import PositionalBinaryEncoder
from src.predictor import ExamPredictor, score_to_category
from src.preprocessor import DataPreprocessor

DATASET = "data/raw/student_performance.csv"
METADATA = "data/metadata/student_metadata.csv"
SALIDA_PREDICCIONES = "data/output/predicciones_student.csv"
SALIDA_METRICAS = "data/output/metricas_student.csv"

RHO = 0.65  # D-13
K = 4       # D-09


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def pureza(nodos, categorias):
    """Proporción de alumnos que comparten la categoría EM mayoritaria de su nodo.

    Pureza = (1/N) · Σ_j máx_c |C_j ∩ L_c|. Los alumnos sin nodo (-1) no cuentan.
    """
    tabla = pd.DataFrame({"nodo": nodos, "categoria": categorias})
    tabla = tabla[tabla["nodo"] >= 0]
    mayoritarios = tabla.groupby("nodo")["categoria"].agg(lambda c: c.value_counts().iloc[0])
    return mayoritarios.sum() / len(tabla)


def tasa_compresion(cantidad_nodos, cantidad_alumnos):
    """CR = nodos / alumnos. El documento de arquitectura pide 0,05 ≤ CR ≤ 0,20."""
    return cantidad_nodos / cantidad_alumnos


def tasa_resets(total_resets, cantidad_alumnos):
    """RR = resets / alumnos: cuántas veces por alumno falló la prueba de vigilancia."""
    return total_resets / cantidad_alumnos


def mae(nota_real, nota_esperada):
    """Error absoluto medio: promedio de |nota real − nota esperada|, en escala 0–10."""
    return np.mean(np.abs(np.asarray(nota_real) - np.asarray(nota_esperada)))


def exactitud_categoria(categoria_real, categoria_pronosticada):
    """Proporción de alumnos cuya categoría pronosticada coincide con la real."""
    return np.mean(np.asarray(categoria_real) == np.asarray(categoria_pronosticada))


def varianza_por_nodo(nodos, notas):
    """Varianza media de la nota dentro de cada nodo (menor es mejor)."""
    tabla = pd.DataFrame({"nodo": nodos, "nota": notas})
    tabla = tabla[tabla["nodo"] >= 0]
    return float(tabla.groupby("nodo")["nota"].var(ddof=0).mean())


# ---------------------------------------------------------------------------
# Pipeline completo con el ρ elegido
# ---------------------------------------------------------------------------

def binarizar(prep, tabla):
    """Escala las tareas a 100 y convierte cada alumno en un vector de 20 bits."""
    escalada = prep.scale_to_base_100_int(tabla)
    encoder = PositionalBinaryEncoder(prep.metadata_df["binning"].to_dict())
    X, _ = encoder.fit_transform(escalada, prep.feature_cols)
    return X


def main():
    # 1. Datos: chequeo de 24 hs y separación 80% / 20%.
    prep = DataPreprocessor(METADATA)
    datos = prep.validate_hourly_consistency(prep.load(DATASET))
    entrenamiento, prueba = prep.train_test_split(datos)

    # 2. EM sobre las variables sin adaptar: da la categoría de cada alumno.
    X_cont = entrenamiento[prep.feature_cols].values.astype(float)
    em = EMClusterer(n_components=K).fit(X_cont)
    categorias_em = em.compute_spectral_risk(academic_weights(prep.feature_cols))
    grupos, _ = em.predict(X_cont)
    cat_train = np.array([categorias_em[g] for g in grupos])
    metricas_em = em.metrics(X_cont)

    # 3. ART 1 aprende los patrones binarios del 80% de entrenamiento.
    X_train = binarizar(prep, entrenamiento)
    X_test = binarizar(prep, prueba)
    red = CarpenterGrossbergNetwork(X_train.shape[1], max_categories=3000, rho=RHO)
    nodos_train = red.fit_predict(X_train)

    # 4. Cada nodo se calibra con la nota media (0–10) y la categoría EM de sus alumnos.
    nota_train = entrenamiento["exam_score"].values / 10
    nota_test = prueba["exam_score"].values / 10
    predictor = ExamPredictor(red).calibrate(nodos_train, cat_train, nota_train)

    # 5. Predicción del 20% reservado, sin modificar los pesos de la red.
    pred = predictor.predict(X_test)
    cat_real = [score_to_category(n) for n in nota_test]

    # 6. CSV de predicciones por alumno (la nota real va solo para comparar).
    predicciones = pd.DataFrame({
        "estudiante": np.arange(1, len(prueba) + 1),
        "nodo_art1": pred["nodo_art1"],
        "categoria_pronosticada": pred["categoria_pronosticada"],
        "nota_esperada": pred["nota_esperada"],
        "confianza_baja": pred["confianza_baja"],
        "nota_real": np.round(nota_test, 2),
        "categoria_real": cat_real,
    })
    predicciones.to_csv(SALIDA_PREDICCIONES, index=False)

    # 7. Resumen de métricas.
    n = len(X_train)
    metricas = {
        "rho": RHO,
        "K": K,
        "BIC": round(metricas_em["BIC"], 1),
        "AIC": round(metricas_em["AIC"], 1),
        "silhouette": round(metricas_em["silhouette"], 3),
        "entropia_responsabilidades": round(metricas_em["entropia_responsabilidades"], 3),
        "nodos": red.num_active_clusters,
        "CR": round(tasa_compresion(red.num_active_clusters, n), 4),
        "RR": round(tasa_resets(red.total_resets, n), 2),
        "pureza_EM": round(pureza(nodos_train, cat_train), 3),
        "varianza_nota_por_nodo": round(varianza_por_nodo(nodos_train, nota_train), 3),
        "MAE": round(mae(nota_test, pred["nota_esperada"]), 3),
        "acierto_categoria": round(exactitud_categoria(cat_real, pred["categoria_pronosticada"]), 3),
        "confianza_baja": round(pred["confianza_baja"].mean(), 3),
    }
    pd.DataFrame([metricas]).to_csv(SALIDA_METRICAS, index=False)

    for nombre, valor in metricas.items():
        print(f"{nombre:28s} {valor}")
    print(f"Predicciones guardadas en {SALIDA_PREDICCIONES}")


if __name__ == "__main__":
    main()
