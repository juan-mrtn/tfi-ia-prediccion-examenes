"""Métricas de validación ex-post y predicción del próximo examen (ISSUE-303 y 304).

Reúne en un solo lugar las métricas de la red y de la predicción, y genera el CSV
de predicciones por alumno con el ρ elegido (D-13).

Uso (desde la carpeta raíz del proyecto):
    python -m src.metrics            dataset 1 (simulado)
    python -m src.metrics factors    dataset 2 (Student Performance Factors, D-18)

Resultados (con <nombre> = student o factors):
    data/output/predicciones_<nombre>.csv   una fila por alumno del 20% reservado
    data/output/metricas_<nombre>.csv       las métricas del modelo en una fila
"""

import sys


import numpy as np
import pandas as pd

from src.art1 import CarpenterGrossbergNetwork
from src.em_clustering import EMClusterer, academic_weights
from src.encoder import PositionalBinaryEncoder
from src.predictor import ExamPredictor, score_to_category
from src.preprocessor import DataPreprocessor

# Los dos datasets que pide la consigna (ítem 5). Cada uno trae su metadata con
# las 5 variables de entrada y la nota objetivo.
DATASETS = {
    "student": {
        "nombre": "Dataset 1: simulado (10.000 alumnos)",
        "corto": "1 · Simulado",
        "datos": "data/raw/student_performance.csv",
        "metadata": "data/metadata/student_metadata.csv",
    },
    "factors": {
        "nombre": "Dataset 2: Student Performance Factors (6.607 alumnos)",
        "corto": "2 · Student Performance Factors",
        "datos": "data/raw/StudentPerformanceFactors.csv",
        "metadata": "data/metadata/factors_metadata.csv",
    },
}

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


def entrenar_pipeline(rho=RHO, dataset="student", ruta_datos=None, ruta_metadata=None,
                      filas=None):
    """Corre el pipeline completo (EM + ART 1) y devuelve todo lo que se calculó.

    Lo usan main(), la CLI (CarGross.py) y la app visual (src/app_visual.py), así
    los tres muestran exactamente los mismos resultados. dataset es una clave de
    DATASETS; ruta_datos y ruta_metadata, si se pasan, reemplazan sus rutas.
    filas limita el dataset a sus primeras filas (lo usa la demo de la CLI).
    """
    # 1. Datos: chequeo de 24 hs y separación 80% / 20%.
    rutas = DATASETS[dataset]
    prep = DataPreprocessor(ruta_metadata or rutas["metadata"])
    datos = prep.load(ruta_datos or rutas["datos"])
    if filas is not None:
        datos = datos.head(filas)
    datos = prep.validate_hourly_consistency(datos)
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
    red = CarpenterGrossbergNetwork(X_train.shape[1], max_categories=3000, rho=rho)
    nodos_train = red.fit_predict(X_train)

    # 4. Cada nodo se calibra con la nota media (0–10) y la categoría EM de sus alumnos.
    # En el dataset 2 hay una nota de 101: se recorta a 100.
    objetivo = prep.target_col
    nota_train = entrenamiento[objetivo].clip(0, 100).values / 10
    nota_test = prueba[objetivo].clip(0, 100).values / 10
    predictor = ExamPredictor(red).calibrate(nodos_train, cat_train, nota_train)

    # 5. Predicción del 20% reservado, sin modificar los pesos de la red.
    pred = predictor.predict(X_test)
    cat_real = [score_to_category(n) for n in nota_test]

    # 6. Tabla de predicciones por alumno (la nota real va solo para comparar).
    predicciones = pd.DataFrame({
        "estudiante": np.arange(1, len(prueba) + 1),
        "nodo_art1": pred["nodo_art1"],
        "categoria_pronosticada": pred["categoria_pronosticada"],
        "nota_esperada": pred["nota_esperada"],
        "confianza_baja": pred["confianza_baja"],
        "nota_real": np.round(nota_test, 2),
        "categoria_real": cat_real,
    })

    # 7. Resumen de métricas.
    n = len(X_train)
    metricas = {
        "dataset": dataset,
        "rho": rho,
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
        "acierto_categoria": round(
            exactitud_categoria(cat_real, pred["categoria_pronosticada"]), 3),
        "confianza_baja": round(pred["confianza_baja"].mean(), 3),
    }

    return {
        "prep": prep,
        "em": em,
        "red": red,
        "predictor": predictor,
        "entrenamiento": entrenamiento,
        "nota_train": nota_train,
        "nodos_train": nodos_train,
        "cat_train": cat_train,
        "predicciones": predicciones,
        "metricas": metricas,
    }


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "student"
    if dataset not in DATASETS:
        print(f"Dataset desconocido: {dataset}. Opciones: {', '.join(DATASETS)}")
        sys.exit(2)
    salida_predicciones = f"data/output/predicciones_{dataset}.csv"
    salida_metricas = f"data/output/metricas_{dataset}.csv"

    resultado = entrenar_pipeline(dataset=dataset)
    resultado["predicciones"].to_csv(salida_predicciones, index=False)
    pd.DataFrame([resultado["metricas"]]).to_csv(salida_metricas, index=False)

    for nombre, valor in resultado["metricas"].items():
        print(f"{nombre:28s} {valor}")
    print(f"Predicciones guardadas en {salida_predicciones}")


if __name__ == "__main__":
    main()
