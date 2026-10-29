"""
CarGross.py - Sistema híbrido EM (GMM) + Carpenter-Grossberg 1 (ART 1).

Predice la nota esperada y la categoría del próximo examen de un estudiante
a partir de sus hábitos diarios y antecedentes académicos.
Punto de entrada único (CLI). Ver el manual con: python CarGross.py --man
"""
import argparse
import os
import sys

from src.man_pages import (SALIDA_ARCHIVO, SALIDA_ARGUMENTOS, SALIDA_DATOS,
                           SALIDA_INESPERADA, SALIDA_OK, ManPageViewer)

MODOS = ["train", "eval", "predict", "sweep", "demo"]

# El atajo --dataset usa las claves de DATASETS de src/metrics.py.
DATASET_POR_NUMERO = {"1": "student", "2": "factors"}

# La demo usa las primeras filas del dataset 1 para terminar en pocos segundos.
FILAS_DEMO = 2000

CATEGORIAS = ["Promoción", "Aprobado Regular", "Recuperatorio", "Desaprobado"]


class ErrorArgumentos(Exception):
    """Opción de la línea de comandos mal escrita, faltante o inválida."""


class LectorArgumentos(argparse.ArgumentParser):
    """argparse sale solo y muestra su ayuda; acá se lanza el error para mostrar el man."""

    def error(self, message):
        raise ErrorArgumentos(message)


def crear_lector():
    lector = LectorArgumentos(prog="CarGross.py", add_help=False)
    lector.add_argument("--mode", choices=MODOS)
    lector.add_argument("-d", "--datos")
    lector.add_argument("-m", "--metadata")
    lector.add_argument("--dataset", choices=list(DATASET_POR_NUMERO), default="1")
    lector.add_argument("-i", "--alumnos")
    lector.add_argument("-o", "--salida", default=os.path.join("data", "output"))
    lector.add_argument("--rho", type=float, default=0.65)
    lector.add_argument("--man", "-h", "--help", action="store_true")
    return lector


def validar_argumentos(args):
    """Chequeos que argparse no hace solo."""
    if args.mode is None:
        raise ErrorArgumentos("falta --mode (" + ", ".join(MODOS) + ").")
    if (args.datos is None) != (args.metadata is None):
        raise ErrorArgumentos("-d y -m van juntos: el dataset y su metadata.")
    if not 0 < args.rho <= 1:
        raise ErrorArgumentos("--rho debe estar entre 0 y 1.")
    if args.mode == "predict" and args.alumnos is None:
        raise ErrorArgumentos("el modo predict necesita -i con el CSV de alumnos.")
    if args.mode != "predict" and args.alumnos is not None:
        raise ErrorArgumentos("-i solo se usa en el modo predict.")
    for ruta in (args.datos, args.metadata, args.alumnos):
        if ruta is None:
            continue
        if not ruta.lower().endswith(".csv"):
            raise ErrorArgumentos(f"{ruta} no es un archivo .csv.")
        if not os.path.isfile(ruta):
            raise FileNotFoundError(f"No existe el archivo {ruta}")


def nombre_salida(args, dataset):
    """student o factors con --dataset; con -d, el nombre del archivo de datos."""
    if args.datos is None:
        return dataset
    return os.path.splitext(os.path.basename(args.datos))[0]


def guardar(tabla, carpeta, archivo):
    """Escribe una tabla en CSV y avisa dónde quedó."""
    ruta = os.path.join(carpeta, archivo)
    tabla.to_csv(ruta, index=False)
    print(f"Guardado: {ruta}")


def matriz_confusion(predicciones):
    """Tabla categoría real (filas) contra categoría pronosticada (columnas)."""
    import pandas as pd

    matriz = pd.crosstab(predicciones["categoria_real"],
                         predicciones["categoria_pronosticada"])
    matriz = matriz.reindex(index=CATEGORIAS, columns=CATEGORIAS, fill_value=0)
    matriz.index.name = "real \\ pronosticada"
    return matriz.reset_index()


def mostrar_metricas(metricas):
    for clave, valor in metricas.items():
        print(f"{clave:28s} {valor}")


# ---------------------------------------------------------------------------
# Modos
# ---------------------------------------------------------------------------

def modo_train(resultado, carpeta, nombre):
    """Guarda los nodos que aprendió ART 1: tamaño, nota media, perfil EM y prototipo."""
    import pandas as pd

    predictor = resultado["predictor"]
    red = resultado["red"]
    nodos = sorted(predictor.node_size)
    tabla = pd.DataFrame({
        "nodo": nodos,
        "alumnos": [predictor.node_size[n] for n in nodos],
        "nota_media": [round(predictor.node_expected_score[n], 2) for n in nodos],
        "perfil_em": [predictor.node_category[n] for n in nodos],
        "prototipo": ["".join(map(str, red.t[n])) for n in nodos],
    })
    print(f"ART 1 creó {red.num_active_clusters} nodos con rho = {red.rho}.")
    guardar(tabla, carpeta, f"nodos_{nombre}.csv")


def modo_eval(resultado, carpeta, nombre):
    """Valida con el 20% reservado: predicciones, métricas y matriz de confusión."""
    import pandas as pd

    metricas = dict(resultado["metricas"], dataset=nombre)
    matriz = matriz_confusion(resultado["predicciones"])
    mostrar_metricas(metricas)
    print()
    print(matriz.to_string(index=False))
    print()
    guardar(resultado["predicciones"], carpeta, f"predicciones_{nombre}.csv")
    guardar(pd.DataFrame([metricas]), carpeta, f"metricas_{nombre}.csv")
    guardar(matriz, carpeta, f"matriz_confusion_{nombre}.csv")


def leer_alumnos(prep, ruta):
    """Lee los alumnos nuevos y verifica que tengan las variables de entrada."""
    import pandas as pd

    alumnos = pd.read_csv(ruta)
    faltan = [c for c in prep.feature_cols if c not in alumnos.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en {ruta}: {faltan}")
    entradas = alumnos[prep.feature_cols]
    if entradas.isnull().any().any():
        raise ValueError(f"{ruta} tiene valores vacíos.")
    no_numericas = [c for c in prep.feature_cols
                    if not pd.api.types.is_numeric_dtype(entradas[c])]
    if no_numericas:
        raise ValueError(f"Columnas no numéricas en {ruta}: {no_numericas}")
    return alumnos.reset_index(drop=True)


def modo_predict(resultado, ruta_alumnos, carpeta):
    """Predice nota y categoría de cada alumno nuevo (la red no aprende)."""
    from src.metrics import binarizar

    prep = resultado["prep"]
    alumnos = leer_alumnos(prep, ruta_alumnos)
    pred = resultado["predictor"].predict(binarizar(prep, alumnos))
    pred.insert(0, "estudiante", range(1, len(alumnos) + 1))

    print(pred[["estudiante", "nota_esperada", "categoria_pronosticada"]]
          .to_string(index=False))
    nombre = os.path.splitext(os.path.basename(ruta_alumnos))[0]
    guardar(pred, carpeta, f"prediccion_{nombre}.csv")


def modo_sweep(args, dataset, nombre):
    """Barrido de rho de 0,10 a 0,90 (el mismo de python -m src.barrido_rho)."""
    from src.barrido_rho import barrer_rho, preparar_datos

    prep, entrenamiento, prueba = preparar_datos(dataset, args.datos, args.metadata)
    guardar(barrer_rho(prep, entrenamiento, prueba), args.salida, f"barrido_rho_{nombre}.csv")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def ejecutar(argv):
    """Corre la CLI. Puede lanzar excepciones: main() las atrapa."""
    args = crear_lector().parse_args(argv)
    if args.man:
        ManPageViewer().mostrar()
        return SALIDA_OK
    validar_argumentos(args)
    os.makedirs(args.salida, exist_ok=True)
    dataset = DATASET_POR_NUMERO[args.dataset]
    nombre = nombre_salida(args, dataset)

    if args.mode == "sweep":
        modo_sweep(args, dataset, nombre)
        return SALIDA_OK

    # Se importa acá para que --man y los errores de opciones respondan al instante.
    from src.metrics import entrenar_pipeline

    filas = None
    if args.mode == "demo":
        dataset, nombre, filas = "student", "demo", FILAS_DEMO
        print(f"Demo: primeros {FILAS_DEMO} alumnos del dataset 1.")
    print(f"Entrenando EM + ART 1 con rho = {args.rho}...")
    resultado = entrenar_pipeline(args.rho, dataset, args.datos, args.metadata, filas)

    if args.mode == "train":
        modo_train(resultado, args.salida, nombre)
    elif args.mode == "predict":
        modo_predict(resultado, args.alumnos, args.salida)
    else:
        modo_eval(resultado, args.salida, nombre)
    return SALIDA_OK


def main(argv=None) -> int:
    """Punto de entrada: todo error muestra el manual y sale con un código controlado."""
    if argv is None:
        argv = sys.argv[1:]
    try:
        return ejecutar(argv)
    except ErrorArgumentos as error:
        codigo, mensaje = SALIDA_ARGUMENTOS, error
    except OSError as error:
        codigo, mensaje = SALIDA_ARCHIVO, error
    except (ValueError, KeyError) as error:
        codigo, mensaje = SALIDA_DATOS, f"datos inválidos: {error}"
    except KeyboardInterrupt:
        codigo, mensaje = SALIDA_INESPERADA, "interrumpido por el usuario."
    except Exception as error:  # noqa: BLE001 (la cátedra pide control total)
        codigo, mensaje = SALIDA_INESPERADA, f"{type(error).__name__}: {error}"
    ManPageViewer().mostrar(error=mensaje)
    return codigo


if __name__ == "__main__":
    sys.exit(main())
