"""Tests de la CLI (CarGross.py) y del manual (src/man_pages.py)."""

import io

import pandas as pd
import pytest

from CarGross import main
from src.man_pages import (SALIDA_ARCHIVO, SALIDA_ARGUMENTOS, SALIDA_DATOS, SALIDA_OK,
                           ManPageViewer)

DATOS = "data/raw/student_performance.csv"
METADATA = "data/metadata/student_metadata.csv"


def test_man_se_muestra_con_la_opcion(capsys):
    assert main(["--man"]) == SALIDA_OK
    assert "CÓDIGOS DE SALIDA" in capsys.readouterr().out


def test_man_con_error_lo_escribe_arriba():
    salida = io.StringIO()
    ManPageViewer().mostrar(error="algo falló", salida=salida)
    assert salida.getvalue().startswith("ERROR: algo falló")


@pytest.mark.parametrize("argv", [
    [],                                         # falta --mode
    ["--mode", "otro"],                         # modo inexistente
    ["--mode", "eval", "--rho", "1.5"],         # rho fuera de rango
    ["--mode", "eval", "--rho", "alto"],        # rho no numérico
    ["--mode", "eval", "-d", DATOS],            # -d sin -m
    ["--mode", "eval", "-i", DATOS],            # -i fuera de predict
    ["--mode", "eval", "-d", "x.txt", "-m", "y.txt"],  # no son .csv
    ["--mode", "predict"],                      # predict sin -i
    ["--mode", "eval", "--dataset", "3"],       # dataset inexistente
    ["--opcion-rara"],
])
def test_opciones_invalidas_muestran_man(argv, capsys):
    assert main(argv) == SALIDA_ARGUMENTOS
    error = capsys.readouterr().err
    assert error.startswith("ERROR:")
    assert "SINOPSIS" in error


def test_archivo_inexistente(capsys):
    codigo = main(["--mode", "eval", "-d", "no_existe.csv", "-m", METADATA])
    assert codigo == SALIDA_ARCHIVO
    assert "no_existe.csv" in capsys.readouterr().err


def test_dataset_sin_columnas(tmp_path, capsys):
    malo = tmp_path / "malo.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(malo, index=False)
    codigo = main(["--mode", "eval", "-d", str(malo), "-m", METADATA, "-o", str(tmp_path)])
    assert codigo == SALIDA_DATOS
    assert "Faltan columnas" in capsys.readouterr().err


def test_eval_y_predict_escriben_csv(tmp_path):
    # Un dataset chico (400 alumnos) para que el test sea rápido.
    chico = tmp_path / "chico.csv"
    pd.read_csv(DATOS).head(400).to_csv(chico, index=False)
    base = ["-d", str(chico), "-m", METADATA, "-o", str(tmp_path)]

    assert main(["--mode", "eval"] + base) == SALIDA_OK
    metricas = pd.read_csv(tmp_path / "metricas_chico.csv")
    assert metricas.loc[0, "dataset"] == "chico"
    assert len(pd.read_csv(tmp_path / "predicciones_chico.csv")) == 80

    # Alumnos nuevos: solo las variables de entrada, sin la nota.
    nuevos = tmp_path / "nuevos.csv"
    pd.DataFrame({"study_hours": [2, 9], "attendance": [40, 95], "internet_usage": [8, 1],
                  "assignments_completed": [5, 19], "previous_score": [35, 90]}
                 ).to_csv(nuevos, index=False)
    assert main(["--mode", "predict", "-i", str(nuevos)] + base) == SALIDA_OK
    pred = pd.read_csv(tmp_path / "prediccion_nuevos.csv")
    assert list(pred["estudiante"]) == [1, 2]
    assert pred["nota_esperada"].between(0, 10).all()


def test_predict_sin_variables_de_entrada(tmp_path, capsys):
    chico = tmp_path / "chico.csv"
    pd.read_csv(DATOS).head(200).to_csv(chico, index=False)
    nuevos = tmp_path / "nuevos.csv"
    pd.DataFrame({"study_hours": [3]}).to_csv(nuevos, index=False)
    codigo = main(["--mode", "predict", "-i", str(nuevos), "-d", str(chico), "-m", METADATA,
                   "-o", str(tmp_path)])
    assert codigo == SALIDA_DATOS
    assert "Faltan columnas" in capsys.readouterr().err


def test_train_guarda_los_nodos(tmp_path):
    chico = tmp_path / "chico.csv"
    pd.read_csv(DATOS).head(300).to_csv(chico, index=False)
    assert main(["--mode", "train", "-d", str(chico), "-m", METADATA,
                 "-o", str(tmp_path)]) == SALIDA_OK
    nodos = pd.read_csv(tmp_path / "nodos_chico.csv", dtype={"prototipo": str})
    assert nodos["alumnos"].sum() == 240        # el 80% de 300
    assert nodos["prototipo"].str.len().eq(20).all()


def test_matriz_de_confusion_en_eval(tmp_path):
    chico = tmp_path / "chico.csv"
    pd.read_csv(DATOS).head(300).to_csv(chico, index=False)
    main(["--mode", "eval", "-d", str(chico), "-m", METADATA, "-o", str(tmp_path)])
    matriz = pd.read_csv(tmp_path / "matriz_confusion_chico.csv")
    assert matriz.iloc[:, 1:].to_numpy().sum() == 60    # el 20% de 300
