"""Manual (man) de CarGross.py. Se muestra con --man y ante cualquier error."""

import sys

# Códigos de salida controlados (los usa CarGross.py y los documenta el manual).
SALIDA_OK = 0
SALIDA_ARGUMENTOS = 2      # opción mal escrita, faltante o con valor inválido
SALIDA_ARCHIVO = 3         # no se encontró o no se pudo leer/escribir un CSV
SALIDA_DATOS = 4           # el CSV o su metadata no cumplen las reglas
SALIDA_INESPERADA = 1      # cualquier otro error

MANUAL = """\
CARGROSS(1)            Manual de CarGross.py            CARGROSS(1)

NOMBRE
    CarGross.py - predictor híbrido EM + ART 1 (Carpenter-Grossberg 1)
    de la nota y la categoría del próximo examen.

SINOPSIS
    python CarGross.py --mode MODO [-d DATOS.csv -m METADATA.csv | --dataset 1|2]
                       [-i ALUMNOS.csv] [-o CARPETA] [--rho RHO]
    python CarGross.py --man

DESCRIPCIÓN
    Entrena EM (K = 4) sobre el 80% del dataset, codifica a cada alumno
    en 20 bits (10 bins, 4 bits por variable) y entrena la red ART 1.
    La nota esperada (0 a 10) es la nota media del nodo que resuena.
    Categorías: Promoción (> 8), Aprobado Regular (6 a 8),
    Recuperatorio (4 a 6), Riesgo Crítico (< 4).
    Toda la entrada y la salida es en archivos .csv.

MODOS
    train     Entrena y guarda los nodos de ART 1 en nodos_<nombre>.csv
              (alumnos, nota media, perfil EM y prototipo de bits).
    eval      Valida con el 20% reservado. Muestra las métricas y la
              matriz de confusión y guarda predicciones_<nombre>.csv,
              metricas_<nombre>.csv y matriz_confusion_<nombre>.csv.
    predict   Predice los alumnos nuevos del CSV indicado con -i (solo
              necesita las variables de entrada, no la nota). Guarda
              prediccion_<nombre de ALUMNOS>.csv.
    sweep     Prueba rho de 0,10 a 0,90 (paso 0,05). Guarda
              barrido_rho_<nombre>.csv. Tarda alrededor de un minuto.
    demo      Igual que eval con los primeros 2000 alumnos del dataset 1.
              Termina en pocos segundos. Guarda los archivos *_demo.csv.

    <nombre> es student o factors con --dataset, o el nombre del
    archivo de datos con -d.

OPCIONES
    --mode MODO          train, eval, predict, sweep o demo (obligatorio).
    -d, --datos CSV      Dataset de entrenamiento.
    -m, --metadata CSV   Reglas de las variables (rol, tipo, min, max,
                         escalar_a_100, binning). Va junto con -d.
    --dataset 1|2        Atajo: 1 = simulado (por defecto),
                         2 = Student Performance Factors.
    -i, --alumnos CSV    Alumnos a predecir (solo en modo predict).
    -o, --salida DIR     Carpeta de salida (por defecto data/output).
    --rho RHO            Vigilancia de ART 1, entre 0 y 1 (por defecto 0.65).
    --man, -h, --help    Muestra este manual.

EJEMPLOS
    python CarGross.py --mode demo
    python CarGross.py --mode eval
    python CarGross.py --mode eval --dataset 2
    python CarGross.py --mode eval -d data/raw/student_performance.csv \\
        -m data/metadata/student_metadata.csv
    python CarGross.py --mode predict -i alumnos_nuevos.csv

CÓDIGOS DE SALIDA
    0   Terminó bien.
    1   Error inesperado.
    2   Opciones inválidas o faltantes.
    3   No se encontró o no se pudo leer/escribir un archivo.
    4   Datos inválidos (faltan columnas, hay vacíos o valores no numéricos).

    Ante cualquier error se muestra el motivo y este manual.

VER TAMBIÉN
    python -m src.metrics, python -m src.barrido_rho,
    streamlit run src/app_visual.py, docs/Registro_de_decisiones.docx
"""


class ManPageViewer:
    """Muestra el manual, con un mensaje de error arriba si lo hay."""

    def __init__(self, texto=MANUAL):
        self.texto = texto

    def mostrar(self, error=None, salida=None):
        """Escribe el error (si hay) y el manual. Por defecto en stderr si hay error."""
        if salida is None:
            salida = sys.stderr if error else sys.stdout
        if error:
            salida.write(f"ERROR: {error}\n\n")
        salida.write(self.texto)
