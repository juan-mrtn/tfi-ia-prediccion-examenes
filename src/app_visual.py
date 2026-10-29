"""App visual en el navegador del predictor EM + ART 1 (complemento de CarGross.py).

Usa los mismos módulos de src/ que la CLI, con ρ = 0,65 (D-13). No reemplaza a
CarGross.py: es una forma más visual de mostrar el modelo. Arriba hay un control
para pasar del dataset 1 (simulado) al dataset 2 (Student Performance Factors, D-18).

Uso (desde la carpeta raíz del proyecto):
    streamlit run src/app_visual.py

Se abre sola en el navegador (http://localhost:8501). Para cortarla: Ctrl + C.
"""

import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

# La app se ejecuta como script: se agrega la carpeta raíz para poder importar src.*
# y se trabaja desde ahí para que las rutas data/... funcionen.
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)

from src.em_clustering import CATEGORIES, EMClusterer  # noqa: E402
from src.encoder import PositionalBinaryEncoder  # noqa: E402
from src.barrido_rho import SALIDAS as BARRIDOS_RHO  # noqa: E402
from src.metrics import DATASETS, RHO, binarizar, entrenar_pipeline  # noqa: E402

# Las 4 categorías van de mejor a peor nota. Un solo color (azul), más oscuro = mejor nota.
COLORES_CATEGORIA = ["#184f95", "#256abf", "#3987e5", "#86b6ef"]
AZUL = "#2a78d6"
NARANJA = "#eb6834"


def variables_de(modelo):
    """Texto, rango y valor inicial de cada variable de entrada, sacados del metadata.

    El valor inicial es la mediana del 80% de entrenamiento.
    """
    prep = modelo["prep"]
    variables = {}
    for col in prep.feature_cols:
        fila = prep.metadata_df.loc[col]
        minimo, maximo = int(fila["min"]), int(fila["max"])
        inicial = int(modelo["entrenamiento"][col].median())
        variables[col] = (f"{fila['descripcion']} ({minimo} a {maximo})", minimo, maximo, inicial)
    return variables


def horas_de_control(prep):
    """Variables de horas que no entran a la red y solo sirven para el chequeo de 24 hs."""
    meta = prep.metadata_df
    return list(meta.index[(meta["tipo"] == "horas") & (meta["rol"] == "control")])


# ---------------------------------------------------------------------------
# Modelo (se entrena una sola vez y queda guardado mientras la app esté abierta)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Entrenando EM + ART 1 con ρ = 0,65 (unos segundos)...")
def cargar_modelo(dataset):
    return entrenar_pipeline(RHO, dataset)


@st.cache_data(show_spinner="Probando EM con K de 2 a 7...")
def tabla_bic(_modelo, dataset):
    """BIC y AIC para cada K (como en D-08), sobre el 80% de entrenamiento."""
    prep = _modelo["prep"]
    X = _modelo["entrenamiento"][prep.feature_cols].values.astype(float)
    em = EMClusterer()
    em.find_optimal_k(X)
    return em.selection_


def predecir_alumnos(modelo, alumnos):
    """Predice nota y categoría para una tabla de alumnos con las 5 variables de entrada."""
    alumnos = alumnos.reset_index(drop=True)
    prep = modelo["prep"]
    X_bin = binarizar(prep, alumnos)
    pred = modelo["predictor"].predict(X_bin)

    # Perfil EM de cada alumno (sobre los valores sin adaptar).
    em = modelo["em"]
    grupos, probabilidades = em.predict(alumnos[prep.feature_cols].values.astype(float))
    pred["perfil_em"] = [em.risk_categories_[g] for g in grupos]
    return pred, X_bin, probabilidades


def bits_del_alumno(modelo, alumno):
    """Tabla con el valor, el bin y los 4 bits de cada variable (para explicar el vector)."""
    prep = modelo["prep"]
    escalado = prep.scale_to_base_100_int(alumno)
    encoder = PositionalBinaryEncoder(prep.metadata_df["binning"].to_dict())
    bins = encoder.to_bins(escalado, prep.feature_cols)
    filas = []
    for col in prep.feature_cols:
        b = int(bins[col].iloc[0])
        filas.append({"variable": col,
                      "valor original": int(alumno[col].iloc[0]),
                      "valor usado": int(escalado[col].iloc[0]),
                      "bin (0 a 10)": b,
                      "bits": "".join(str(x) for x in encoder.int_to_4bit_positional(b))})
    return pd.DataFrame(filas)


def escala_categorias():
    return alt.Scale(domain=CATEGORIES, range=COLORES_CATEGORIA)


# ---------------------------------------------------------------------------
# Pestaña 1: un alumno
# ---------------------------------------------------------------------------

def pestana_alumno(modelo):
    st.subheader("Predicción para un alumno")
    st.caption("Mové los controles y la predicción se actualiza sola. "
               "La nota del examen nunca entra a la red: solo se usa para calibrar los nodos.")

    prep = modelo["prep"]
    dataset = modelo["metricas"]["dataset"]
    col_entrada, col_resultado = st.columns([1, 2], gap="large")

    with col_entrada:
        valores = {}
        # La key incluye el dataset para que cada uno recuerde sus propios valores.
        for col, (texto, minimo, maximo, inicial) in variables_de(modelo).items():
            valores[col] = st.slider(texto, minimo, maximo, inicial, key=f"{dataset}_{col}")
        control = {}
        for col in horas_de_control(prep):
            control[col] = st.slider(
                "Horas de sueño (solo para el chequeo de 24 hs)", 0, 24, 7, key=f"{dataset}_{col}",
                help="No entra a la red (D-04): no cambia la nota. Solo se usa para "
                     "detectar datos imposibles: sueño + máx(estudio, internet) > 24.")

    alumno = pd.DataFrame([valores])
    pred, _, probabilidades = predecir_alumnos(modelo, alumno)
    fila = pred.iloc[0]

    with col_resultado:
        # Chequeo de 24 hs (D-05): no frena la predicción, solo avisa.
        if prep.validate_hourly_consistency(pd.DataFrame([{**valores, **control}])).empty:
            st.warning("Estos datos no pasan el chequeo de 24 hs "
                       "(las horas del día suman más de 24). "
                       "En el dataset este alumno se descartaría.")

        c1, c2, c3 = st.columns([2, 3, 1.5])
        c1.metric("Nota esperada", f"{fila['nota_esperada']:.2f} / 10")
        c2.metric("Categoría", fila["categoria_pronosticada"])
        c3.metric("Nodo ART 1", int(fila["nodo_art1"]))
        st.caption("La nota esperada es el promedio de los alumnos del nodo y la categoría sale "
                   "de esa nota: Promoción > 8, Aprobado Regular 6 a 8, Recuperatorio 4 a 6, "
                   f"Riesgo Crítico < 4. Perfil EM más común en el nodo: {fila['perfil_em_nodo']}.")
        if fila["confianza_baja"]:
            st.info("Confianza baja: ningún nodo superó la vigilancia ρ. "
                    "Se usó el nodo más parecido.")

        # Vector de 20 bits que ve la red.
        st.markdown("**Lo que ve la red: 5 variables × 4 bits = 20 bits**")
        tabla_bits = bits_del_alumno(modelo, alumno)
        st.dataframe(tabla_bits, hide_index=True, width="stretch")
        st.code(" ".join(tabla_bits["bits"]), language=None)

    st.divider()
    izquierda, derecha = st.columns(2, gap="large")

    # Probabilidad de pertenecer a cada perfil EM.
    with izquierda:
        st.markdown("**Perfil EM: probabilidad de cada grupo**")
        em = modelo["em"]
        datos_em = pd.DataFrame({
            "categoria": [em.risk_categories_[k] for k in range(em.n_components)],
            "probabilidad": probabilidades[0],
        })
        grafico = alt.Chart(datos_em).mark_bar(cornerRadiusEnd=4, height=22).encode(
            x=alt.X("probabilidad:Q", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", title=None)),
            y=alt.Y("categoria:N", sort=CATEGORIES, title=None, axis=alt.Axis(labelLimit=220)),
            color=alt.Color("categoria:N", scale=escala_categorias(), legend=None),
            tooltip=[alt.Tooltip("categoria:N", title="Perfil"),
                     alt.Tooltip("probabilidad:Q", title="Probabilidad", format=".1%")],
        ).properties(height=170)
        st.altair_chart(grafico, width="stretch")
        st.caption(f"EM lo ubica en **{fila['perfil_em']}**. "
                   "EM da probabilidades, no una sola respuesta.")

    # Notas reales de los alumnos de entrenamiento que cayeron en el mismo nodo.
    with derecha:
        nodo = int(fila["nodo_art1"])
        notas = modelo["nota_train"]
        del_nodo = pd.DataFrame({"nota": notas[modelo["nodos_train"] == nodo]})
        st.markdown(f"**Alumnos del nodo {nodo}: {len(del_nodo)} de entrenamiento**")
        if len(del_nodo):
            barras = alt.Chart(del_nodo).mark_bar(color=AZUL, cornerRadiusEnd=4).encode(
                x=alt.X("nota:Q", bin=alt.Bin(step=0.5, extent=[0, 10]),
                        title="Nota real del examen (0 a 10)"),
                y=alt.Y("count():Q", title="Alumnos"),
                tooltip=[alt.Tooltip("nota:Q", bin=alt.Bin(step=0.5, extent=[0, 10]), title="Nota"),
                         alt.Tooltip("count():Q", title="Alumnos")],
            )
            media = alt.Chart(pd.DataFrame({"nota": [fila["nota_esperada"]]})).mark_rule(
                color=NARANJA, strokeWidth=2).encode(
                x="nota:Q", tooltip=[alt.Tooltip("nota:Q", title="Nota esperada")])
            st.altair_chart((barras + media).properties(height=170), width="stretch")
            st.caption("La línea naranja es la nota esperada: el promedio de estos alumnos.")


# ---------------------------------------------------------------------------
# Pestaña 2: métricas del modelo
# ---------------------------------------------------------------------------

def pestana_metricas(modelo):
    m = modelo["metricas"]

    st.subheader("EM (mezcla de gaussianas)")
    c = st.columns(5)
    c[0].metric("K (grupos)", m["K"])
    c[1].metric("BIC", f"{m['BIC']:.0f}")
    c[2].metric("AIC", f"{m['AIC']:.0f}")
    c[3].metric("Silhouette", f"{m['silhouette']:.2f}",
                help="De −1 a 1: qué tan separados están los grupos. Más alto es mejor.")
    c[4].metric("Entropía", f"{m['entropia_responsabilidades']:.2f}",
                help="Qué tan dudoso está EM al asignar a cada alumno. Más bajo es mejor.")

    seleccion = tabla_bic(modelo, m["dataset"])
    largo = seleccion.melt(id_vars="K", value_vars=["BIC", "AIC"],
                           var_name="criterio", value_name="valor")
    lineas = alt.Chart(largo).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=70)).encode(
        x=alt.X("K:O", title="Cantidad de grupos (K)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("valor:Q", scale=alt.Scale(zero=False), title="Valor (más bajo es mejor)"),
        color=alt.Color("criterio:N", scale=alt.Scale(domain=["BIC", "AIC"], range=[AZUL, NARANJA]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["K:O", "criterio:N", alt.Tooltip("valor:Q", format=",.0f")],
    ).properties(height=260)
    st.altair_chart(lineas, width="stretch")
    k_minimo = int(seleccion.loc[seleccion["BIC"].idxmin(), "K"])
    st.caption(f"Con este dataset el BIC más bajo está en K = {k_minimo}. "
               "Se usa K = 4 en los dos datasets: una clase por categoría (D-09).")

    st.divider()
    st.subheader(f"ART 1 con ρ = {m['rho']}")
    c = st.columns(5)
    c[0].metric("Nodos", m["nodos"])
    c[1].metric("Compresión (CR)", f"{m['CR']:.3f}",
                help="Nodos / alumnos. Se pide entre 0,05 y 0,20.")
    c[2].metric("Pureza", f"{m['pureza_EM']:.1%}",
                help="Alumnos que comparten la categoría EM mayoritaria de su nodo.")
    c[3].metric("MAE", f"{m['MAE']:.2f} / 10",
                help="Error promedio entre la nota real y la esperada en el 20% reservado.")
    c[4].metric("Acierto de categoría", f"{m['acierto_categoria']:.1%}")

    barrido_rho = BARRIDOS_RHO[m["dataset"]]
    if os.path.exists(barrido_rho):
        barrido = pd.read_csv(barrido_rho)
        st.markdown("**Barrido de ρ** (por qué se eligió 0,65)")
        izquierda, derecha = st.columns(2, gap="large")
        izquierda.altair_chart(grafico_barrido(barrido, "pureza_EM", "Pureza", ".0%"),
                               width="stretch")
        derecha.altair_chart(grafico_barrido(barrido, "MAE", "MAE (sobre 10)", ".2f"),
                             width="stretch")
        if m["dataset"] == "student":
            st.caption("Con ρ = 0,65 la pureza pasa el 80% y el MAE es el más bajo "
                       "del barrido (D-13).")
        else:
            st.caption("En el dataset 2 se usa el mismo ρ = 0,65 para comparar la red "
                       "con los mismos parámetros (D-18).")
    else:
        st.info("Para ver el gráfico del barrido de ρ, corré antes: "
                f"`python -m src.barrido_rho {m['dataset']}`")

    st.divider()
    st.subheader("Validación en el 20% reservado")
    predicciones = modelo["predicciones"]

    # Comparación con lo más simple posible: darle a todos la nota promedio.
    promedio = modelo["nota_train"].mean()
    mae_promedio = (predicciones["nota_real"] - promedio).abs().mean()
    mayoritaria = predicciones["categoria_real"].value_counts(normalize=True)
    st.caption(f"Si a todos se les diera la nota promedio ({promedio:.2f}), el MAE sería "
               f"{mae_promedio:.2f} y el acierto de categoría {mayoritaria.iloc[0]:.1%} "
               f"(todos en {mayoritaria.index[0]}). La red tiene que mejorar eso.")
    izquierda, derecha = st.columns(2, gap="large")
    with izquierda:
        puntos = alt.Chart(predicciones).mark_circle(size=40, opacity=0.35, color=AZUL).encode(
            x=alt.X("nota_real:Q", title="Nota real", scale=alt.Scale(domain=[0, 10])),
            y=alt.Y("nota_esperada:Q", title="Nota esperada", scale=alt.Scale(domain=[0, 10])),
            tooltip=["estudiante", "nota_real", "nota_esperada", "nodo_art1"],
        )
        diagonal = alt.Chart(pd.DataFrame({"x": [0, 10], "y": [0, 10]})).mark_line(
            color="#898781", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
        st.altair_chart((puntos + diagonal).properties(height=320), width="stretch")
        st.caption("Cada punto es un alumno que la red no vio. "
                   "Sobre la línea punteada, la predicción es exacta.")
    with derecha:
        st.markdown("**Categoría real (filas) vs. pronosticada (columnas)**")
        cruce = pd.crosstab(predicciones["categoria_real"], predicciones["categoria_pronosticada"])
        cruce = cruce.reindex(index=CATEGORIES, columns=CATEGORIES, fill_value=0)
        st.dataframe(cruce, width="stretch")
        st.caption("La diagonal son los aciertos.")


def grafico_barrido(barrido, columna, titulo, formato):
    """Una métrica del barrido de ρ, con una línea vertical en el ρ elegido."""
    linea = alt.Chart(barrido).mark_line(color=AZUL, strokeWidth=2,
                                         point=alt.OverlayMarkDef(size=60, color=AZUL)).encode(
        x=alt.X("rho:Q", title="ρ (vigilancia)"),
        y=alt.Y(f"{columna}:Q", title=titulo, scale=alt.Scale(zero=False),
                axis=alt.Axis(format=formato)),
        tooltip=[alt.Tooltip("rho:Q", title="ρ"), alt.Tooltip("clusters:Q", title="Nodos"),
                 alt.Tooltip(f"{columna}:Q", title=titulo, format=formato)],
    )
    elegido = alt.Chart(pd.DataFrame({"rho": [RHO]})).mark_rule(
        color=NARANJA, strokeWidth=2).encode(x="rho:Q")
    return (linea + elegido).properties(height=220)


# ---------------------------------------------------------------------------
# Pestaña 3: un CSV con muchos alumnos
# ---------------------------------------------------------------------------

def pestana_csv(modelo):
    st.subheader("Predecir un archivo CSV")
    prep = modelo["prep"]
    variables = variables_de(modelo)
    columnas = list(variables)
    control = horas_de_control(prep)
    texto = ("El CSV tiene que tener estas columnas (enteros): " +
             ", ".join(f"`{c}`" for c in columnas) + ".")
    if control:
        texto += f" Si trae `{control[0]}`, también se aplica el chequeo de 24 hs."
    st.markdown(texto)

    ejemplo = pd.DataFrame([{c: inicial for c, (_, _, _, inicial) in variables.items()}])
    st.download_button("Descargar CSV de ejemplo", ejemplo.to_csv(index=False),
                       file_name="alumnos_ejemplo.csv", mime="text/csv")

    archivo = st.file_uploader("Elegí un CSV", type="csv",
                               key=f"csv_{modelo['metricas']['dataset']}")
    if archivo is None:
        return

    # Control de errores: cualquier problema con el archivo se muestra y no corta la app.
    try:
        alumnos = pd.read_csv(archivo)
        faltan = [c for c in columnas if c not in alumnos.columns]
        if faltan:
            st.error(f"Faltan columnas en el CSV: {', '.join(faltan)}")
            return
        if alumnos[columnas].isnull().any().any():
            st.error("El CSV tiene valores vacíos en las columnas de entrada.")
            return
        # Chequeo de 24 hs: en el dataset 1 necesita sleep_hours; en el dataset 2 el sueño
        # ya es una de las entradas.
        if all(c in alumnos.columns for c in control):
            alumnos = prep.validate_hourly_consistency(alumnos)
            descartados = modelo["prep"].invalid_rows_
            if descartados:
                st.warning(f"Se descartaron {descartados} alumnos por el chequeo de 24 hs.")
        if alumnos.empty:
            st.error("No quedaron alumnos para predecir.")
            return
        pred, _, _ = predecir_alumnos(modelo, alumnos)
    except Exception as error:  # noqa: BLE001
        st.error(f"No se pudo leer el archivo: {error}")
        return

    resultado = pd.concat([alumnos[columnas].reset_index(drop=True),
                           pred[["nodo_art1", "nota_esperada", "categoria_pronosticada",
                                 "perfil_em_nodo", "confianza_baja"]]], axis=1)

    conteo = resultado["categoria_pronosticada"].value_counts().reindex(CATEGORIES, fill_value=0)
    conteo = conteo.rename_axis("categoria").reset_index(name="alumnos")
    barras = alt.Chart(conteo).mark_bar(cornerRadiusEnd=4, height=22).encode(
        x=alt.X("alumnos:Q", title="Alumnos"),
        y=alt.Y("categoria:N", sort=CATEGORIES, title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("categoria:N", scale=escala_categorias(), legend=None),
        tooltip=["categoria:N", "alumnos:Q"],
    ).properties(height=170)
    st.altair_chart(barras, width="stretch")

    st.dataframe(resultado, hide_index=True, width="stretch")
    st.download_button("Descargar predicciones", resultado.to_csv(index=False),
                       file_name="predicciones.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Predictor EM + ART 1", layout="wide")
    st.title("Predictor del próximo examen")
    st.caption("TFI Inteligencia Artificial · EM (mezcla de gaussianas) + red ART 1 "
               f"(Carpenter-Grossberg) · ρ = {RHO}")

    # Control para pasar de un dataset al otro (la consigna pide 2 datasets).
    selector, _ = st.columns([2, 3])
    dataset = selector.select_slider("Dataset", options=list(DATASETS),
                                     format_func=lambda clave: DATASETS[clave]["corto"])
    st.caption(DATASETS[dataset]["nombre"])
    if dataset == "factors":
        st.info("Dataset 2 (benchmark): Student Performance Factors de Kaggle. Las horas de "
                "estudio son por semana y las tutorías por mes, y las dos se escalan a 100. "
                "El 98% de las notas está entre 6 y 8, así que casi todos son Aprobado Regular.")

    modelo = cargar_modelo(dataset)
    alumno, metricas, csv = st.tabs(["Un alumno", "Métricas del modelo", "Predecir un CSV"])
    with alumno:
        pestana_alumno(modelo)
    with metricas:
        pestana_metricas(modelo)
    with csv:
        pestana_csv(modelo)


main()
