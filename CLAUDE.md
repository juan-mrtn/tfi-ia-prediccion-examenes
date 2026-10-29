# CLAUDE.md - TFI Inteligencia Artificial

Contexto del proyecto para no perder el hilo. Mantener actualizado cuando se tome una decisión.

## Objetivo

Predictor no supervisado híbrido que estima, para un estudiante, la **nota esperada (0–10)** y la
**categoría** de su próximo examen, a partir de sus hábitos diarios y antecedentes académicos.

- **EM** = algoritmo Expectation-Maximization (mezcla de gaussianas, GMM).
- **ART 1** = red Carpenter-Grossberg 1 (Box 3, Clifford Lau, 1992).
- Categorías: Promoción (ŷ > 8), Aprobado Regular (6–8), Recuperatorio (4–6), Riesgo Crítico (< 4).

**Fuera de alcance (corrección de cátedra):** alertas de deserción, recomendaciones o intervenciones
pedagógicas, y cualquier uso que no sea predecir el próximo examen. "Abandono" es solo el nombre de
la categoría más baja; el sistema no predice deserción.

## Equipo

Acuña, Juan (líder) · Blanchet, Sofía (código) · Parodi, Facundo (arquitectura) ·
Pereyra, Lucrecia (testing) · Schell, Brian (repositorio) · Viñuela, Tomás (documentación).
Docentes: Dra. Daniela López De Luise, Prof. Alejandro Pérez. UADER, FCyT, IDTI Lab.

## Documentos de referencia

- `docs/Arquitectura_EM_ART1_TFI (2).docx` y `docs/especificacion_arquitectura_em_art1.md`: arquitectura,
  fórmulas, backlog (ISSUE-101 a 403) y plan de verificación.
- `1er-avance-TPF.pdf`: primera entrega (tiene partes ya corregidas, ver abajo).
- `TPF.2026.Trabajo integrador_v1.pdf`: consigna.
- `Lau.pp5.a.11.pdf`, `Lau.pp12.a.14.pdf`: bibliografía ART 1 (no se versionan, son pesados).

## Restricciones de cátedra

- Punto de entrada único `CarGross.py`, CLI, PEP 8, sin warnings.
- Entrada y salida solo en `.csv` (también metadatos).
- Control total de excepciones: ante error se muestra el manual `man` y se sale con código controlado.
- Dos datasets (≥ 50 filas, ≥ 5 variables): el simulado (D-10) y el benchmark Student Performance
  Factors (D-18).
- Entregables: manual de referencia (con FAQ y sección REFERENCIAS), informe final con carátula,
  guion de PPT con audio.

## Pipeline

1. **EM** sobre las variables continuas sin adaptar. K elegido por BIC/AIC, con una cantidad de
   clases adecuada para la red. Esas clases son las que se usan para entrenar.
2. **Riesgo por eigenvectores:** el nivel de cada cluster se deriva del autovector principal de su
   matriz de covarianza. Si el dataset es bueno, EM separa por nivel de riesgo.
3. **Preprocesamiento:** escalado a base 100, enteros sin decimales, histograma de 10 bins,
   codificación posicional binaria de 4 bits.
4. **ART 1:** cada instancia se asocia a su clase EM, así la red aprende los patrones de cada clase.
5. **Predicción:** el alumno nuevo se asigna al nodo resonante; se devuelve la nota media del nodo y
   su categoría. `exam_score` nunca entra al vector de entrada.
6. **Validación ex-post** sobre el 20% reservado.

## Correcciones del docente al 1er avance

Son previas al documento de arquitectura y ya están aplicadas en él.


- Clasificar con EM sobre las variables sin adaptar; las clases resultantes son las de entrenamiento
  y deben ser una cantidad adecuada para la arquitectura de la red.
- Asociar las variables de cada instancia a las clases del EM para enseñar los patrones de cada clase.
- Tareas: un único número genera demasiado sesgo. Generar **una variable por cada tipo de tarea**.
- Notas previas: deben ser de materias comparables (las que tengan en común); si no, sesgan.
- No usar decimales.
- Órdenes de magnitud similares en todas las variables (2 o 3 dígitos).
- Estado final binario (Placed / Not Placed) no sirve: usar mayor granularidad.
- Escalar a 100 en todos los casos: `valor_nuevo = ((Max - Min) / 100) * valor_actual`.
- Agregar las métricas de EM (log-likelihood, BIC, AIC, silhouette, entropía de responsabilidades).
- Binarizar con histograma de 10 valores y notación posicional binaria (0 a 10).
- El riesgo de cada cluster EM se deriva de su eigenvector.

## Dataset 1 original (Kaggle, reemplazado por el simulado, ver D-10)

`data/raw/student_performance_kaggle_original.csv`, 10.000 filas. Columnas: study_hours (0–11), attendance (0–100),
sleep_hours (≤ 9), internet_usage (0–11), assignments_completed (0–20), previous_score (0–95),
exam_score (0–100, con decimales), placement_status (Placed 83,6% / Not Placed 16,4%).

- Ninguna fila viola la regla de 24 hs.
- El 35% de `exam_score` vale 100 y la media es 86,7: en escala 0–10 casi todos caen en Promoción,
  así que las categorías quedan desbalanceadas.
- Correlación con exam_score: study_hours 0,56 · assignments 0,39 · previous_score 0,32 ·
  attendance 0,22 · sleep 0,14 · internet −0,15. Las variables son casi independientes entre sí.
- No existen columnas tp1/tp2/taller. El documento de arquitectura divide `assignments_completed` en
  task_tp1, task_tp2 y task_taller según `data/metadata/student_metadata.csv`, pero no define la regla.

## Decisiones

- Se usarán pocas variables (4 a 5) para que el sistema sirva con otros datasets.
- `placement_status` no se usa como entrada (es un resultado).
- **Tareas (2026-10-27):** `assignments_completed` se mantiene como **una sola variable escalada a 100**
  (`valor × 100 / (Max − Min)`, Max = 20, entero sin decimales). No se desagrega en tp1/tp2/taller
  porque el dataset no trae las tareas por tipo y dividirla sería inventar datos (3 columnas
  perfectamente correlacionadas que triplicarían su peso). Descartarla tampoco: es la 2ª variable más
  relacionada con la nota (r = 0,39; R² lineal baja de 64% a 49% sin ella). La mezcla de tipos de
  tarea queda como limitación en el manual. Consultar al docente.
- **Variables (2026-10-27):** study_hours, assignments_completed (a 100), previous_score, attendance,
  internet_usage. sleep_hours queda afuera.
- **Filtro de 24 hs:** queda solo como chequeo de calidad de datos (sleep_hours no entra a la red).
- **Horas en 3 dígitos (pedido del docente):** study_hours e internet_usage NO se
  escalan a 100; se guardan como 000 a 024. El escalado a 100 es solo para las tareas (attendance y
  previous_score ya están en 0–100).
- **Bins de horas (D-07):** no uniformes. 0–1→0, 2→1, 3→2, 4→3, 5→4, 6→5, 7→6, 8→7, 9–10→8,
  11–12→9, >12→10. Las demás variables (0–100) usan bin = valor / 10.
- **K del EM (D-08):** K de 2 a 7, se elige el menor BIC; AIC como control. Elegido por el equipo, no lo
  exige la cátedra.
- **K = 4 (D-09):** el BIC baja hasta K = 7 sin mínimo; se fija K = 4 (una clase por categoría).
- **Dataset simulado (D-10, 2026-10-27, aprobado por la docente):** el de Kaggle es demasiado
  sintético, así que `data/raw/student_performance.csv` pasa a ser un dataset simulado de 10.000 alumnos
  generado por `scripts/generar_dataset_simulado.py` (semilla 2026, 4 perfiles latentes: comprometido,
  regular, distraído, desenganchado). Mismas columnas y rangos, enteros. El original quedó en
  `data/raw/student_performance_kaggle_original.csv`. Nota media 62, 0,9% en 100. BIC con mínimo en
  K = 4–5; los grupos explican el 40% de la nota (Kaggle: 12%). Validación:
  `scripts/validar_dataset_simulado.py`. **En el informe aclarar que son datos simulados.**
- **EM implementado (D-11):** `src/em_clustering.py`, GMM full, n_init 5, reg_covar 1e-3. Con el dataset
  simulado el BIC tiene mínimo en K = 4 y ℛ ordena bien las 4 categorías (nota media 75,9 / 65,0 / 54,7 / 44,5).
- **Codificación (D-12):** solo binaria posicional de 4 bits (20 bits por alumno). `src/encoder.py`,
  `src/art1.py` y `src/predictor.py` implementados. ρ = 0,65: pureza 81%, 871 nodos, MAE 0,94/10.
  Barrido de ρ: `python -m src.barrido_rho` → `data/output/barrido_rho_art1.csv`.
- **ρ = 0,65 (D-13, 2026-10-28):** primer ρ con pureza ≥ 80% (81,4%), CR 0,109 y menor MAE (0,94/10).
  Explicado en `docs/Informe_avance_TFI.docx`, sección 6.3.
- **Métricas (ISSUE-303/304, 2026-10-28):** `src/metrics.py` reúne pureza, CR, RR, MAE, acierto y varianza
  por nodo; `python -m src.metrics` genera `data/output/predicciones_student.csv` y `metricas_student.csv`.
  Con ρ = 0,65: 871 nodos, pureza 81,4%, MAE 0,935, acierto 57,2%. El barrido reutiliza estas funciones.
- **App visual (D-14, 2026-10-28, Juan eligió):** `src/app_visual.py` con Streamlit, se ejecuta aparte con
  `streamlit run src/app_visual.py`. Complementa a la CLI (CarGross.py sigue siendo el punto de entrada
  único que pide la cátedra). Reutiliza `entrenar_pipeline()` de `src/metrics.py`, así muestra los
  mismos resultados que `python -m src.metrics`.
- **Categoría desde la nota (D-15, 2026-10-28, pedido de Juan):** `ExamPredictor.predict` calcula la
  categoría con la nota esperada (Promoción > 8, Regular 6–8, Recuperatorio 4–6, Riesgo < 4). Antes
  usaba el perfil EM más común del nodo y a veces no coincidía con la nota (7,78 → Recuperatorio).
  El perfil EM queda en la columna `perfil_em_nodo`. Acierto de categoría: 47,4% → 57,2%; MAE y
  pureza no cambian, así que D-13 sigue valiendo.
- **Confianza baja (D-16, 2026-10-28):** si ningún nodo pasa la vigilancia, el predictor usa el nodo
  que más unos del alumno cubre (‖T ∩ X‖/‖X‖) y, si empatan, el de más alumnos. Antes usaba el más
  activado, que suele ser un nodo de 1 alumno (ej.: tareas 10 daba 4,70 y tareas 9 daba 6,16; ahora
  5,88). En el 20% reservado pasa con 1 alumno, así que las métricas no cambian.
  Limitación a explicar: el código binario posicional no es ordinal (0111 = 7 y 1000 = 8 no comparten
  bits) y 399 de 847 nodos tienen menos de 5 alumnos, así que la nota no siempre sube con cada variable.
- **Nota no monótona (D-17, 2026-10-28, Juan decidió):** se mantiene el modelo tal cual. En el 17,7% de
  los casos mejorar una variable baja la nota más de 0,3 puntos (ART 1 compara bits, código no ordinal,
  nodos chicos). Va como limitación en el informe de avance (secciones 6.4 y 8) y en
  docs/informe_evaluativo.md. No proponer más alternativas para esto.
- **Dataset 2 (D-18, 2026-10-28, Juan eligió):** Student Performance Factors (Kaggle,
  `data/raw/StudentPerformanceFactors.csv`, 6.607 alumnos, sintético). Reglas en
  `data/metadata/factors_metadata.csv`: Hours_Studied (horas por SEMANA, 0–44, escalada a 100),
  Attendance, Sleep_Hours (horas, entra a la red), Tutoring_Sessions (0–8, escalada a 100),
  Previous_Scores; objetivo Exam_Score (una nota de 101 se recorta a 100). Ningún dataset público tiene
  las 5 variables del dataset 1. Mismo pipeline, K = 4 y ρ = 0,65: 394 nodos, pureza 84,6%, MAE 0,23
  (nota promedio para todos: 0,29), acierto 98% pero el 98% de las notas es Aprobado Regular. BIC sin
  mínimo hasta K = 7. Código: `DATASETS` en `src/metrics.py`; `python -m src.metrics factors`,
  `python -m src.barrido_rho factors`; la app tiene un control "Dataset" arriba. El chequeo de 24 hs
  ahora sale del metadata (variables tipo horas).
- **CLI (D-19, 2026-10-28, ISSUE-301):** `CarGross.py` con `--mode train|eval|predict|sweep|demo`, datos con
  `-d datos.csv -m metadata.csv` o el atajo `--dataset 1|2` (1 = simulado, por defecto), `-i` para los
  alumnos nuevos de predict, `-o` carpeta (data/output) y `--rho` (0,65). Reutiliza `entrenar_pipeline()`
  y `barrer_rho()`, así da los mismos números que `python -m src.metrics`. predict usa el modelo entrenado
  con el 80%, como la app. eval agrega `matriz_confusion_<nombre>.csv`. demo = eval con los primeros 2000
  alumnos del dataset 1 (~2,5 s). Errores: se muestra el motivo y el man (`src/man_pages.py`) y se sale
  con 2 (opciones), 3 (archivo), 4 (datos) o 1 (inesperado); nunca un traceback. Tests en
  `tests/test_cli_man.py`.
- Las decisiones se registran también en `docs/Registro_de_decisiones.docx`.

## Pendientes

- Ajustar el documento de arquitectura (sección 4.2) con las decisiones de tareas, horas y bins.
- Confirmar con el docente la fórmula de escalado: escrita como `((Max − Min) / 100) × valor`
  achica la escala; se usa la inversa `valor × 100 / (Max − Min)`.
- Confirmar con el docente que se acepta una sola variable de tareas para este dataset.
- Actualizar el documento de arquitectura (sección 4.2, ISSUE-203) con la decisión de tareas.
- Verificar si `previous_score` corresponde a materias comparables.
- Repartir los roles del RACI y cargar los issues en GitHub.
