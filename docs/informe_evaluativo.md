# Informe evaluativo (informe final)

El informe completo está en **`docs/Informe_evaluativo_TFI.docx`** (corridas con ambos datasets,
evaluación de cada etapa, alcance y limitaciones). Sus figuras están en `docs/figuras_evaluativo/`.

## Resumen de las corridas (K = 4, ρ = 0,65, 20% reservado)

| Métrica | Dataset 1 · simulado | Dataset 2 · Student Performance Factors |
|---|---|---|
| BIC con mínimo en K = 4 | Sí | No (sigue bajando hasta K = 7) |
| Silhouette (meta ≥ 0,40) | 0,19 | 0,32 |
| Nodos ART 1 / CR | 871 / 0,109 | 394 / 0,075 |
| Pureza respecto de EM (meta ≥ 80%) | 81,4% | 84,6% |
| MAE del sistema | 0,94 | 0,23 |
| MAE prediciendo la media | 1,33 | 0,29 |
| MAE usando solo EM | 0,99 | 0,24 |
| Acierto de categoría | 57,2% | 98,0% |
| Acierto diciendo siempre la más común | 43,0% | 98,0% |

En el dataset 2 el 98% de los alumnos es Aprobado Regular, así que su acierto de categoría es trivial.

## Limitaciones

### La nota esperada no siempre sube al mejorar un hábito (D-17)

Se tomaron 300 alumnos del 20% reservado y se les mejoró una variable por vez (+1 h de estudio,
+2 tareas, +10 de asistencia o +10 de nota previa). En promedio la nota esperada sube (+0,11, +0,05,
+0,11 y +0,32 puntos), pero en el 17,7% de los casos baja más de 0,3 puntos. Ejemplo: con 9 tareas
el alumno recibe 6,16 y con 10 tareas, 5,88.

Causas:

- ART 1 no sabe que "más es mejor": solo compara bits. Un cambio de bin lleva al alumno a otro nodo,
  con otros compañeros y otra nota media.
- El código binario posicional (D-12) no es ordinal: 4 (`0100`) y 5 (`0101`) difieren en un bit,
  pero 7 (`0111`) y 8 (`1000`) no comparten ninguno.
- 399 de los 847 nodos tienen menos de 5 alumnos, así que su nota media es ruidosa.

Es una limitación propia de la arquitectura (ART 1 con la codificación binaria que pide la cátedra),
no un error del código. Se decidió mantener el modelo así. La diferencia suele estar dentro del error
normal del modelo (MAE 0,94 sobre 10), pero cerca de un límite de categoría puede cambiarla.

Las demás limitaciones están en la sección 7.2 de `docs/Informe_evaluativo_TFI.docx`.
