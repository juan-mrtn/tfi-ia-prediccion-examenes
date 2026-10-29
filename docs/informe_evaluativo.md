# Informe evaluativo (informe final)

Borrador. Se completa al cerrar el proyecto.

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
