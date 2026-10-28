"""Fase 3: red ART 1 (Carpenter-Grossberg 1) según el Box 3 de Lau (1992).

Idea general, en palabras simples
---------------------------------
La red recibe a cada alumno como un vector de unos y ceros (20 bits: 5 variables
de 4 bits cada una) y lo guarda en un "nodo" junto con los alumnos parecidos.
Si no hay ningún nodo lo bastante parecido, crea uno nuevo.

La red tiene dos capas:
- F1 (entrada): una neurona por bit del vector. Acá entra el alumno.
- F2 (salida): una neurona por nodo (grupo de alumnos parecidos).

Cada nodo j de F2 tiene dos juegos de pesos:
- b[j] (pesos de abajo hacia arriba): sirven para elegir qué nodo se parece más.
- t[j] (pesos de arriba hacia abajo): son el "prototipo" del nodo, es decir,
  los bits que tienen en común todos los alumnos que cayeron en ese nodo.

Pasos para cada alumno x:
1. Elección: se calcula la activación de cada nodo, mu_j = b[j] · x.
   El nodo con mayor activación es el primer candidato.
2. Vigilancia: se mide qué parte de los unos del alumno está también en el
   prototipo: ||t[j] AND x|| / ||x||. Si es mayor que rho, el nodo "resuena"
   (acepta al alumno). Si no, hay un "reset": se descarta ese nodo y se
   prueba con el siguiente de mayor activación.
3. Aprendizaje (rápido): el nodo que aceptó al alumno se queda solo con los
   bits en común: t[j] = t[j] AND x, y b[j] = t[j] / (0.5 + ||t[j]||).
4. Nodo nuevo: si ningún nodo resuena, se crea uno nuevo cuyo prototipo es
   el propio alumno.

rho (vigilancia) controla qué tan exigente es la red: con rho alto pide
parecidos casi exactos y crea muchos nodos; con rho bajo agrupa más.

Pesos iniciales (Box 3): b = 1 / (1 + N) y t = 1, con N = cantidad de bits.
"""

from typing import Tuple

import numpy as np


class CarpenterGrossbergNetwork:
    """Red ART 1 con aprendizaje rápido.

    Los nombres en inglés se mantienen porque los usan predictor.py y los tests.

    Atributos principales:
        b: pesos de abajo hacia arriba, una fila por nodo.
        t: prototipos (pesos de arriba hacia abajo), una fila por nodo.
        num_active_clusters: cuántos nodos están en uso.
        total_resets: cuántos resets hubo en la última pasada por los datos.
        empty_inputs: cuántos vectores llegaron sin ningún uno.
    """

    def __init__(self, input_dim: int, max_categories: int = 500, rho: float = 0.75):
        """Crea la red vacía.

        input_dim: cantidad de bits por alumno (neuronas de F1).
        max_categories: máximo de nodos que puede tener F2.
        rho: vigilancia, entre 0 (exclusivo) y 1.
        """
        if not 0.0 < rho <= 1.0:
            raise ValueError("rho debe estar en (0, 1].")
        self.input_dim = input_dim
        self.max_categories = max_categories
        self.rho = rho

        # Pesos iniciales del Box 3. Se reserva lugar para todos los nodos
        # posibles, pero solo se usan los primeros num_active_clusters.
        self.b = np.full((max_categories, input_dim), 1.0 / (1.0 + input_dim))
        self.t = np.ones((max_categories, input_dim), dtype=np.int8)

        # Contadores
        self.num_active_clusters = 0
        self.total_resets = 0
        self.empty_inputs = 0

    # ------------------------------------------------------------------
    # Pasos de la red
    # ------------------------------------------------------------------

    def _activaciones(self, x: np.ndarray) -> np.ndarray:
        """Paso 1, elección: mu_j = b[j] · x para cada nodo en uso."""
        nodos_en_uso = self.num_active_clusters
        return self.b[:nodos_en_uso] @ x

    def _pasa_vigilancia(self, j: int, x: np.ndarray, unos_x: int) -> bool:
        """Paso 2, vigilancia: ¿el prototipo del nodo j cubre más de rho de los unos de x?"""
        unos_en_comun = self.t[j] @ x  # ||t[j] AND x||, porque son bits 0/1
        return unos_en_comun / unos_x > self.rho

    def _aprender(self, j: int, x: np.ndarray) -> None:
        """Paso 3, aprendizaje rápido: el nodo j se queda con los bits en común."""
        self.t[j] = self.t[j] & x
        self.b[j] = self.t[j] / (0.5 + self.t[j].sum())

    def _crear_nodo(self, x: np.ndarray) -> int:
        """Paso 4, nodo nuevo: ocupa el siguiente lugar libre de F2 y aprende x."""
        if self.num_active_clusters >= self.max_categories:
            raise RuntimeError("Se alcanzó la capacidad máxima de categorías en F2.")
        nuevo = self.num_active_clusters
        self.num_active_clusters += 1
        # El nodo nuevo tiene t = 1, así que t AND x = x: su prototipo es el alumno.
        self._aprender(nuevo, x)
        return nuevo

    # ------------------------------------------------------------------
    # Uso de la red
    # ------------------------------------------------------------------

    def evaluate_pattern(self, x: np.ndarray, update: bool = True) -> Tuple[int, int, bool]:
        """Presenta un alumno x a la red y devuelve (nodo, resets, resonó).

        update=True: modo entrenamiento, la red aprende (y puede crear nodos).
        update=False: modo predicción, los pesos no cambian. Si ningún nodo
        resuena, devuelve el de mayor activación con resonó = False.
        Un vector sin unos no se puede evaluar: devuelve (-1, 0, False).
        """
        unos_x = int(x.sum())  # ||x||
        if unos_x == 0:
            self.empty_inputs += 1
            return -1, 0, False

        mu = self._activaciones(x)

        # Se prueban los nodos de mayor a menor activación (en empate, el de
        # menor número). Cada nodo que no pasa la vigilancia cuenta un reset.
        resets = 0
        for j in np.argsort(-mu, kind="stable"):
            if self._pasa_vigilancia(j, x, unos_x):
                if update:
                    self.total_resets += resets
                    self._aprender(j, x)
                return int(j), resets, True
            resets += 1

        # Ningún nodo resonó.
        if not update:
            # En predicción no se crean nodos: se devuelve el más activado.
            mas_activado = int(np.argmax(mu)) if len(mu) else -1
            return mas_activado, resets, False

        nuevo = self._crear_nodo(x)
        self.total_resets += resets
        return nuevo, resets, True

    def fit_predict(self, X_bin: np.ndarray, max_epochs: int = 5) -> np.ndarray:
        """Entrena con todos los alumnos y devuelve el nodo de cada uno.

        Los alumnos se presentan varias veces (épocas) hasta que ninguno
        cambia de nodo, o hasta max_epochs.
        """
        nodos = np.full(len(X_bin), -1)
        for _ in range(max_epochs):
            self.total_resets = 0
            nodos_nuevos = np.array([self.evaluate_pattern(x)[0] for x in X_bin])
            if np.array_equal(nodos_nuevos, nodos):
                break  # nadie cambió de nodo: la red ya está estable
            nodos = nodos_nuevos
        return nodos

    def predict(self, X_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Asigna alumnos nuevos a un nodo sin aprender. Devuelve (nodos, resonó)."""
        resultados = [self.evaluate_pattern(x, update=False) for x in X_bin]
        nodos = np.array([nodo for nodo, _, _ in resultados])
        resono = np.array([ok for _, _, ok in resultados])
        return nodos, resono
