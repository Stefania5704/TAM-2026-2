"""
PASO 1 — Entrenar el modelo del cuaderno "Actividad de clase" y GUARDARLO en disco.

Es la misma red de la clase:
    Entrada (784) -> Densa 64 (ReLU) -> Densa 64 (ReLU) -> Densa 10 (Softmax)

Única mejora: se normalizan los píxeles de [0, 255] a [0, 1] (ver explicación abajo).

Uso:
    python entrenar_modelo.py
Resultado:
    modelo_mnist.keras   (el modelo ya entrenado, listo para usarse en app.py)
"""
import os

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# 1. DATOS
# MNIST: 70 000 imágenes de 28x28 píxeles en escala de grises con dígitos
# escritos a mano (60 000 para entrenar, 10 000 para evaluar).
# Formato: dígito BLANCO (255) sobre fondo NEGRO (0), centrado en la imagen.
# ---------------------------------------------------------------------------
if os.path.exists("mnist.npz"):          # copia local opcional (sin internet)
    d = np.load("mnist.npz")
    x_train, y_train, x_test, y_test = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
else:                                     # descarga normal, igual que en el cuaderno
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# "Aplanar": cada imagen 28x28 se vuelve un vector de 784 números,
# porque una capa Densa recibe vectores, no matrices.
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)

# Normalizar a [0, 1]. Con valores de 0 a 255 las sumas ponderadas de la
# primera capa salen enormes y el entrenamiento arranca inestable
# (en el cuaderno, la pérdida de la época 1 fue 1.56). Con [0, 1] converge más rápido.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ---------------------------------------------------------------------------
# 2. MODELO (idéntico al del cuaderno)
# ---------------------------------------------------------------------------
In = tf.keras.layers.Input(shape=(784,))
Cp1 = tf.keras.layers.Dense(64, activation="relu")(In)
Cp2 = tf.keras.layers.Dense(64, activation="relu")(Cp1)
Out = tf.keras.layers.Dense(10, activation="softmax")(Cp2)
Model = tf.keras.Model(inputs=In, outputs=Out)
Model.summary()

# ---------------------------------------------------------------------------
# 3. ENTRENAMIENTO
# optimizer='adam'                           -> regla para ajustar los pesos
# loss='sparse_categorical_crossentropy'     -> mide qué tan equivocada está la predicción
# epochs                                     -> cuántas veces se recorre todo el dataset
# validation_split=0.1                       -> aparta el 10% para vigilar sobreajuste
# ---------------------------------------------------------------------------
Model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
Model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.1)

# ---------------------------------------------------------------------------
# 4. EVALUACIÓN con imágenes que el modelo NUNCA vio
# (el cuaderno solo mostraba la exactitud de entrenamiento)
# ---------------------------------------------------------------------------
perdida, exactitud = Model.evaluate(x_test, y_test, verbose=0)
print(f"\nExactitud en el conjunto de prueba: {exactitud:.4f}")

# ---------------------------------------------------------------------------
# 5. GUARDAR — esto es "bajar el modelo": los pesos aprendidos quedan en un
# archivo y ya no hay que volver a entrenar cada vez que se quiera usar.
# ---------------------------------------------------------------------------
Model.save("modelo_mnist.keras")
print("Modelo guardado en modelo_mnist.keras")
