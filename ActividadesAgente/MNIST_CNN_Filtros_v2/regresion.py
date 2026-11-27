"""
Regresión con redes neuronales: una recta (y = m·x + b) y un coseno.

Aquí NO hay Streamlit: solo datos, redes y entrenamiento. Lo usan tanto
entrenar_modelos.py (para la tabla de resumen) como las páginas de la app.

Ideas del cuaderno scratchpad_4.ipynb que se conservan:
  * Partición 80/20 (entrenamiento / prueba) con train_test_split.
  * Dentro del entrenamiento se aparta otro 20 % para VALIDACIÓN.
  * Pérdida MSE (error cuadrático medio) y optimizador Adam.
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

SEMILLA = 42
FS = 1000                      # frecuencia de muestreo del coseno (Hz), como en el cuaderno
FOURIER_FRECUENCIAS = tuple(np.arange(0.5, 6.01, 0.5))   # banco de frecuencias (Hz) para la versión mejorada


# ---------------------------------------------------------------------------
# 1. DATOS
# ---------------------------------------------------------------------------
def datos_lineal(m=3.0, b=1.0, ruido=0.0, semilla=SEMILLA):
    """x entre -10 y 10 (paso 0,01), como en el cuaderno. y = m·x + b (+ ruido gaussiano)."""
    x = np.arange(-10, 10, 0.01).reshape(-1, 1).astype("float32")
    y = m * x + b
    if ruido > 0:
        y = y + np.random.default_rng(semilla).normal(0, ruido, y.shape)
    return x, y.astype("float32")


def datos_coseno(f=2.0, fs=FS):
    """t entre -2 y 2 s, cos(2π f t). Con f = 2 Hz hay 8 oscilaciones completas."""
    x = np.arange(-2, 2, 1 / fs).reshape(-1, 1).astype("float32")
    y = np.cos(2 * np.pi * f * x).astype("float32")
    return x, y


def particionar(x, y):
    """80 % entrenamiento / 20 % prueba (random_state = 42, igual que el cuaderno)."""
    return train_test_split(x, y, test_size=0.20, random_state=SEMILLA)


# ---------------------------------------------------------------------------
# 2. REDES
# ---------------------------------------------------------------------------
def crear_red_lineal():
    """Una sola neurona SIN activación: y = w·x + b. Solo 2 parámetros."""
    entrada = keras.Input(shape=(1,), name="entrada")
    salida = layers.Dense(1, name="neurona")(entrada)
    return keras.Model(entrada, salida, name="red_lineal")


class CaracteristicasFourier(layers.Layer):
    """
    Convierte la entrada x en [x, sin(2π f1 x), cos(2π f1 x), sin(2π f2 x), ...].
    Es como darle a la red un "peine" de relojes de distintas velocidades: así no tiene
    que inventar la oscilación, solo elegir cuáles relojes combinar.
    """

    def __init__(self, frecuencias=FOURIER_FRECUENCIAS, **kwargs):
        super().__init__(**kwargs)
        self.frecuencias = [float(f) for f in frecuencias]

    def call(self, x):
        w = tf.constant(self.frecuencias, dtype=x.dtype) * (2.0 * np.pi)
        ang = x * w[None, :]
        return tf.concat([x, tf.sin(ang), tf.cos(ang)], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg["frecuencias"] = self.frecuencias
        return cfg


def crear_red_coseno(neuronas=10, capas=3, activacion="selu", fourier=False, lr=1e-3):
    """Red totalmente densa para el coseno. Con capas=3, neuronas=10 y 'selu' es la del cuaderno."""
    entrada = keras.Input(shape=(1,), name="entrada")
    x = CaracteristicasFourier(name="fourier")(entrada) if fourier else entrada
    for i in range(capas):
        x = layers.Dense(neuronas, activation=activacion, name=f"densa_{i + 1}")(x)
    salida = layers.Dense(1, name="salida")(x)
    red = keras.Model(entrada, salida, name="red_coseno")
    red.compile(optimizer=keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return red


# Configuraciones con nombre (para comparar). La primera es EXACTAMENTE la del cuaderno.
CONFIGS_COSENO = {
    "cuaderno": dict(neuronas=10, capas=3, activacion="selu", fourier=False, lr=1e-3,
                     epocas=100, early=False, lote=32,
                     titulo="Red del cuaderno (3 × 10, SELU, 100 épocas)"),
    "tanh_ancha": dict(neuronas=64, capas=3, activacion="tanh", fourier=False, lr=1e-2,
                       epocas=300, early=True, lote=128,
                       titulo="Más neuronas + tanh + tasa mayor + EarlyStopping"),
    "fourier": dict(neuronas=32, capas=2, activacion="tanh", fourier=True, lr=1e-2,
                    epocas=300, early=True, lote=128,
                    titulo="Entradas de Fourier (sin y cos) + tanh + EarlyStopping"),
}


# ---------------------------------------------------------------------------
# 3. ENTRENAMIENTO
# ---------------------------------------------------------------------------
def _callbacks(early, paciencia):
    if not early:
        return []
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia,
                                          restore_best_weights=True)]


def _resumen_historial(hist, early):
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    val = h["val_loss"]
    mejor = int(np.argmin(val)) + 1
    return h, len(val), (mejor if early else None)


def entrenar_lineal(m=3.0, b=1.0, ruido=0.0, epocas=100, lr=0.05, lote=64):
    """Entrena la neurona única. Devuelve un diccionario con todo lo que la página necesita."""
    keras.utils.set_random_seed(SEMILLA)
    x, y = datos_lineal(m, b, ruido)
    x_tr, x_te, y_tr, y_te = particionar(x, y)
    red = crear_red_lineal()
    red.compile(optimizer=keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    t0 = time.time()
    hist = red.fit(x_tr, y_tr, validation_split=0.2, epochs=epocas, batch_size=lote, verbose=0)
    seg = time.time() - t0
    w, bb = red.layers[-1].get_weights()
    pred_te = red.predict(x_te, verbose=0)
    mse_te = float(np.mean((pred_te - y_te) ** 2))
    h, usadas, _ = _resumen_historial(hist, False)
    orden = np.argsort(x_te.ravel())
    return {"m_real": float(m), "b_real": float(b), "ruido": float(ruido),
            "m_aprendido": float(w[0, 0]), "b_aprendido": float(bb[0]),
            "mse_prueba": mse_te, "parametros": int(red.count_params()),
            "epocas_usadas": usadas, "historial": h, "segundos": round(seg, 1),
            "n_total": int(len(x)),
            "x_te": x_te.ravel()[orden], "y_te": y_te.ravel()[orden],
            "pred_te": pred_te.ravel()[orden]}


def entrenar_coseno(f=2.0, neuronas=10, capas=3, activacion="selu", fourier=False, lr=1e-3,
                    epocas=100, early=False, paciencia=30, lote=32):
    """Entrena una red para el coseno de frecuencia f. Devuelve todo lo que la página necesita."""
    keras.utils.set_random_seed(SEMILLA)
    x, y = datos_coseno(f)
    x_tr, x_te, y_tr, y_te = particionar(x, y)
    red = crear_red_coseno(neuronas, capas, activacion, fourier, lr)
    t0 = time.time()
    hist = red.fit(x_tr, y_tr, validation_split=0.2, epochs=epocas, batch_size=lote,
                   verbose=0, callbacks=_callbacks(early, paciencia))
    seg = time.time() - t0
    pred_te = red.predict(x_te, verbose=0)
    mse_te = float(np.mean((pred_te - y_te) ** 2))
    h, usadas, mejor = _resumen_historial(hist, early)
    orden = np.argsort(x_te.ravel())
    x_all = x.ravel()
    pred_all = red.predict(x, verbose=0, batch_size=1024).ravel()
    return {"f": float(f), "mse_prueba": mse_te, "mae_prueba": float(np.mean(np.abs(pred_te - y_te))),
            "varianza_senal": float(np.var(y)), "parametros": int(red.count_params()),
            "epocas_usadas": usadas, "mejor_epoca": mejor, "historial": h, "segundos": round(seg, 1),
            "x_te": x_te.ravel()[orden], "y_te": y_te.ravel()[orden], "pred_te": pred_te.ravel()[orden],
            "x_todo": x_all, "y_todo": y.ravel(), "pred_todo": pred_all}
