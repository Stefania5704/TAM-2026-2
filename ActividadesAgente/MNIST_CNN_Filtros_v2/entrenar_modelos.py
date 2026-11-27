"""
PASO 1 — Entrenar todas las redes del proyecto y guardar los resultados en modelos/.

MNIST (clasificación). Se entrenan TRES redes:

    Red densa (la de la clase), con píxeles NORMALIZADOS (0-1):
        784 píxeles en fila -> Densa 64 (ReLU) -> Densa 64 (ReLU) -> Densa 10 (Softmax)

    La MISMA red densa con píxeles SIN normalizar (0-255, como en el cuaderno):
        sirve solo para comparar cómo cambia el aprendizaje.

    Red convolucional (igual a la del cuaderno scratchpad_4.ipynb, 50 186 parámetros):
        28x28x1 -> conv_1: 32 filtros 3x3 (ReLU, same) -> pool_1 (2x2)   => 14x14x32
                -> conv_2: 64 filtros 3x3 (ReLU, same) -> pool_2 (2x2)   =>  7x7x64
                -> Flatten (3136) -> Densa 10 (Softmax)

Todas usan EarlyStopping: se detienen cuando la pérdida de validación deja de
mejorar durante 3 épocas seguidas y se queda con los MEJORES pesos.

Regresión (recta y coseno): ver regresion.py. Aquí se entrenan con sus valores por
defecto para llenar la página de resumen.

Uso (con el entorno virtual activo, dentro de esta carpeta):
    python entrenar_modelos.py                 # máximo 30 épocas, paciencia 3
    python entrenar_modelos.py --epocas 5      # más rápido, para probar
    python entrenar_modelos.py --sin-regresion # solo MNIST

Resultado (carpeta modelos/):
    modelo_densa.keras        red densa entrenada (normalizada)
    modelo_cnn.keras          red convolucional entrenada
    metricas.json             exactitud, pérdida, parámetros, épocas usadas, historiales,
                              y los resultados de regresión
    resultados_prueba.npz     imágenes de prueba + predicciones de ambas redes (para la app)
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # menos mensajes de TensorFlow

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos")
SEMILLA = 42


# ---------------------------------------------------------------------------
# 1. DATOS
# ---------------------------------------------------------------------------
def cargar_mnist():
    """Devuelve (x_train, y_train), (x_test, y_test) con píxeles 0-255 (uint8)."""
    if os.path.exists("mnist.npz"):                 # copia local opcional (sin internet)
        d = np.load("mnist.npz")
        return (d["x_train"], d["y_train"]), (d["x_test"], d["y_test"])
    return keras.datasets.mnist.load_data()         # se descarga una sola vez (~11 MB)


# ---------------------------------------------------------------------------
# 2. MODELOS
# Se usa la API "funcional" de Keras y se le pone NOMBRE a cada capa.
# Los nombres son importantes: la app los usa para "abrir" la red y sacar
# la salida de cada capa convolucional (los mapas de características).
# ---------------------------------------------------------------------------
def crear_red_densa():
    entrada = keras.Input(shape=(784,), name="entrada")
    x = layers.Dense(64, activation="relu", name="densa_1")(entrada)
    x = layers.Dense(64, activation="relu", name="densa_2")(x)
    salida = layers.Dense(10, activation="softmax", name="salida")(x)
    return keras.Model(entrada, salida, name="red_densa")


def crear_red_convolucional():
    """La misma arquitectura del cuaderno: 2 bloques convolución + pooling y una capa final."""
    entrada = keras.Input(shape=(28, 28, 1), name="entrada")
    # padding="same": se agrega un borde de ceros para que la imagen NO se encoja
    # al aplicar el filtro 3x3 (28x28 entra, 28x28 sale).
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv_1")(entrada)
    # MaxPooling 2x2: de cada bloque de 2x2 se queda con el valor MÁS ALTO.
    # Reduce el mapa a la mitad (28 -> 14) y conserva "dónde hubo una respuesta fuerte".
    x = layers.MaxPooling2D((2, 2), name="pool_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_2")(x)
    x = layers.Flatten(name="aplanar")(x)                       # 7*7*64 = 3136 valores
    salida = layers.Dense(10, activation="softmax", name="salida")(x)
    return keras.Model(entrada, salida, name="red_convolucional")


# ---------------------------------------------------------------------------
# 3. ENTRENAR Y EVALUAR
# ---------------------------------------------------------------------------
def entrenar(modelo, x_tr, y_tr, x_val, y_val, epocas, lote, paciencia):
    modelo.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    modelo.summary()
    # EarlyStopping: vigila la pérdida de VALIDACIÓN; si no mejora en `paciencia` épocas,
    # corta y devuelve los pesos de la mejor época (restore_best_weights).
    parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia,
                                           restore_best_weights=True)
    t0 = time.time()
    hist = modelo.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                      epochs=epocas, batch_size=lote, verbose=2, callbacks=[parada])
    return hist.history, time.time() - t0


def matriz_confusion(y_real, y_pred):
    m = np.zeros((10, 10), dtype=int)
    np.add.at(m, (y_real, y_pred), 1)
    return m


def main():
    ap = argparse.ArgumentParser(description="Entrena las redes de MNIST y de regresión.")
    ap.add_argument("--epocas", type=int, default=30, help="máximo de épocas (EarlyStopping puede cortar antes)")
    ap.add_argument("--paciencia", type=int, default=3)
    ap.add_argument("--lote", type=int, default=128)
    ap.add_argument("--sin-regresion", action="store_true", help="no entrenar las redes de regresión")
    args = ap.parse_args()

    keras.utils.set_random_seed(SEMILLA)            # resultados repetibles
    os.makedirs(CARPETA, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = cargar_mnist()
    print(f"Entrenamiento: {x_train.shape}  Prueba: {x_test.shape}")

    # Validación = los últimos 6000 ejemplos de entrenamiento (54 000 para aprender)
    n_val = 6000
    y_tr_, yval = y_train[:-n_val], y_train[-n_val:]

    # Dos versiones de los píxeles: crudos (0-255, como el cuaderno) y normalizados (0-1)
    crudo = {"tr": x_train[:-n_val].astype("float32"), "val": x_train[-n_val:].astype("float32"),
             "te": x_test.astype("float32")}
    norm = {k: v / 255.0 for k, v in crudo.items()}

    # La red densa recibe vectores de 784; la CNN recibe imágenes 28x28x1
    plano = lambda a: a.reshape(len(a), 784)
    imagen = lambda a: a.reshape(len(a), 28, 28, 1)

    resultados = {}
    predicciones = {}
    trabajos = [("densa_sin_norm", crear_red_densa, plano, crudo),
                ("densa", crear_red_densa, plano, norm),
                ("cnn", crear_red_convolucional, imagen, norm)]
    for nombre, crear, formato, datos in trabajos:
        print("\n" + "=" * 70 + f"\n  Entrenando red {nombre.upper()}\n" + "=" * 70)
        keras.utils.set_random_seed(SEMILLA)        # misma semilla para una comparación justa
        modelo = crear()
        historial, segundos = entrenar(modelo, formato(datos["tr"]), y_tr_, formato(datos["val"]), yval,
                                       args.epocas, args.lote, args.paciencia)
        perdida, exactitud = modelo.evaluate(formato(datos["te"]), y_test, verbose=0)
        probs = modelo.predict(formato(datos["te"]), verbose=0, batch_size=512)
        pred = probs.argmax(axis=1)
        usadas = len(historial["loss"])
        mejor = int(np.argmin(historial["val_loss"])) + 1

        if nombre != "densa_sin_norm":              # la versión sin normalizar solo se compara
            predicciones[nombre] = pred.astype(np.uint8)
            predicciones[nombre + "_confianza"] = probs.max(axis=1).astype(np.float16)
            modelo.save(os.path.join(CARPETA, f"modelo_{nombre}.keras"))
        resultados[nombre] = {
            "exactitud_prueba": float(exactitud),
            "perdida_prueba": float(perdida),
            "parametros": int(modelo.count_params()),
            "segundos_entrenamiento": round(segundos, 1),
            "epocas_usadas": usadas,
            "mejor_epoca": mejor,
            "se_detuvo_antes": usadas < args.epocas,
            "historial": {k: [float(v) for v in vals] for k, vals in historial.items()},
            "matriz_confusion": matriz_confusion(y_test, pred).tolist(),
        }
        print(f"\n>>> {nombre}: exactitud en prueba = {exactitud:.4f}  "
              f"({modelo.count_params():,} parámetros, {usadas} épocas, mejor la {mejor}, {segundos:.0f} s)")

    resultados["config"] = {"epocas": args.epocas, "paciencia": args.paciencia, "lote": args.lote,
                            "semilla": SEMILLA, "n_entrenamiento": int(len(y_tr_)), "n_validacion": n_val,
                            "n_prueba": int(len(x_test)), "version_tensorflow": tf.__version__}

    if not args.sin_regresion:
        import regresion_resumen
        resultados["regresion"] = regresion_resumen.entrenar_todo()

    with open(os.path.join(CARPETA, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # Imágenes de prueba (uint8, comprimidas) + predicciones: la app las usa para
    # los "Ejemplos MNIST" y para la galería de errores, sin volver a calcular nada.
    np.savez_compressed(os.path.join(CARPETA, "resultados_prueba.npz"),
                        x=x_test.astype(np.uint8), y=y_test.astype(np.uint8), **predicciones)

    print("\nListo. Archivos guardados en:", CARPETA)
    for nombre in ("densa_sin_norm", "densa", "cnn"):
        r = resultados[nombre]
        print(f"  {nombre:15s} exactitud {r['exactitud_prueba']:.2%}   parámetros {r['parametros']:,}   "
              f"épocas {r['epocas_usadas']}")


if __name__ == "__main__":
    main()
