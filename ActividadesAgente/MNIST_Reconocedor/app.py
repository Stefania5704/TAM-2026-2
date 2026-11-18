"""
PASO 3 — Interfaz web para arrastrar (o dibujar) un número y que el modelo lo reconozca.

Librería: Gradio. Construye una página web local con componentes listos
(zona para soltar imágenes, lienzo de dibujo, barras de probabilidad) a partir
de una función de Python. No hay que escribir HTML ni JavaScript.

Uso:
    python app.py
y abrir en el navegador la dirección que aparece (normalmente http://127.0.0.1:7860)
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # menos mensajes de TensorFlow

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

from preprocesamiento import preparar_imagen

# Cargar el modelo YA ENTRENADO (no se vuelve a entrenar: solo se usa = "inferencia")
modelo = tf.keras.models.load_model("modelo_mnist.keras")


def predecir(imagen):
    """Imagen -> (probabilidades por dígito, imagen 28x28 que ve el modelo)."""
    if imagen is None:
        return None, None
    # El lienzo de dibujo entrega un diccionario; la imagen final está en "composite"
    if isinstance(imagen, dict):
        imagen = imagen.get("composite")
        if imagen is None:
            return None, None
    vector, vista = preparar_imagen(imagen)
    if vista.max() == 0:
        return {"(no se detectó ningún trazo)": 1.0}, None

    probs = modelo.predict(vector, verbose=0)[0]           # 10 probabilidades que suman 1
    resultado = {str(d): float(p) for d, p in enumerate(probs)}
    # Se amplía la vista 28x28 a 280x280 sin suavizar, para ver los píxeles
    vista_grande = Image.fromarray(vista).resize((280, 280), Image.NEAREST)
    return resultado, vista_grande


with gr.Blocks(title="Reconocedor de dígitos — MNIST") as demo:
    gr.Markdown(
        "# Reconocedor de dígitos escritos a mano\n"
        "Arrastra una foto de un número (0-9) o dibújalo. "
        "A la derecha verás las probabilidades del modelo y la imagen de 28×28 "
        "que realmente recibe la red."
    )
    with gr.Row():
        with gr.Column():
            with gr.Tab("Arrastrar imagen"):
                entrada_img = gr.Image(type="pil", label="Suelta aquí tu imagen",
                                       sources=["upload", "clipboard", "webcam"])
                boton_img = gr.Button("Reconocer", variant="primary")
            with gr.Tab("Dibujar"):
                entrada_dib = gr.Sketchpad(type="pil", label="Dibuja un dígito",
                                           brush=gr.Brush(default_size=18,
                                                          colors=["#000000"],
                                                          color_mode="fixed"))
                boton_dib = gr.Button("Reconocer dibujo", variant="primary")
        with gr.Column():
            salida = gr.Label(num_top_classes=3, label="Predicción")
            vista = gr.Image(label="Lo que ve el modelo (28×28)")

    boton_img.click(predecir, inputs=entrada_img, outputs=[salida, vista])
    entrada_img.change(predecir, inputs=entrada_img, outputs=[salida, vista])
    boton_dib.click(predecir, inputs=entrada_dib, outputs=[salida, vista])

if __name__ == "__main__":
    demo.launch()
