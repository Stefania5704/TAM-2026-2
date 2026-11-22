"""
Reconocedor de dígitos escritos a mano con Streamlit.

Cómo ejecutarlo (desde la carpeta del proyecto, con el .venv activo):
    streamlit run app_streamlit.py
y abrir en el navegador: http://localhost:8501

IDEA CLAVE DE STREAMLIT:
Este archivo se ejecuta COMPLETO, de arriba a abajo, cada vez que el usuario
hace algo en la página (subir una imagen, mover un control, etc.).
Cada st.algo(...) "dibuja" un elemento en la página, en el orden en que aparece.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # menos mensajes de TensorFlow

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from preprocesamiento import preparar_imagen

# ---------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA (debe ser la primera instrucción de Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Reconocedor de dígitos — MNIST", page_icon="🔢", layout="wide")


# ---------------------------------------------------------------------------
# 2. CARGAR EL MODELO UNA SOLA VEZ
# Como el script se re-ejecuta en cada interacción, sin este decorador el
# modelo se cargaría de nuevo cada vez (lento). @st.cache_resource lo guarda
# en memoria la primera vez y luego lo reutiliza.
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_mnist.keras")


modelo = cargar_modelo()


def predecir(imagen):
    """Imagen PIL -> (probabilidades de 0 a 9, imagen 28x28 que ve el modelo)."""
    vector, vista = preparar_imagen(imagen)
    if vista.max() == 0:                     # no se encontró ningún trazo
        return None, None
    probs = modelo.predict(vector, verbose=0)[0]
    return probs, vista


def mostrar_resultado(imagen, nombre=""):
    """Dibuja en la página: imagen original | lo que ve el modelo | probabilidades."""
    probs, vista = predecir(imagen)
    if probs is None:
        st.warning(f"No se detectó ningún trazo en {nombre or 'la imagen'}.")
        return

    digito = int(np.argmax(probs))
    confianza = float(probs[digito])

    col1, col2, col3 = st.columns([1, 1, 2])          # tres columnas lado a lado
    with col1:
        st.image(imagen, caption=f"Original {nombre}", width="stretch")
    with col2:
        vista_grande = Image.fromarray(vista).resize((280, 280), Image.NEAREST)
        st.image(vista_grande, caption="Lo que ve el modelo (28×28)", width="stretch")
    with col3:
        st.metric("Predicción", digito, f"confianza {confianza:.1%}", delta_color="off")
        if confianza < 0.6:
            st.info("Confianza baja: revisa que la imagen tenga un solo dígito, "
                    "trazo grueso y fondo liso.")
        tabla = pd.DataFrame({"probabilidad": probs}, index=[str(d) for d in range(10)])
        tabla.index.name = "dígito"
        st.bar_chart(tabla, height=220)


# ---------------------------------------------------------------------------
# 3. BARRA LATERAL: información sobre el modelo
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Sobre el modelo")
    st.markdown(
        "- **Datos:** MNIST (60 000 imágenes de entrenamiento)\n"
        "- **Red:** 784 → 64 (ReLU) → 64 (ReLU) → 10 (softmax)\n"
        "- **Parámetros:** 55 050\n"
        "- **Entrada:** imagen 28×28, dígito claro sobre fondo oscuro"
    )
    st.caption("Tu imagen se convierte automáticamente al formato MNIST "
               "antes de pasarla a la red (ver preprocesamiento.py).")

# ---------------------------------------------------------------------------
# 4. CUERPO DE LA PÁGINA
# ---------------------------------------------------------------------------
st.title("🔢 Reconocedor de dígitos escritos a mano")
st.write("Arrastra una o varias fotos de números (0-9), o toma una foto con la cámara.")

pestana_subir, pestana_camara = st.tabs(["📁 Subir imágenes", "📷 Cámara"])

with pestana_subir:
    # st.file_uploader crea la zona donde se arrastran los archivos.
    # Devuelve una lista de archivos (o lista vacía si no hay ninguno).
    archivos = st.file_uploader(
        "Arrastra aquí tus imágenes",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=True,
    )
    for archivo in archivos:
        st.divider()
        mostrar_resultado(Image.open(archivo), archivo.name)

with pestana_camara:
    foto = st.camera_input("Toma una foto de un número escrito en papel")
    if foto is not None:
        mostrar_resultado(Image.open(foto), "(cámara)")
