"""
Atelier neuronal — clasificación (MNIST) y regresión con redes neuronales.

Cómo ejecutarla (dentro de esta carpeta, con el entorno virtual activo):
    python entrenar_modelos.py      # solo la primera vez (crea la carpeta modelos/)
    streamlit run app.py            # abre http://localhost:8501

Este archivo es el "portero": carga el estilo y define las páginas.
Cada página vive en la carpeta paginas/ y trae sus propios controles en la barra lateral.

RECORDATORIO DE STREAMLIT:
El script de la página activa se ejecuta COMPLETO, de arriba a abajo, cada vez que
mueves un control. Por eso lo pesado se guarda en caché (@st.cache_resource /
@st.cache_data) o en st.session_state.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import streamlit as st

from comun import AQUI

st.set_page_config(page_title="Atelier neuronal", page_icon="🌸", layout="wide")

# Inyectar nuestro CSS (tipografías, colores, tarjetas)
with open(os.path.join(AQUI, "estilo.css"), encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

navegacion = st.navigation({
    "Clasificación MNIST": [
        st.Page("paginas/mnist.py", title="Dígitos escritos a mano", icon="🔢", default=True),
    ],
    "Regresión": [
        st.Page("paginas/regresion_lineal.py", title="Regresión lineal", icon="📈"),
        st.Page("paginas/regresion_coseno.py", title="Regresión de un coseno", icon="〰️"),
    ],
    "Balance": [
        st.Page("paginas/resumen.py", title="Resumen de modelos", icon="📋"),
    ],
})
navegacion.run()
