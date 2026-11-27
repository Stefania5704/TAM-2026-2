"""
Utilidades compartidas por todas las páginas de la app.

Aquí viven las rutas, los formatos de número y la carga de métricas, para no
repetirlos en cada página.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELOS = os.path.join(AQUI, "modelos")


def miles(n):
    return f"{n:,}".replace(",", ".")


def pct(x, dec=1):
    return f"{x * 100:.{dec}f} %".replace(".", ",")


def num(x, dec=3):
    """Número con coma decimal (formato español)."""
    return f"{x:.{dec}f}".replace(".", ",")


def mostrar(fig):
    """Dibuja una figura de matplotlib en la página y luego la cierra (libera memoria)."""
    st.pyplot(fig)
    plt.close(fig)


@st.cache_resource
def cargar_resultados():
    """Lee metricas.json y las imágenes de prueba. Devuelve (metricas, prueba) o (None, None)."""
    ruta_m = os.path.join(MODELOS, "metricas.json")
    ruta_p = os.path.join(MODELOS, "resultados_prueba.npz")
    if not (os.path.exists(ruta_m) and os.path.exists(ruta_p)):
        return None, None
    with open(ruta_m, encoding="utf-8") as f:
        metricas = json.load(f)
    d = np.load(ruta_p)
    return metricas, {k: d[k] for k in d.files}


def exigir_modelos(archivos):
    """Si falta algún archivo en modelos/, muestra cómo entrenarlo y detiene la página."""
    faltan = [a for a in archivos if not os.path.exists(os.path.join(MODELOS, a))]
    if faltan:
        st.title("Falta entrenar las redes")
        st.write("No encontré estos archivos en la carpeta `modelos/`: " + ", ".join(faltan))
        st.write("Entrénalas una vez con este comando (tarda unos minutos) y recarga la página:")
        st.code("python entrenar_modelos.py", language="bash")
        st.stop()
