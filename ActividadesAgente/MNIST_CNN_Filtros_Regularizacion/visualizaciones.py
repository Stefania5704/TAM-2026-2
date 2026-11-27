"""
Funciones de dibujo y de cálculo que usa la app (para que app.py quede más corto).

Aquí NO hay nada de Streamlit: solo numpy + matplotlib.
Cada función recibe arreglos de números y devuelve una figura o una imagen.
"""
import base64
import io

import matplotlib

matplotlib.use("Agg")                      # dibujar sin ventana (en un servidor)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# ---------------------------------------------------------------------------
# PALETA (la misma que usa el CSS de la app)
# ---------------------------------------------------------------------------
CIRUELA = "#3B1C3F"      # texto y tinta principal
ORQUIDEA = "#B03A78"     # acento: lo importante, lo que "se activa"
CUARZO = "#F3D3E0"       # rosa suave para rellenos
NIEBLA = "#F8F3F9"       # fondo
SALVIA = "#4E7F70"       # valores negativos (contraste frío)
ORO_ROSA = "#D8A48F"     # detalles

# Mapa "pétalo": 0 = casi blanco, máximo = ciruela. Para ACTIVACIONES (siempre >= 0 tras ReLU).
PETALO = LinearSegmentedColormap.from_list(
    "petalo", ["#FBF7FB", CUARZO, "#DE86B0", ORQUIDEA, CIRUELA])
# Mapa divergente: negativo = salvia, cero = blanco, positivo = orquídea. Para PESOS / KERNELS.
PESOS = LinearSegmentedColormap.from_list(
    "pesos", ["#2F5D50", SALVIA, "#B9D5CB", "#FBF7FB", "#EBB0CB", ORQUIDEA, "#6B1C48"])
# Mapa "noche": cómo ve el modelo el dígito (trazo claro sobre fondo oscuro, como MNIST).
NOCHE = LinearSegmentedColormap.from_list("noche", ["#22102A", "#7A2E62", "#E6A3C3", "#FFF6FA"])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": CIRUELA, "axes.labelcolor": CIRUELA,
    "xtick.color": CIRUELA, "ytick.color": CIRUELA,
    "axes.edgecolor": CUARZO,
})


# ---------------------------------------------------------------------------
# IMÁGENES PEQUEÑAS (para incrustar en HTML)
# ---------------------------------------------------------------------------
def arreglo_a_png_b64(a, cmap=NOCHE, escala=8, vmin=None, vmax=None, tamano=None):
    """Matriz 2D -> PNG en base64 (texto) agrandado con píxeles nítidos.
    tamano=(ancho, alto) fuerza un tamaño exacto en píxeles."""
    a = np.asarray(a, dtype=float)
    lo = a.min() if vmin is None else vmin
    hi = a.max() if vmax is None else vmax
    norm = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    rgba = (cmap(np.clip(norm, 0, 1)) * 255).astype(np.uint8)
    tamano = tamano or (a.shape[1] * escala, a.shape[0] * escala)
    img = Image.fromarray(rgba).resize(tamano, Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def digito_grande(vista, lado=280):
    """Imagen 28x28 (0-255) -> imagen PIL grande coloreada con el mapa 'noche'."""
    rgba = (NOCHE(np.asarray(vista, float) / 255.0) * 255).astype(np.uint8)
    return Image.fromarray(rgba).resize((lado, lado), Image.NEAREST)


# ---------------------------------------------------------------------------
# GRILLAS DE MAPAS Y DE FILTROS
# ---------------------------------------------------------------------------
def _grilla(n, columnas, lado=1.05):
    filas = int(np.ceil(n / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(columnas * lado, filas * lado * 1.12))
    ejes = np.atleast_1d(ejes).ravel()
    for ax in ejes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.patch.set_alpha(0)
    return fig, ejes


def grilla_mapas(mapas, columnas=8, compartida=True, resaltar=None, titulo_fn=None):
    """
    mapas: arreglo (alto, ancho, canales) — la salida de una capa para UNA imagen.
    compartida=True  -> todos los mapas con la misma escala (se ve qué filtro responde MÁS)
    compartida=False -> cada mapa con su propia escala (se ve la FORMA aunque sea débil)
    resaltar: índice de un filtro para marcarlo con un borde.
    """
    c = mapas.shape[-1]
    fig, ejes = _grilla(c, columnas)
    vmax = float(mapas.max()) or 1.0
    for i in range(len(ejes)):
        ax = ejes[i]
        if i >= c:
            ax.axis("off"); continue
        m = mapas[..., i]
        ax.imshow(m, cmap=PETALO, vmin=0, vmax=vmax if compartida else (m.max() or 1.0),
                  interpolation="nearest")
        titulo = titulo_fn(i) if titulo_fn else f"f{i}"
        ax.set_title(titulo, fontsize=7, pad=2, color=CIRUELA)
        if resaltar is not None and i == resaltar:
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(ORQUIDEA); s.set_linewidth(2.5)
    fig.tight_layout(pad=0.3)
    return fig


def kernels_de_capa(pesos, canal_entrada=None):
    """
    pesos de Conv2D: (3, 3, canales_entrada, filtros).
    Cada filtro de conv_2 y conv_3 tiene una "rebanada" 3x3 POR CADA canal de entrada.
    canal_entrada=None -> promedio de todas las rebanadas (una idea general del filtro)
    canal_entrada=k    -> solo la rebanada que mira el canal k
    Devuelve (3, 3, filtros).
    """
    if pesos.shape[2] == 1:
        return pesos[:, :, 0, :]
    if canal_entrada is None:
        return pesos.mean(axis=2)
    return pesos[:, :, canal_entrada, :]


def grilla_kernels(k, columnas=8, resaltar=None):
    """k: (3, 3, filtros). Escala simétrica: el blanco es exactamente el cero."""
    f = k.shape[-1]
    fig, ejes = _grilla(f, columnas, lado=0.95)
    lim = float(np.abs(k).max()) or 1.0
    for i in range(len(ejes)):
        ax = ejes[i]
        if i >= f:
            ax.axis("off"); continue
        ax.imshow(k[..., i], cmap=PESOS, vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(f"f{i}", fontsize=7, pad=2, color=CIRUELA)
        es_resaltado = resaltar is not None and i == resaltar
        for s in ax.spines.values():          # marco fino para separar un filtro del otro
            s.set_visible(True)
            s.set_color(ORQUIDEA if es_resaltado else "#D9BFD0")
            s.set_linewidth(2.5 if es_resaltado else 0.8)
    fig.tight_layout(pad=0.3, w_pad=1.1, h_pad=0.6)
    return fig


def matriz_anotada(m, cmap=PESOS, simetrica=True, formato="{:+.2f}", tam=2.4, resaltar_centro=False):
    """Matriz pequeña (p. ej. 3x3) con el valor escrito en cada celda."""
    m = np.asarray(m, float)
    fig, ax = plt.subplots(figsize=(tam, tam))
    if simetrica:
        lim = float(np.abs(m).max()) or 1.0
        ax.imshow(m, cmap=cmap, vmin=-lim, vmax=lim)
    else:
        ax.imshow(m, cmap=cmap, vmin=0, vmax=float(m.max()) or 1.0)
    lo, hi = ax.images[0].get_clim()
    for (i, j), v in np.ndenumerate(m):
        t = (v - lo) / (hi - lo) if hi > lo else 0
        claro = (t > 0.78 or t < 0.12) if simetrica else t > 0.55
        ax.text(j, i, formato.format(v), ha="center", va="center", fontsize=9,
                color="white" if claro else CIRUELA)
    if resaltar_centro:
        ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False, ec=ORQUIDEA, lw=2))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.patch.set_alpha(0)
    fig.tight_layout(pad=0.1)
    return fig


def imagen_con_ventana(img, fila, col, cmap=NOCHE, tam=2.6, titulo=None):
    """Muestra una imagen 2D y dibuja un recuadro 3x3 centrado en (fila, col)."""
    fig, ax = plt.subplots(figsize=(tam, tam))
    ax.imshow(img, cmap=cmap, interpolation="nearest")
    ax.add_patch(plt.Rectangle((col - 1.5, fila - 1.5), 3, 3, fill=False, ec=ORQUIDEA, lw=2))
    ax.set_xticks([]); ax.set_yticks([])
    if titulo:
        ax.set_title(titulo, fontsize=9, color=CIRUELA)
    fig.patch.set_alpha(0)
    fig.tight_layout(pad=0.2)
    return fig


def mapa_simple(m, cmap=PETALO, tam=2.6, titulo=None, simetrica=False, marcar=None):
    fig, ax = plt.subplots(figsize=(tam, tam))
    if simetrica:
        lim = float(np.abs(m).max()) or 1.0
        ax.imshow(m, cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
    else:
        ax.imshow(m, cmap=cmap, interpolation="nearest")
    if marcar is not None:
        f, c = marcar
        ax.add_patch(plt.Rectangle((c - 0.5, f - 0.5), 1, 1, fill=False, ec=ORQUIDEA, lw=2))
    ax.set_xticks([]); ax.set_yticks([])
    if titulo:
        ax.set_title(titulo, fontsize=9, color=CIRUELA)
    fig.patch.set_alpha(0)
    fig.tight_layout(pad=0.2)
    return fig


def pesos_primera_capa_densa(W, columnas=8, resaltar=None):
    """W: (784, 64). Cada columna, vuelta a 28x28, muestra qué píxeles 'le gustan' a esa neurona."""
    n = W.shape[1]
    fig, ejes = _grilla(n, columnas, lado=1.0)
    lim = float(np.abs(W).max()) or 1.0
    for i in range(len(ejes)):
        ax = ejes[i]
        if i >= n:
            ax.axis("off"); continue
        ax.imshow(W[:, i].reshape(28, 28), cmap=PESOS, vmin=-lim, vmax=lim)
        ax.set_title(f"n{i}", fontsize=7, pad=2)
        if resaltar is not None and i == resaltar:
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(ORQUIDEA); s.set_linewidth(2.5)
    fig.tight_layout(pad=0.3)
    return fig


def matriz_confusion(m, titulo):
    m = np.asarray(m)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    # Escala logarítmica suave: así los pocos errores también se ven
    ax.imshow(np.log1p(m), cmap=PETALO)
    for (i, j), v in np.ndenumerate(m):
        if v:
            ax.text(j, i, str(v), ha="center", va="center", fontsize=6.5,
                    color="white" if i == j else CIRUELA)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicción"); ax.set_ylabel("dígito real")
    ax.set_title(titulo, fontsize=11, color=CIRUELA)
    ax.tick_params(length=0, labelsize=8)
    fig.patch.set_alpha(0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CÁLCULO: convolución "a mano" (igual a la de Keras)
# ---------------------------------------------------------------------------
def convolucion_same(img, kernel):
    """
    Correlación 2D con relleno de ceros ("same"), exactamente lo que hace Conv2D:
        salida[i, j] = suma_{u,v} kernel[u, v] * img[i+u-1, j+v-1]
    (Keras NO voltea el kernel; técnicamente es una correlación cruzada.)
    """
    img = np.asarray(img, float)
    k = np.asarray(kernel, float)
    p = k.shape[0] // 2
    relleno = np.pad(img, p)
    ventanas = np.lib.stride_tricks.sliding_window_view(relleno, k.shape)
    return np.einsum("ijuv,uv->ij", ventanas, k)


def ventana_3x3(img, fila, col):
    """La vecindad 3x3 alrededor de (fila, col), con ceros fuera del borde."""
    relleno = np.pad(np.asarray(img, float), 1)
    return relleno[fila:fila + 3, col:col + 3]


# ---------------------------------------------------------------------------
# CURVAS DE ENTRENAMIENTO (Altair, con la misma paleta)
# ---------------------------------------------------------------------------
def _estilo_altair(grafico):
    return (grafico.configure_view(strokeWidth=0)
            .configure_axis(labelColor=CIRUELA, titleColor=CIRUELA, gridColor="#EFE2EA",
                            domainColor="#E9D6E2")
            .configure_legend(labelColor=CIRUELA, titleColor=CIRUELA))


def curvas_perdida(historial, color=ORQUIDEA, altura=260, parada=None, escala_log=False,
                   titulo_y="pérdida"):
    """
    Curvas de entrenamiento (línea continua) y validación (punteada) de una red.
    historial: diccionario de Keras con "loss" y "val_loss".
    parada: época (empezando en 1) donde se detuvo / quedó el mejor modelo; se marca con una línea.
    """
    import altair as alt
    import pandas as pd
    n = len(historial["loss"])
    df = pd.DataFrame({
        "época": list(range(1, n + 1)) * 2,
        "conjunto": ["entrenamiento"] * n + ["validación"] * n,
        titulo_y: list(historial["loss"]) + list(historial["val_loss"]),
    })
    escala = alt.Scale(type="log") if escala_log else alt.Scale(zero=False)
    capa = (alt.Chart(df).mark_line(strokeWidth=2.2, color=color)
            .encode(x=alt.X("época:Q", axis=alt.Axis(tickMinStep=1, labelAngle=0)),
                    y=alt.Y(f"{titulo_y}:Q", scale=escala),
                    strokeDash=alt.StrokeDash("conjunto:N",
                                              scale=alt.Scale(domain=["entrenamiento", "validación"],
                                                              range=[[1, 0], [5, 4]]),
                                              legend=alt.Legend(orient="top", title=None)),
                    tooltip=["conjunto", "época", alt.Tooltip(f"{titulo_y}:Q", format=".5f")]))
    if parada is not None:
        regla = (alt.Chart(pd.DataFrame({"época": [parada]}))
                 .mark_rule(color=SALVIA, strokeWidth=2, strokeDash=[2, 3])
                 .encode(x="época:Q"))
        capa = capa + regla
    return _estilo_altair(capa.properties(height=altura))
