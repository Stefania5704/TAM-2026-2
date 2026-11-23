"""
Atelier neuronal — cómo ven un dígito una red DENSA y una red CONVOLUCIONAL.

Cómo ejecutarla (dentro de esta carpeta, con el entorno virtual activo):
    python entrenar_modelos.py      # solo la primera vez (crea la carpeta modelos/)
    streamlit run app.py            # abre http://localhost:8501

RECORDATORIO DE STREAMLIT:
El archivo se ejecuta COMPLETO, de arriba a abajo, cada vez que mueves un control.
Por eso lo pesado (cargar modelos, calcular mapas) se guarda en caché con
@st.cache_resource / @st.cache_data: se calcula una vez y se reutiliza.
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import visualizaciones as viz
from preprocesamiento import preparar_imagen

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELOS = os.path.join(AQUI, "modelos")

st.set_page_config(page_title="Atelier neuronal · MNIST", page_icon="🌸", layout="wide")

# Inyectar nuestro CSS (tipografías, colores, tarjetas)
with open(os.path.join(AQUI, "estilo.css"), encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ===========================================================================
# 1. CARGAR MODELOS Y RESULTADOS (una sola vez)
# ===========================================================================
archivos_necesarios = ["modelo_densa.keras", "modelo_cnn.keras", "metricas.json",
                       "resultados_prueba.npz"]
faltan = [a for a in archivos_necesarios if not os.path.exists(os.path.join(MODELOS, a))]
if faltan:
    st.title("Falta entrenar las redes")
    st.write("No encontré estos archivos en la carpeta `modelos/`: " + ", ".join(faltan))
    st.write("Entrénalas una vez con este comando (tarda unos minutos) y recarga la página:")
    st.code("python entrenar_modelos.py", language="bash")
    st.stop()


@st.cache_resource(show_spinner="Cargando las redes entrenadas…")
def cargar_todo():
    import tensorflow as tf                       # se importa aquí para que la app abra rápido
    densa = tf.keras.models.load_model(os.path.join(MODELOS, "modelo_densa.keras"))
    cnn = tf.keras.models.load_model(os.path.join(MODELOS, "modelo_cnn.keras"))

    # "Abrir" la CNN: un modelo nuevo que, con la misma entrada, devuelve la salida
    # de CADA capa interesante. Es como poner sensores dentro de la red.
    nombres_cnn = ["conv_1", "pool_1", "conv_2", "pool_2", "conv_3", "salida"]
    espia_cnn = tf.keras.Model(cnn.input, [cnn.get_layer(n).output for n in nombres_cnn])
    nombres_densa = ["densa_1", "densa_2", "salida"]
    espia_densa = tf.keras.Model(densa.input, [densa.get_layer(n).output for n in nombres_densa])

    pesos = {n: [w.copy() for w in cnn.get_layer(n).get_weights()] for n in ("conv_1", "conv_2", "conv_3")}
    pesos["densa_1"] = [w.copy() for w in densa.get_layer("densa_1").get_weights()]
    return espia_cnn, espia_densa, nombres_cnn, nombres_densa, pesos


@st.cache_resource
def cargar_resultados():
    with open(os.path.join(MODELOS, "metricas.json"), encoding="utf-8") as f:
        metricas = json.load(f)
    d = np.load(os.path.join(MODELOS, "resultados_prueba.npz"))
    return metricas, {k: d[k] for k in d.files}


espia_cnn, espia_densa, NOMBRES_CNN, NOMBRES_DENSA, PESOS = cargar_todo()
METRICAS, PRUEBA = cargar_resultados()


@st.cache_data(show_spinner=False, max_entries=64)
def analizar(vista_bytes):
    """Pasa el dígito por las dos redes y guarda la salida de cada capa."""
    vista = np.frombuffer(vista_bytes, dtype=np.uint8).reshape(28, 28)
    x = vista.astype("float32") / 255.0
    salidas_cnn = espia_cnn(x.reshape(1, 28, 28, 1), training=False)
    salidas_densa = espia_densa(x.reshape(1, 784), training=False)
    cnn = {n: np.asarray(s)[0] for n, s in zip(NOMBRES_CNN, salidas_cnn)}
    densa = {n: np.asarray(s)[0] for n, s in zip(NOMBRES_DENSA, salidas_densa)}
    return cnn, densa


def miles(n):
    return f"{n:,}".replace(",", ".")


def mostrar(fig):
    """Dibuja una figura de matplotlib en la página y luego la cierra (libera memoria)."""
    st.pyplot(fig)
    plt.close(fig)


def pct(x, dec=1):
    return f"{x * 100:.{dec}f} %".replace(".", ",")


# ===========================================================================
# 2. BARRA LATERAL: ELEGIR EL DÍGITO
# ===========================================================================
with st.sidebar:
    st.markdown("## Elige un dígito")
    fuente = st.radio("¿De dónde sale la imagen?",
                      ["Ejemplos de MNIST", "Carpeta del proyecto", "Subir imagen", "Cámara"],
                      label_visibility="collapsed")

    vista, etiqueta_real, nombre_fuente = None, None, ""

    if fuente == "Ejemplos de MNIST":
        st.caption("Imágenes del conjunto de prueba: las redes nunca las vieron al entrenar.")
        quiero = st.selectbox("Dígito", ["cualquiera"] + list(range(10)))
        if "semilla_ejemplo" not in st.session_state:
            st.session_state.semilla_ejemplo = 7
        if st.button("Mostrar otro ejemplo", width="stretch"):
            st.session_state.semilla_ejemplo += 1
        candidatos = np.arange(len(PRUEBA["y"])) if quiero == "cualquiera" \
            else np.flatnonzero(PRUEBA["y"] == quiero)
        rng = np.random.default_rng(st.session_state.semilla_ejemplo)
        idx = int(rng.choice(candidatos))
        vista = PRUEBA["x"][idx]
        etiqueta_real = int(PRUEBA["y"][idx])
        nombre_fuente = f"MNIST prueba n.º {idx}"

    elif fuente == "Carpeta del proyecto":
        archivos = []
        for carpeta in ("mis_imagenes", "ejemplos"):
            ruta = os.path.join(AQUI, carpeta)
            if os.path.isdir(ruta):
                archivos += [os.path.join(carpeta, a) for a in sorted(os.listdir(ruta))
                             if a.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]
        if not archivos:
            st.info("Guarda fotos de tus dígitos en la carpeta `mis_imagenes/`.")
        else:
            elegido = st.selectbox("Archivo", archivos)
            vista = preparar_imagen(Image.open(os.path.join(AQUI, elegido)))[1]
            nombre_fuente = elegido

    elif fuente == "Subir imagen":
        sub = st.file_uploader("Foto de un dígito", type=["png", "jpg", "jpeg", "bmp", "webp"])
        if sub:
            vista = preparar_imagen(Image.open(sub))[1]
            nombre_fuente = sub.name

    else:
        foto = st.camera_input("Toma una foto de un dígito")
        if foto:
            vista = preparar_imagen(Image.open(foto))[1]
            nombre_fuente = "cámara"

    if vista is not None and vista.max() > 0:
        st.image(viz.digito_grande(vista, 240), caption="Lo que reciben las redes (28 × 28)",
                 width="stretch")
        if etiqueta_real is not None:
            st.markdown(f"Etiqueta real: **{etiqueta_real}**")
        if fuente != "Ejemplos de MNIST":
            st.caption("Tu imagen pasó por `preprocesamiento.py`: se quitó el fondo, se recortó, "
                       "se escaló a 20 px y se centró por su centro de masa, como en MNIST.")

if vista is None or vista.max() == 0:
    st.markdown("<div class='hero'><div><h1>Atelier neuronal</h1>"
                "<p class='bajada'>Elige o sube un dígito en la barra lateral para empezar. "
                "Si tu foto no tiene un trazo visible, prueba con tinta más oscura y fondo liso.</p>"
                "</div></div>", unsafe_allow_html=True)
    st.stop()

vista = np.ascontiguousarray(vista, dtype=np.uint8)
CNN, DENSA = analizar(vista.tobytes())
x01 = vista.astype(float) / 255.0
p_cnn, p_densa = CNN["salida"], DENSA["salida"]
d_cnn, d_densa = int(p_cnn.argmax()), int(p_densa.argmax())


# ===========================================================================
# 3. PORTADA: el dígito y las 16 respuestas de la primera capa
# ===========================================================================
m1 = CNN["conv_1"]
lim1 = float(m1.max()) or 1.0
tira = "".join(f"<img src='data:image/png;base64,{viz.arreglo_a_png_b64(m1[..., i], viz.PETALO, 4, 0, lim1)}' "
               f"alt='mapa del filtro {i}'>" for i in range(m1.shape[-1]))
entrada_b64 = viz.arreglo_a_png_b64(vista, viz.NOCHE, 6, 0, 255)
st.markdown(f"""
<div class="hero">
  <div>
    <h1>Cómo mira una red neuronal un número escrito a mano</h1>
    <p class="bajada">Dos redes entrenadas con MNIST reciben el mismo dígito. La densa lo lee como
    una fila de 784 píxeles sueltos; la convolucional lo recorre con filtros de 3 × 3 y construye,
    capa por capa, una descripción cada vez más abstracta del trazo.</p>
    <div class="hero-datos">
      <div><b>{pct(METRICAS['cnn']['exactitud_prueba'])}</b>exactitud de la convolucional</div>
      <div><b>{pct(METRICAS['densa']['exactitud_prueba'])}</b>exactitud de la densa</div>
      <div><b>{miles(METRICAS['cnn']['parametros'])} / {miles(METRICAS['densa']['parametros'])}</b>parámetros (CNN / densa)</div>
    </div>
  </div>
  <div class="hero-visual">
    <figure class="entrada"><img src="data:image/png;base64,{entrada_b64}" alt="dígito de entrada">
      <figcaption>Entrada, 28 × 28</figcaption></figure>
    <div class="tira-caja"><div class="tira">{tira}</div>
      <figcaption>Las 16 salidas de la primera capa convolucional para este dígito</figcaption></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_pred, tab_densa, tab_cnn, tab_lab, tab_comp = st.tabs(
    ["Predicción", "Red densa", "Red convolucional, capa por capa", "Laboratorio de filtros", "Comparación"])


def grafico_probabilidades(p, color):
    df = pd.DataFrame({"dígito": [str(i) for i in range(10)], "probabilidad": p})
    df["elegido"] = df["probabilidad"] == df["probabilidad"].max()
    return (alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(x=alt.X("dígito:N", axis=alt.Axis(labelAngle=0, title=None)),
                    y=alt.Y("probabilidad:Q", scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(format="%", title=None, tickCount=5)),
                    color=alt.condition("datum.elegido", alt.value(color), alt.value(viz.CUARZO)),
                    tooltip=["dígito", alt.Tooltip("probabilidad:Q", format=".2%")])
            .properties(height=200)
            .configure_view(strokeWidth=0)
            .configure_axis(labelFont="Manrope", labelColor=viz.CIRUELA, gridColor="#EFE2EA",
                            domainColor="#E9D6E2"))


# ===========================================================================
# 4. PESTAÑA: PREDICCIÓN
# ===========================================================================
with tab_pred:
    c1, c2 = st.columns(2, gap="large")
    for col, nombre, clase, p, d, color in [
            (c1, "Red convolucional", "cnn", p_cnn, d_cnn, viz.ORQUIDEA),
            (c2, "Red densa", "densa", p_densa, d_densa, viz.SALVIA)]:
        with col:
            acierto = ""
            if etiqueta_real is not None:
                acierto = " — acierta" if d == etiqueta_real else f" — se equivoca (era {etiqueta_real})"
            st.markdown(f"<div class='prediccion {clase}'><div class='modelo'>{nombre}</div>"
                        f"<div class='numero'>{d}</div>"
                        f"<div>confianza <span class='confianza'>{pct(float(p[d]))}</span>{acierto}</div>"
                        f"</div>", unsafe_allow_html=True)
            st.altair_chart(grafico_probabilidades(p, color), width="stretch")
    if d_cnn == d_densa:
        st.markdown(f"<div class='nota'>Las dos redes coinciden en que es un <b>{d_cnn}</b>. "
                    "Fíjate en la confianza: la convolucional suele estar más segura porque reconoce "
                    "trazos locales (curvas, esquinas) y no solo qué píxeles están encendidos.</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='nota'>Las redes no están de acuerdo: la convolucional dice <b>{d_cnn}</b> "
                    f"y la densa <b>{d_densa}</b>. Revisa la pestaña de la red convolucional para ver qué "
                    "partes del trazo activaron cada filtro.</div>", unsafe_allow_html=True)
    with st.expander("¿De dónde salen estas probabilidades? (softmax)"):
        st.write("La última capa produce 10 números cualesquiera $z_0,\\dots,z_9$ (uno por dígito). "
                 "Softmax los convierte en probabilidades positivas que suman 1:")
        st.latex(r"P(\text{dígito}=k)=\frac{e^{z_k}}{\sum_{j=0}^{9} e^{z_j}}")
        st.write("La predicción es el dígito con la probabilidad más alta. Ambas redes se entrenaron "
                 "minimizando la entropía cruzada $-\\log P(\\text{dígito correcto})$ con el optimizador Adam.")


# ===========================================================================
# 5. PESTAÑA: RED DENSA
# ===========================================================================
with tab_densa:
    st.markdown("## La red densa lee píxeles, no formas")
    st.write("Antes de entrar, la imagen de 28 × 28 se desenrolla en una sola fila de 784 números. "
             "Desde ese momento la red ya no sabe qué píxeles eran vecinos: cada neurona de la primera "
             "capa mira los 784 a la vez y les da un peso a cada uno.")
    vec_b64 = viz.arreglo_a_png_b64(x01.reshape(1, 784), viz.NOCHE, vmin=0, vmax=1, tamano=(1568, 40))
    st.markdown(f"<div class='vector'><img src='data:image/png;base64,{vec_b64}' alt='vector de 784 píxeles'>"
                "<small>Tu dígito como vector: fila 0 a la izquierda, fila 27 a la derecha.</small></div>",
                unsafe_allow_html=True)
    st.markdown("<div class='flujo'>"
                "<div class='paso'><b>entrada</b>784</div>"
                "<div class='paso'><b>densa_1</b>64, ReLU</div>"
                "<div class='paso'><b>densa_2</b>64, ReLU</div>"
                "<div class='paso'><b>salida</b>10, softmax</div></div>", unsafe_allow_html=True)

    st.markdown("### Qué se activa dentro")
    a1, a2 = DENSA["densa_1"], DENSA["densa_2"]
    c1, c2, c3 = st.columns(3)
    with c1:
        mostrar(viz.mapa_simple(a1.reshape(8, 8), titulo="densa_1 (64 neuronas en 8 × 8)", tam=2.8))
    with c2:
        mostrar(viz.mapa_simple(a2.reshape(8, 8), titulo="densa_2 (64 neuronas en 8 × 8)", tam=2.8))
    with c3:
        st.write("Cada cuadrito es una neurona; más oscuro significa que se activó más. "
                 f"En densa_1 se encendieron {int((a1 > 0).sum())} de 64 neuronas; "
                 f"las demás quedaron en 0 por la ReLU.")
        st.latex(r"h = \mathrm{ReLU}(W^\top x + b)")
        st.latex(r"\mathrm{ReLU}(z)=\max(0,\,z)")

    st.markdown("### Qué mira cada neurona de la primera capa")
    st.write("Los 784 pesos de una neurona, reorganizados como imagen de 28 × 28, muestran a qué "
             "píxeles responde: en rosa los que la encienden y en verde los que la apagan. "
             "Se parecen más a manchas que a trazos, porque la red no tiene noción de vecindad.")
    W, b = PESOS["densa_1"]
    neurona = st.select_slider("Neurona", options=list(range(64)), value=int(a1.argmax()),
                               help="Por defecto se elige la neurona que más se activó con tu dígito.")
    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        mostrar(viz.pesos_primera_capa_densa(W, resaltar=neurona))
    with c2:
        contrib = W[:, neurona].reshape(28, 28) * x01
        cc1, cc2 = st.columns(2)
        with cc1:
            mostrar(viz.mapa_simple(W[:, neurona].reshape(28, 28), viz.PESOS, 2.2,
                                      f"pesos de n{neurona}", simetrica=True))
        with cc2:
            mostrar(viz.mapa_simple(contrib, viz.PESOS, 2.2, "peso × píxel", simetrica=True))
        z = float(contrib.sum() + b[neurona])
        st.latex(rf"z_{{{neurona}}}=\sum_{{i=1}}^{{784}} w_i x_i + b = {contrib.sum():.3f} {b[neurona]:+.3f} = {z:.3f}")
        st.latex(rf"\mathrm{{ReLU}}(z_{{{neurona}}}) = {max(z, 0):.3f}")
        st.markdown(f"<span class='verificacion'>Valor que calculó Keras: {a1[neurona]:.3f}</span>",
                    unsafe_allow_html=True)


# ===========================================================================
# 6. PESTAÑA: RED CONVOLUCIONAL, CAPA POR CAPA
# ===========================================================================
CAPAS = {
    "conv_1": dict(n=1, entrada="la imagen", pool="pool_1", previa=None),
    "conv_2": dict(n=2, entrada="pool_1", pool="pool_2", previa="pool_1"),
    "conv_3": dict(n=3, entrada="pool_2", pool=None, previa="pool_2"),
}

with tab_cnn:
    st.markdown("## La red convolucional recorre la imagen con filtros")
    st.write("Un filtro es una matriz pequeña de 3 × 3 pesos que se desliza por toda la imagen. "
             "En cada posición multiplica sus 9 pesos por los 9 píxeles que tiene debajo, suma, "
             "agrega un sesgo y aplica ReLU. El resultado es un **mapa de características**: "
             "una imagen que dice *dónde* apareció el patrón que ese filtro sabe detectar.")

    capa = st.radio("Capa", list(CAPAS), horizontal=True,
                    format_func=lambda c: f"Capa {CAPAS[c]['n']}: {c} ({PESOS[c][0].shape[-1]} filtros)")
    info = CAPAS[capa]
    K, B = PESOS[capa]                         # K: (3, 3, canales_entrada, filtros)
    salida = CNN[capa]                         # (alto, ancho, filtros)
    c_in, n_f = K.shape[2], K.shape[3]

    pasos = [("entrada", "28 × 28 × 1", ""), ("conv_1", "28 × 28 × 16", "conv"), ("pool_1", "14 × 14 × 16", ""),
             ("conv_2", "14 × 14 × 32", "conv"), ("pool_2", "7 × 7 × 32", ""), ("conv_3", "7 × 7 × 64", "conv"),
             ("aplanar", "3136", ""), ("salida", "10, softmax", "")]
    st.markdown("<div class='flujo'>" + "".join(
        f"<div class='paso {cl}{' activa' if n == capa else ''}'><b>{n}</b>{forma}</div>"
        for n, forma, cl in pasos) + "</div>", unsafe_allow_html=True)

    # ---- Los filtros (pesos aprendidos) ----
    st.markdown(f"### Los {n_f} filtros que aprendió {capa}")
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c2:
        if c_in == 1:
            st.write("Cada filtro mira un solo canal (la imagen en grises), así que es exactamente "
                     "una matriz 3 × 3. Rosa: peso positivo. Verde: peso negativo. Blanco: cero.")
            canal_ver = None
        else:
            st.write(f"Aquí la entrada ya no es una imagen sino {c_in} mapas apilados. Cada filtro tiene "
                     f"una rebanada 3 × 3 para cada uno: en total 3 × 3 × {c_in} = {9 * c_in} pesos. "
                     "Puedes ver el promedio de las rebanadas o una sola.")
            opcion = st.selectbox("Rebanada a mostrar", ["promedio de todos los canales"] +
                                  [f"canal {i} de {info['entrada']}" for i in range(c_in)])
            canal_ver = None if opcion.startswith("promedio") else int(opcion.split()[1])
        st.latex(rf"\underbrace{{3\times3\times{c_in}}}_{{\text{{pesos}}}}\times{n_f}+"
                 rf"\underbrace{{{n_f}}}_{{\text{{sesgos}}}}={9 * c_in * n_f + n_f:,}".replace(",", r"\,")
                 + r"\ \text{parámetros}")
    with c1:
        mostrar(viz.grilla_kernels(viz.kernels_de_capa(K, canal_ver), columnas=8 if n_f <= 32 else 16))

    # ---- Salida: mapas de características ----
    st.markdown(f"### La salida de {capa} para tu dígito")
    compartida = st.toggle("Misma escala de color para todos los mapas", value=True,
                           help="Activado: se ve qué filtros responden con más fuerza. "
                                "Desactivado: cada mapa usa su propia escala y se ve la forma aunque sea débil.")
    mostrar(viz.grilla_mapas(salida, columnas=8 if n_f <= 32 else 16, compartida=compartida))
    alto = salida.shape[0]
    st.markdown(f"<div class='nota'>{n_f} mapas de {alto} × {alto}. "
                + {1: "La primera capa detecta cosas simples: bordes horizontales, verticales, diagonales, "
                      "el lado izquierdo o derecho del trazo.",
                   2: "La segunda capa combina esos bordes: aparecen esquinas, curvas y finales de trazo.",
                   3: "La tercera capa trabaja en 7 × 7: cada píxel 've' una zona grande del dígito original, "
                      "así que sus mapas son más abstractos (partes del número, no bordes)."}[info["n"]]
                + "</div>", unsafe_allow_html=True)

    if info["pool"]:
        with st.expander(f"Ver la salida de {info['pool']} (MaxPooling 2 × 2)"):
            st.write("De cada bloque de 2 × 2 se conserva el valor más alto. El mapa queda a la mitad "
                     "y la red se vuelve tolerante a pequeños desplazamientos del trazo.")
            st.latex(r"\text{pool}_{i,j}=\max\left(y_{2i,2j},\,y_{2i,2j+1},\,y_{2i+1,2j},\,y_{2i+1,2j+1}\right)")
            mostrar(viz.grilla_mapas(CNN[info["pool"]], columnas=8 if n_f <= 32 else 16,
                                       compartida=compartida))

    # Activación promedio por filtro
    medias = salida.mean(axis=(0, 1))
    df_m = pd.DataFrame({"filtro": [f"f{i}" for i in range(n_f)], "activación media": medias,
                         "orden": range(n_f)})

    # ---- Zoom a un filtro ----
    st.markdown("### Un filtro de cerca")
    filtro = st.select_slider("Filtro", options=list(range(n_f)), value=int(medias.argmax()),
                              format_func=lambda i: f"f{i}",
                              help="Por defecto se elige el filtro con mayor activación media.")
    barras = (alt.Chart(df_m).mark_bar()
              .encode(x=alt.X("filtro:N", sort=alt.SortField("orden"), axis=alt.Axis(labelAngle=-90, title=None)),
                      y=alt.Y("activación media:Q", axis=alt.Axis(title="activación media")),
                      color=alt.condition(alt.datum.orden == filtro, alt.value(viz.ORQUIDEA), alt.value(viz.CUARZO)),
                      tooltip=["filtro", alt.Tooltip("activación media:Q", format=".3f")])
              .properties(height=150).configure_view(strokeWidth=0)
              .configure_axis(labelColor=viz.CIRUELA, titleColor=viz.CIRUELA, gridColor="#EFE2EA"))
    st.altair_chart(barras, width="stretch")

    entrada_capa = x01[..., None] if info["previa"] is None else CNN[info["previa"]]

    # Verificación: recalcular TODA la capa con numpy y compararla con Keras
    manual = sum(viz.convolucion_same(entrada_capa[..., c], K[:, :, c, filtro]) for c in range(c_in)) + B[filtro]
    manual = np.maximum(manual, 0)
    error = float(np.abs(manual - salida[..., filtro]).max())

    if c_in == 1:
        fila_max, col_max = np.unravel_index(salida[..., filtro].argmax(), salida.shape[:2])
        cs1, cs2 = st.columns(2)
        fila = cs1.slider("Fila del píxel", 0, alto - 1, int(fila_max))
        col = cs2.slider("Columna del píxel", 0, alto - 1, int(col_max))
        ventana = viz.ventana_3x3(x01, fila, col)
        kern = K[:, :, 0, filtro]
        prod = ventana * kern
        g1, g2, g3, g4, g5 = st.columns(5)
        with g1:
            mostrar(viz.imagen_con_ventana(x01, fila, col, titulo="entrada", tam=2.2))
        with g2:
            mostrar(viz.matriz_anotada(ventana, viz.PETALO, simetrica=False, formato="{:.2f}", tam=2.2))
            st.caption("ventana 3 × 3 de píxeles")
        with g3:
            mostrar(viz.matriz_anotada(kern, tam=2.2))
            st.caption(f"pesos del filtro f{filtro}")
        with g4:
            mostrar(viz.matriz_anotada(prod, tam=2.2))
            st.caption("producto elemento a elemento")
        with g5:
            mostrar(viz.mapa_simple(salida[..., filtro], tam=2.2, titulo=f"mapa de f{filtro}",
                                      marcar=(fila, col)))
        z = float(prod.sum() + B[filtro])
        st.latex(rf"y_{{{fila},{col}}}=\mathrm{{ReLU}}\Big(\sum_{{u,v}} K_{{u,v}}\,x_{{{fila}+u,\,{col}+v}} + b\Big)"
                 rf"=\mathrm{{ReLU}}({prod.sum():.3f} {B[filtro]:+.3f})=\mathrm{{ReLU}}({z:.3f})={max(z, 0):.3f}")
        st.markdown(f"<span class='verificacion'>Keras calculó {salida[fila, col, filtro]:.3f} en ese píxel. "
                    f"Recalculando el mapa completo con numpy, la diferencia máxima es {error:.1e}.</span>",
                    unsafe_allow_html=True)
    else:
        aporte = np.array([np.abs(viz.convolucion_same(entrada_capa[..., c], K[:, :, c, filtro])).sum()
                           for c in range(c_in)])
        canal = st.select_slider("Canal de entrada", options=list(range(c_in)), value=int(aporte.argmax()),
                                 help="Por defecto, el canal que más aporta a este filtro.")
        parcial = viz.convolucion_same(entrada_capa[..., canal], K[:, :, canal, filtro])
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            mostrar(viz.mapa_simple(entrada_capa[..., canal], tam=2.4, titulo=f"canal {canal} de {info['entrada']}"))
        with g2:
            mostrar(viz.matriz_anotada(K[:, :, canal, filtro], tam=2.4))
            st.caption(f"rebanada {canal} del filtro f{filtro}")
        with g3:
            mostrar(viz.mapa_simple(parcial, viz.PESOS, 2.4, "aporte de este canal", simetrica=True))
        with g4:
            mostrar(viz.mapa_simple(salida[..., filtro], tam=2.4, titulo=f"salida final de f{filtro}"))
        st.latex(rf"y^{{(f{filtro})}}=\mathrm{{ReLU}}\Big(\sum_{{c=0}}^{{{c_in - 1}}} K^{{(c)}} \star x^{{(c)}} + b\Big)")
        st.write(f"El mapa final suma los aportes de los {c_in} canales (no solo el que ves), "
                 "agrega el sesgo y aplica ReLU. Así una neurona de esta capa combina varios "
                 "patrones de la capa anterior.")
        st.markdown(f"<span class='verificacion'>Recalculando la suma de los {c_in} canales con numpy, "
                    f"la diferencia máxima con Keras es {error:.1e}.</span>", unsafe_allow_html=True)


# ===========================================================================
# 7. PESTAÑA: LABORATORIO DE FILTROS
# ===========================================================================
CLASICOS = {
    "Borde vertical (Sobel x)": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    "Borde horizontal (Sobel y)": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    "Diagonal": [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
    "Laplaciano (contornos)": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    "Desenfoque (promedio)": [[1 / 9] * 3] * 3,
    "Realce (nitidez)": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    "Identidad": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
}

with tab_lab:
    st.markdown("## Diseña tu propio filtro")
    st.write("Antes de las redes, los filtros se diseñaban a mano. Elige uno clásico o escribe tus "
             "propios 9 números y observa qué resalta en tu dígito. Abajo verás cuál de los 16 filtros "
             "que aprendió conv_1 se parece más al tuyo.")
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        base = st.selectbox("Punto de partida", list(CLASICOS))
        tabla = pd.DataFrame(np.array(CLASICOS[base], float), columns=["izq.", "centro", "der."],
                             index=["arriba", "medio", "abajo"])
        editado = st.data_editor(tabla, key=f"editor_{base}", width="stretch",
                                 column_config={c: st.column_config.NumberColumn(format="%.2f", step=0.25)
                                                for c in tabla.columns})
        kern = editado.to_numpy(dtype=float)
        st.caption("Haz doble clic en una celda para cambiar su valor.")
    conv = viz.convolucion_same(x01, kern)
    with c2:
        g1, g2, g3 = st.columns(3)
        with g1:
            mostrar(viz.matriz_anotada(kern, tam=2.4))
            st.caption("tu filtro")
        with g2:
            mostrar(viz.mapa_simple(conv, viz.PESOS, 2.4, "convolución", simetrica=True))
        with g3:
            mostrar(viz.mapa_simple(np.maximum(conv, 0), viz.PETALO, 2.4, "después de ReLU"))
        st.markdown("<div class='nota'>En la convolución, rosa es respuesta positiva y verde negativa. "
                    "La ReLU borra todo lo verde: por eso un borde vertical de Sobel solo ve "
                    "<i>un</i> lado del trazo. Las redes aprenden pares de filtros con signos opuestos "
                    "para ver ambos lados.</div>", unsafe_allow_html=True)

    st.markdown("### Su pariente más cercano en conv_1")
    K1, B1 = PESOS["conv_1"]
    aprendidos = K1[:, :, 0, :].reshape(9, -1)                    # 9 pesos x 16 filtros
    v = kern.ravel()
    if np.linalg.norm(v) == 0:
        st.info("Tu filtro tiene todos los pesos en cero; cambia algún valor para compararlo.")
    else:
        sim = (v @ aprendidos) / (np.linalg.norm(v) * np.linalg.norm(aprendidos, axis=0) + 1e-12)
        st.latex(r"\text{similitud}(a,b)=\frac{a\cdot b}{\lVert a\rVert\,\lVert b\rVert}\in[-1,1]")
        top = np.argsort(-sim)[:3]
        cols = st.columns(3)
        for col, i in zip(cols, top):
            with col:
                st.markdown(f"**f{i}**, similitud {sim[i]:+.2f}")
                cc1, cc2 = st.columns(2)
                with cc1:
                    mostrar(viz.matriz_anotada(K1[:, :, 0, i], tam=1.9, formato="{:+.1f}"))
                with cc2:
                    mostrar(viz.mapa_simple(CNN["conv_1"][..., i], tam=1.9))


# ===========================================================================
# 8. PESTAÑA: COMPARACIÓN
# ===========================================================================
with tab_comp:
    st.markdown("## ¿Cuál red funciona mejor?")
    md, mc, cfg = METRICAS["densa"], METRICAS["cnn"], METRICAS["config"]
    n_prueba = cfg["n_prueba"]
    err_d = round((1 - md["exactitud_prueba"]) * n_prueba)
    err_c = round((1 - mc["exactitud_prueba"]) * n_prueba)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Exactitud CNN", pct(mc["exactitud_prueba"], 2))
    k2.metric("Exactitud densa", pct(md["exactitud_prueba"], 2))
    k3.metric("Errores CNN / densa", f"{err_c} / {err_d}", help=f"De {miles(n_prueba)} imágenes de prueba")
    k4.metric("Parámetros CNN / densa", f"{mc['parametros'] / 1000:.0f}k / {md['parametros'] / 1000:.0f}k",
              help=f"{miles(mc['parametros'])} y {miles(md['parametros'])} parámetros")
    st.markdown(f"<div class='nota'>Las dos redes tienen casi el mismo número de parámetros y se "
                f"entrenaron igual ({cfg['epocas']} épocas, lotes de {cfg['lote']}, Adam). "
                f"La diferencia viene de la arquitectura: la convolucional reutiliza los mismos 9 pesos "
                f"en todas las posiciones de la imagen, así que aprende patrones locales que sirven "
                f"en cualquier lugar del dígito. Con {err_c} errores frente a {err_d}, "
                f"comete {pct(1 - err_c / max(err_d, 1), 0)} menos errores.</div>", unsafe_allow_html=True)

    st.markdown("### Cómo aprendieron, época por época")
    filas = []
    for nombre, m in [("convolucional", mc), ("densa", md)]:
        h = m["historial"]
        for e in range(len(h["accuracy"])):
            filas.append({"época": e + 1, "red": nombre, "conjunto": "entrenamiento", "exactitud": h["accuracy"][e],
                          "pérdida": h["loss"][e]})
            filas.append({"época": e + 1, "red": nombre, "conjunto": "validación", "exactitud": h["val_accuracy"][e],
                          "pérdida": h["val_loss"][e]})
    df_h = pd.DataFrame(filas)
    colores = alt.Scale(domain=["convolucional", "densa"], range=[viz.ORQUIDEA, viz.SALVIA])
    trazos = alt.Scale(domain=["entrenamiento", "validación"], range=[[1, 0], [5, 4]])

    def curva(y, formato):
        return (alt.Chart(df_h).mark_line(point=True, strokeWidth=2.2)
                .encode(x=alt.X("época:O", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format=formato)),
                        color=alt.Color("red:N", scale=colores, legend=alt.Legend(orient="top", title=None)),
                        strokeDash=alt.StrokeDash("conjunto:N", scale=trazos, legend=None),
                        tooltip=["red", "conjunto", "época", alt.Tooltip(f"{y}:Q", format=".4f")])
                .properties(height=260).configure_view(strokeWidth=0)
                .configure_axis(labelColor=viz.CIRUELA, titleColor=viz.CIRUELA, gridColor="#EFE2EA"))
    c1, c2 = st.columns(2, gap="large")
    c1.altair_chart(curva("exactitud", "%"), width="stretch")
    c2.altair_chart(curva("pérdida", ".2f"), width="stretch")
    st.caption("Línea continua: datos de entrenamiento. Línea punteada: validación (6 000 imágenes que no "
               "se usan para ajustar los pesos). Si la de entrenamiento sigue mejorando y la de validación "
               "no, la red empieza a memorizar (sobreajuste).")

    st.markdown("### Matrices de confusión en las 10 000 imágenes de prueba")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        mostrar(viz.matriz_confusion(mc["matriz_confusion"], "Red convolucional"))
    with c2:
        mostrar(viz.matriz_confusion(md["matriz_confusion"], "Red densa"))
    st.caption("La diagonal son los aciertos. Cada número fuera de la diagonal cuenta cuántas veces "
               "se confundió el dígito de la fila con el de la columna.")

    st.markdown("### Dígitos que la densa falla y la convolucional acierta")
    y, pd_, pc_ = PRUEBA["y"], PRUEBA["densa"], PRUEBA["cnn"]
    casos = np.flatnonzero((pd_ != y) & (pc_ == y))
    st.write(f"Hay {len(casos)} casos así (y {int(((pd_ == y) & (pc_ != y)).sum())} al revés).")
    if len(casos):
        muestra = casos[:16]
        cols = st.columns(8)
        for j, i in enumerate(muestra):
            with cols[j % 8]:
                st.image(viz.digito_grande(PRUEBA["x"][i], 112),
                         caption=f"real {y[i]}, densa {pd_[i]}", width="stretch")
