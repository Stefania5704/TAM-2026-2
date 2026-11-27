"""
Página «Regresión de un coseno» — por qué falla la red del cuaderno y cómo arreglarla.

Parte 1: la red del cuaderno (resultado precalculado en entrenar_modelos.py).
Parte 2: las versiones mejoradas, comparadas con números.
Parte 3: laboratorio — tú eliges la red y pulsas «Entrenar». El resultado se guarda en
         st.session_state, así que mover un control NO reentrena.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import visualizaciones as viz
from comun import cargar_resultados, num

METRICAS, _ = cargar_resultados()
PRE = (METRICAS or {}).get("regresion", {}).get("coseno")
if not PRE:
    st.title("Falta entrenar las redes")
    st.write("No encontré los resultados de regresión. Genéralos con:")
    st.code("python entrenar_modelos.py", language="bash")
    st.stop()

NOMBRE_MEJOR = PRE["mejor"]
CUADERNO, MEJOR = PRE["resultados"]["cuaderno"], PRE["resultados"][NOMBRE_MEJOR]
AJ0 = MEJOR["ajustes"]                       # valores por defecto de los controles = la mejor receta


def grafico_senal(res, altura=300):
    """Señal real contra predicción de la red (toda la señal, no solo la parte de prueba)."""
    paso = max(1, len(res["x_todo"]) // 800)
    df = pd.DataFrame({"tiempo (s)": np.asarray(res["x_todo"])[::paso],
                       "señal real": np.asarray(res["y_todo"])[::paso],
                       "predicción de la red": np.asarray(res["pred_todo"])[::paso]})
    largo = df.melt("tiempo (s)", var_name="serie", value_name="amplitud")
    dom = ["señal real", "predicción de la red"]
    g = (alt.Chart(largo).mark_line(strokeWidth=2.2)
         .encode(x=alt.X("tiempo (s):Q"),
                 y=alt.Y("amplitud:Q", scale=alt.Scale(domain=[-1.6, 1.6])),
                 color=alt.Color("serie:N", scale=alt.Scale(domain=dom, range=[viz.CIRUELA, viz.ORQUIDEA]),
                                 legend=alt.Legend(orient="top", title=None)),
                 strokeDash=alt.StrokeDash("serie:N", scale=alt.Scale(domain=dom, range=[[1, 0], [6, 3]]),
                                           legend=None),
                 tooltip=["serie", alt.Tooltip("tiempo (s):Q", format=".3f"),
                          alt.Tooltip("amplitud:Q", format=".3f")])
         .properties(height=altura))
    return viz._estilo_altair(g)


def veredicto(res):
    """Texto corto: ¿aprendió o no? Se compara el error con la varianza de la señal."""
    frac = res["mse_prueba"] / res["varianza_senal"]
    if frac > 0.8:
        return ("fallo", f"El error ({num(res['mse_prueba'], 3)}) es casi igual a la varianza de la señal "
                         f"({num(res['varianza_senal'], 3)}): la red no aprendió nada mejor que predecir una "
                         f"línea plana cerca de 0.")
    if frac > 0.1:
        return ("parcial", f"El error ({num(res['mse_prueba'], 3)}) es menor que la varianza "
                           f"({num(res['varianza_senal'], 3)}), pero la red solo captó parte de la onda.")
    return ("exito", f"El error ({num(res['mse_prueba'], 5)}) es mínimo frente a la varianza "
                     f"({num(res['varianza_senal'], 3)}): la red reproduce la onda.")


# ---------------------------------------------------------------------------
# Barra lateral: controles del laboratorio (Parte 3)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Laboratorio")
    st.caption("Estos controles son de la parte 3, al final de la página.")
    frecuencia = st.slider("Frecuencia (Hz)", 0.5, 5.0, float(AJ0["f"]), 0.5,
                           help="Cuántas oscilaciones completas por segundo. La señal dura 4 s.")
    neuronas = st.slider("Neuronas por capa", 2, 128, int(AJ0["neuronas"]), 2)
    capas = st.slider("Número de capas ocultas", 1, 5, int(AJ0["capas"]))
    opciones_act = ["selu", "tanh", "relu"]
    activacion = st.selectbox("Activación", opciones_act, index=opciones_act.index(AJ0["activacion"]))
    with st.expander("Opciones avanzadas"):
        fourier = st.checkbox("Entradas de Fourier (sin y cos)", value=bool(AJ0["fourier"]),
                              help="Le entrega a la red, además de t, un «peine» de senos y cosenos "
                                   "de varias frecuencias.")
        lrs = [0.001, 0.003, 0.01]
        lr = st.select_slider("Tasa de aprendizaje", options=lrs, value=float(AJ0["lr"]),
                              format_func=lambda v: f"{v:g}")
        early = st.checkbox("EarlyStopping (paciencia 30)", value=bool(AJ0["early"]))
        epocas = st.slider("Épocas máximas", 50, 500, int(AJ0["epocas"]), 50)
    entrenar = st.button("Entrenar", type="primary", width="stretch")

AJUSTES = dict(f=float(frecuencia), neuronas=int(neuronas), capas=int(capas), activacion=activacion,
               fourier=bool(fourier), lr=float(lr), early=bool(early), epocas=int(epocas))

if entrenar:
    with st.spinner("Entrenando… (puede tardar de 20 s a 2 min según la red)"):
        import regresion as R                     # TensorFlow se importa solo cuando hace falta
        resultado = R.entrenar_coseno(**AJUSTES, lote=AJ0.get("lote", 64))
    st.session_state["res_coseno"] = (AJUSTES, resultado)
elif "res_coseno" not in st.session_state:
    st.session_state["res_coseno"] = ({k: AJ0[k] for k in AJUSTES}, MEJOR)

# ---------------------------------------------------------------------------
# Introducción
# ---------------------------------------------------------------------------
st.markdown("<div class='hero'><div><h1>Aprender una onda: cuando una red pequeña no alcanza</h1>"
            "<p class='bajada'>La señal es un coseno de 2 Hz medido durante 4 segundos: sube y baja "
            "8 veces. Le damos a la red solo el tiempo <b>t</b> y le pedimos el valor de la onda en ese "
            "instante. Parece fácil, pero la red del cuaderno falla. Veamos por qué y cómo arreglarlo."
            "</p></div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Parte 1: la red del cuaderno falla
# ---------------------------------------------------------------------------
st.markdown("## 1. La red del cuaderno falla")
st.write("Es la red del cuaderno: tres capas de 10 neuronas con activación SELU, entrenada 100 épocas con "
         "Adam. Esto es lo que aprendió:")
a1, a2 = st.columns(2, gap="large")
with a1:
    st.altair_chart(grafico_senal(CUADERNO), width="stretch")
    st.caption("Línea oscura: la señal real. Línea rosa punteada: lo que predice la red.")
with a2:
    st.altair_chart(viz.curvas_perdida(CUADERNO["historial"], color=viz.ORQUIDEA, altura=300,
                                       titulo_y="MSE"), width="stretch")
    st.caption("Las curvas se aplanan casi en 0,5 y no vuelven a bajar.")
b1, b2, b3 = st.columns(3)
b1.metric("Error en prueba (MSE)", num(CUADERNO["mse_prueba"], 3))
b2.metric("Varianza de la señal", num(CUADERNO["varianza_senal"], 3),
          help="Un coseno de amplitud 1 tiene varianza 0,5: es el error de predecir siempre el promedio (0).")
b3.metric("Parámetros", f"{CUADERNO['parametros']}")
st.markdown(f"<div class='nota'><b>Lectura.</b> {veredicto(CUADERNO)[1]}</div>", unsafe_allow_html=True)

with st.expander("¿Por qué falla? (sesgo espectral y capacidad)", expanded=True):
    st.markdown(
        "- **El error de referencia.** Si una red solo predijera el promedio de la señal (0), su error sería "
        "la varianza: 0,5. Un error cercano a 0,5 significa «no aprendí nada útil».\n"
        "- **La onda es difícil de dibujar con poco.** Con 8 subidas y bajadas hay 16 giros que la red debe "
        "construir a partir de un solo número (t). Con 10 neuronas por capa tiene muy pocas «piezas» para "
        "doblar la curva tantas veces. Analogía: intentar dibujar un zigzag de 16 picos con una regla "
        "articulada de solo unas pocas bisagras.\n"
        "- **Las redes aprenden primero lo suave.** Se llama *sesgo espectral*: el descenso de gradiente "
        "ajusta antes las variaciones lentas que las rápidas. Con pocas épocas y tasa de aprendizaje chica, "
        "la red se queda en la solución más fácil: una línea casi plana.\n"
        "- **No es un error de código ni de datos.** La misma red resolvería sin problema una onda más "
        "lenta; el problema es la combinación frecuencia alta + red chica + pocas épocas.")

# ---------------------------------------------------------------------------
# Parte 2: versiones mejoradas
# ---------------------------------------------------------------------------
st.markdown("## 2. Versiones mejoradas")
st.write("Probé varias ideas con la misma señal (2 Hz), la misma partición y la misma semilla. "
         "Todas son redes densas; cambian solo las piezas que dice la tabla.")
filas = []
for nombre, r in PRE["resultados"].items():
    aj = r["ajustes"]
    filas.append({"Receta": aj["titulo"],
                  "Capas × neuronas": f"{aj['capas']} × {aj['neuronas']}",
                  "Activación": aj["activacion"],
                  "Fourier": "sí" if aj["fourier"] else "no",
                  "Parámetros": r["parametros"],
                  "Épocas usadas": r["epocas_usadas"],
                  "Error en prueba (MSE)": float(f"{r['mse_prueba']:.5f}"),
                  "¿Funciona?": {"fallo": "no", "parcial": "a medias", "exito": "sí"}[veredicto(r)[0]],
                  "★": "★ elegida" if nombre == NOMBRE_MEJOR else ""})
st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown(f"**Elegida: {AJ0['titulo']}**")
    st.altair_chart(grafico_senal(MEJOR), width="stretch")
with c2:
    st.markdown("**Entrenamiento y validación**")
    st.altair_chart(viz.curvas_perdida(MEJOR["historial"], color=viz.SALVIA, altura=300,
                                       parada=MEJOR.get("mejor_epoca"), escala_log=True, titulo_y="MSE"),
                    width="stretch")
st.markdown(
    "<div class='nota'><b>Por qué funciona la entrada de Fourier.</b> En vez de pedirle a la red que "
    "<i>invente</i> una oscilación a partir de t, le damos un «peine» de relojes: sin(2π·f·t) y cos(2π·f·t) "
    "para muchas frecuencias f. A la red solo le queda <i>elegir y mezclar</i> los relojes correctos, que es un "
    "problema mucho más fácil que doblar una curva 16 veces. Es la misma idea con la que las "
    "redes modernas representan posiciones y señales periódicas. Ojo: el banco de frecuencias "
    "incluye la frecuencia verdadera, así que esta receta «sabe» que la señal es periódica. "
    "En datos reales la frecuencia no se conoce; por eso también dejé las otras recetas como comparación."
    "</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Parte 3: laboratorio
# ---------------------------------------------------------------------------
st.markdown("## 3. Laboratorio: entrena tu propia red")
st.write("Elige frecuencia, neuronas, capas y activación en la barra lateral y pulsa **Entrenar**. "
         "Con los valores iniciales verás la receta elegida (resultado precalculado); si cambias algo, "
         "la red se entrena de nuevo solo cuando pulses el botón.")
ajustes_ent, res = st.session_state["res_coseno"]
if {k: ajustes_ent[k] for k in AJUSTES} != AJUSTES:
    st.warning("Los controles cambiaron, pero lo que ves se entrenó con otros valores: "
               f"{ajustes_ent['capas']} capas × {ajustes_ent['neuronas']} neuronas, "
               f"{ajustes_ent['activacion']}, {num(ajustes_ent['f'], 1)} Hz"
               f"{', con Fourier' if ajustes_ent['fourier'] else ''}. Pulsa **Entrenar** para actualizar.")

d1, d2 = st.columns(2, gap="large")
with d1:
    st.markdown("**Señal real contra predicción**")
    st.altair_chart(grafico_senal(res), width="stretch")
with d2:
    st.markdown("**Curvas de entrenamiento y validación**")
    st.altair_chart(viz.curvas_perdida(res["historial"], color=viz.ORQUIDEA, altura=300,
                                       parada=res.get("mejor_epoca"), escala_log=True, titulo_y="MSE"),
                    width="stretch")
    if res.get("mejor_epoca"):
        st.caption(f"La línea verde marca la mejor época ({res['mejor_epoca']}): EarlyStopping se quedó con esos pesos.")

e1, e2, e3, e4 = st.columns(4)
e1.metric("Error final en prueba (MSE)", num(res["mse_prueba"], 5))
e2.metric("Error absoluto medio (MAE)", num(res["mae_prueba"], 4))
e3.metric("Varianza explicada", f"{max(0.0, 1 - res['mse_prueba'] / res['varianza_senal']) * 100:.1f}".replace(".", ",") + " %",
          help="1 − MSE / varianza de la señal. 0 % = igual que predecir el promedio; 100 % = perfecto.")
e4.metric("Épocas / parámetros", f"{res['epocas_usadas']} / {res['parametros']}")
tipo, texto = veredicto(res)
st.markdown(f"<div class='nota'><b>Lectura.</b> {texto}</div>", unsafe_allow_html=True)
