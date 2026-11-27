"""
Página «Regresión lineal» — una sola neurona aprende y = m·x + b.

El resultado se guarda en st.session_state: mover un control NO reentrena,
solo el botón «Entrenar» lo hace.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import visualizaciones as viz
from comun import cargar_resultados, miles, num

METRICAS, _ = cargar_resultados()
PRE = (METRICAS or {}).get("regresion", {}).get("lineal")

# ---------------------------------------------------------------------------
# Barra lateral: controles de ESTA página
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Controles")
    m_real = st.slider("Pendiente (m)", -5.0, 5.0, 3.0, 0.5,
                       help="Qué tan inclinada es la recta: cuánto sube y por cada unidad que sube x.")
    b_real = st.slider("Intercepto (b)", -10.0, 10.0, 1.0, 0.5,
                       help="Dónde corta la recta al eje y (el valor de y cuando x = 0).")
    ruido = st.slider("Nivel de ruido", 0.0, 5.0, 0.0, 0.25,
                      help="Desviación estándar del ruido gaussiano que se le suma a y. "
                           "0 = datos perfectos.")
    entrenar = st.button("Entrenar", type="primary", width="stretch")
    st.caption("Mover los controles no entrena: pulsa **Entrenar** (tarda unos 15 s).")

if entrenar:
    with st.spinner("Entrenando la neurona (100 épocas)…"):
        import regresion as R                     # TensorFlow se importa solo cuando hace falta
        st.session_state["res_lineal"] = R.entrenar_lineal(m_real, b_real, ruido)
elif "res_lineal" not in st.session_state and PRE is not None:
    st.session_state["res_lineal"] = PRE          # resultado precalculado (m = 3, b = 1, sin ruido)

res = st.session_state.get("res_lineal")

# ---------------------------------------------------------------------------
# Texto de introducción
# ---------------------------------------------------------------------------
st.markdown("<div class='hero'><div><h1>Una neurona que aprende una recta</h1>"
            "<p class='bajada'>La red más pequeña posible: una sola neurona, sin función de activación. "
            "Solo tiene dos números que ajustar, una pendiente <b>w</b> y un sesgo <b>b</b>, y calcula "
            "<b>ŷ = w · x + b</b>. Si los datos vienen de una recta, la neurona debería descubrir "
            "esos dos números sola.</p></div></div>", unsafe_allow_html=True)

st.markdown("<div class='nota'><b>Analogía.</b> Imagina que tienes una regla y la apoyas sobre puntos "
            "dispersos en una hoja. Puedes girarla (pendiente) y subirla o bajarla (intercepto). "
            "Entrenar es ir moviendo esos dos mandos poquito a poquito, mirando cada vez cuánto se "
            "aleja la regla de los puntos, hasta que el error ya no baja.</div>", unsafe_allow_html=True)

if res is None:
    st.info("Pulsa **Entrenar** en la barra lateral para entrenar la neurona.")
    st.stop()

# Aviso si los controles ya no coinciden con lo que se entrenó
if (res["m_real"], res["b_real"], res["ruido"]) != (float(m_real), float(b_real), float(ruido)):
    st.warning(f"Los controles cambiaron, pero lo que ves se entrenó con m = {num(res['m_real'], 2)}, "
               f"b = {num(res['b_real'], 2)} y ruido = {num(res['ruido'], 2)}. Pulsa **Entrenar** para actualizar.")

# ---------------------------------------------------------------------------
# Resultados: números
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Pendiente aprendida", num(res["m_aprendido"], 3), f"real {num(res['m_real'], 2)}", delta_color="off")
k2.metric("Intercepto aprendido", num(res["b_aprendido"], 3), f"real {num(res['b_real'], 2)}", delta_color="off")
k3.metric("Error en prueba (MSE)", num(res["mse_prueba"], 5))
k4.metric("Parámetros", f"{res['parametros']}", help="1 peso (w) + 1 sesgo (b)")

n_total = res["n_total"]
st.caption(f"Datos: {miles(n_total)} puntos con x entre −10 y 10. Se separó 80 % para entrenar "
           f"y 20 % para probar; dentro del entrenamiento, otro 20 % se usa como validación "
           f"(la red nunca ajusta sus pesos con esos puntos). Épocas: {res['epocas_usadas']}.")

# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("### Datos y predicción")
    x_te, y_te, p_te = map(np.asarray, (res["x_te"], res["y_te"], res["pred_te"]))
    df_pts = pd.DataFrame({"x": x_te, "y": y_te})
    df_pred = pd.DataFrame({"x": x_te, "y": p_te, "serie": "predicción de la red"})
    x2 = np.array([x_te.min(), x_te.max()])
    df_real = pd.DataFrame({"x": x2, "y": res["m_real"] * x2 + res["b_real"], "serie": "recta real"})
    puntos = alt.Chart(df_pts).mark_circle(size=22, opacity=0.55, color=viz.ORQUIDEA).encode(
        x=alt.X("x:Q", scale=alt.Scale(zero=False), axis=alt.Axis(title="x")),
        y=alt.Y("y:Q", scale=alt.Scale(zero=False), axis=alt.Axis(title="y")))
    lineas = (alt.Chart(pd.concat([df_real, df_pred]))
              .mark_line(strokeWidth=2.4)
              .encode(x="x:Q", y="y:Q",
                      color=alt.Color("serie:N",
                                      scale=alt.Scale(domain=["recta real", "predicción de la red"],
                                                      range=[viz.CIRUELA, viz.SALVIA]),
                                      legend=alt.Legend(orient="top", title=None)),
                      strokeDash=alt.StrokeDash("serie:N",
                                                scale=alt.Scale(domain=["recta real", "predicción de la red"],
                                                                range=[[1, 0], [6, 4]]), legend=None)))
    st.altair_chart(viz._estilo_altair((puntos + lineas).properties(height=300)), width="stretch")
    st.caption("Puntos: datos de prueba (no usados para entrenar). Las dos rectas casi se superponen: "
               "la neurona encontró la recta.")
with c2:
    st.markdown("### Curva de pérdida (MSE)")
    st.altair_chart(viz.curvas_perdida(res["historial"], color=viz.ORQUIDEA, altura=300,
                                       escala_log=True, titulo_y="MSE"), width="stretch")
    st.caption("Escala logarítmica: el error cae de decenas a casi cero. "
               "Línea continua: entrenamiento. Punteada: validación.")

# ---------------------------------------------------------------------------
# Explicaciones
# ---------------------------------------------------------------------------
if res["ruido"] > 0:
    st.markdown(f"<div class='nota'>Con ruido de desviación {num(res['ruido'], 2)}, ningún modelo puede bajar "
                f"de un error ≈ ruido² = {num(res['ruido'] ** 2, 3)}: el ruido es azar puro y no se puede "
                f"predecir. Aun así, w y b salen muy cerca de los reales, porque la red promedia el azar "
                f"de {miles(n_total)} puntos.</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='nota'>Sin ruido el error en prueba queda casi en 0: los datos son exactamente "
                "una recta y la neurona puede representarla. Sube el nivel de ruido y vuelve a entrenar "
                "para ver cómo el error ya no puede bajar de cierto piso.</div>", unsafe_allow_html=True)

with st.expander("¿Qué significa «error cuadrático medio» (MSE) y cómo aprende la neurona?"):
    st.write("Para cada punto se mide cuánto falló la predicción, se eleva al cuadrado (así los errores "
             "grandes pesan más y no se cancelan con los negativos) y se promedia:")
    st.latex(r"\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}\left(y_i-\hat y_i\right)^2,\qquad \hat y_i=w\,x_i+b")
    st.write("Adam calcula hacia dónde mover **w** y **b** para que el MSE baje y da un pasito en esa "
             "dirección con cada lote de 64 puntos. Una época es una vuelta completa por los datos de "
             "entrenamiento.")
