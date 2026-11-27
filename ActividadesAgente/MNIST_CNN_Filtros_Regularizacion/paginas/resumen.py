"""
Página «Resumen de modelos» — todos los modelos entrenados en una sola tabla.
"""
import pandas as pd
import streamlit as st

from comun import cargar_resultados, miles, num, pct

METRICAS, _ = cargar_resultados()
REG = (METRICAS or {}).get("regresion")
if METRICAS is None or REG is None or "epocas_usadas" not in METRICAS.get("cnn", {}):
    st.title("Falta entrenar las redes")
    st.write("No encontré los resultados completos. Genéralos con:")
    st.code("python entrenar_modelos.py", language="bash")
    st.stop()

st.markdown("<div class='hero'><div><h1>Todos los modelos, lado a lado</h1>"
            "<p class='bajada'>Seis redes entrenadas en este proyecto: tres clasifican dígitos (MNIST) y "
            "tres resuelven regresiones. La métrica cambia según la tarea: en clasificación "
            "se mide la <b>exactitud</b> (más alta es mejor); en regresión, el <b>error cuadrático medio</b> "
            "(más bajo es mejor).</p></div></div>", unsafe_allow_html=True)

filas = []
for clave, nombre, nota in [("densa_sin_norm", "Densa, píxeles sin normalizar (0-255)", "784 → 64 → 64 → 10"),
                            ("densa", "Densa, píxeles normalizados (0-1)", "784 → 64 → 64 → 10"),
                            ("cnn", "Convolucional (la del cuaderno)", "conv 32 → pool → conv 64 → pool → 10")]:
    m = METRICAS[clave]
    filas.append({"Modelo": nombre, "Tarea": "Clasificación MNIST", "Arquitectura": nota,
                  "Parámetros": miles(m["parametros"]),
                  "Resultado en prueba": f"exactitud {pct(m['exactitud_prueba'], 2)}",
                  "Épocas usadas": f"{m['epocas_usadas']} (mejor: {m['mejor_epoca']})"})

lin = REG["lineal"]
filas.append({"Modelo": "Regresión lineal", "Tarea": "Regresión: y = 3x + 1", "Arquitectura": "1 neurona, sin activación",
              "Parámetros": miles(lin["parametros"]),
              "Resultado en prueba": f"MSE {lin['mse_prueba']:.2e}".replace(".", ","),
              "Épocas usadas": f"{lin['epocas_usadas']}"})
mejor = REG["coseno"]["mejor"]
for nombre, r in REG["coseno"]["resultados"].items():
    aj = r["ajustes"]
    etiqueta = aj["titulo"] + (" ★" if nombre == mejor else "")
    arq = f"{aj['capas']} × {aj['neuronas']} {aj['activacion']}" + (" + Fourier" if aj["fourier"] else "")
    mejor_ep = f" (mejor: {r['mejor_epoca']})" if r.get("mejor_epoca") else ""
    filas.append({"Modelo": etiqueta, "Tarea": "Regresión: coseno de 2 Hz", "Arquitectura": arq,
                  "Parámetros": miles(r["parametros"]),
                  "Resultado en prueba": f"MSE {num(r['mse_prueba'], 4)} (varianza {num(r['varianza_senal'], 2)})",
                  "Épocas usadas": f"{r['epocas_usadas']}{mejor_ep}"})
st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
st.caption("★ = receta elegida para el coseno. MSE = error cuadrático medio sobre el 20 % de datos de prueba "
           "(nunca usados para entrenar). Épocas: cuántas vueltas completas dio la red por los datos; "
           "«mejor» es la época cuyos pesos se conservaron al activar EarlyStopping.")

# ---------------------------------------------------------------------------
# Lo aprendido (los números salen de los resultados guardados)
# ---------------------------------------------------------------------------
md, mc, msn = METRICAS["densa"], METRICAS["cnn"], METRICAS["densa_sin_norm"]
cuaderno = REG["coseno"]["resultados"]["cuaderno"]
elegida = REG["coseno"]["resultados"][mejor]

st.markdown("## Lo que aprendimos")
st.markdown(f"""
1. **La arquitectura importa.** Con un número parecido de parámetros ({miles(mc['parametros'])} contra
   {miles(md['parametros'])}), la convolucional llegó a {pct(mc['exactitud_prueba'], 2)} y la densa a
   {pct(md['exactitud_prueba'], 2)}. Los filtros que se deslizan reutilizan los mismos pesos en toda la imagen,
   y eso encaja con la forma de los dígitos.
2. **Normalizar ayuda.** Con píxeles de 0 a 255 la densa llegó a {pct(msn['exactitud_prueba'], 2)}; con 0 a 1,
   a {pct(md['exactitud_prueba'], 2)}. Entradas en una escala moderada hacen que la pérdida arranque más baja y
   que el aprendizaje sea más estable.
3. **Parar a tiempo es parte del entrenamiento.** EarlyStopping vigila la validación y devuelve los mejores pesos:
   la convolucional usó {mc['epocas_usadas']} épocas, la densa {md['epocas_usadas']}, de un máximo de
   {METRICAS['config']['epocas']}.
4. **Una neurona basta si el problema es lineal.** Con 2 parámetros recuperó m = {num(lin['m_aprendido'], 3)} y
   b = {num(lin['b_aprendido'], 3)} (reales: 3 y 1).
5. **Una red pequeña puede fallar aunque «no tenga errores».** La red del cuaderno para el coseno quedó con MSE
   {num(cuaderno['mse_prueba'], 3)}, casi igual a la varianza de la señal ({num(cuaderno['varianza_senal'], 2)}):
   equivale a predecir una línea plana.
6. **A veces la solución está en los datos de entrada, no en la red.** Darle a la red senos y cosenos de varias
   frecuencias bajó el error a {num(elegida['mse_prueba'], 5)}. Las variantes que solo agrandaron la red o
   cambiaron la activación no resolvieron el problema.
""")
