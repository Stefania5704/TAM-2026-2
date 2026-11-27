"""
Entrena las redes de regresión con sus valores por defecto y las deja en un formato
liviano (listas de números) para guardarlas dentro de metricas.json.

Lo llama entrenar_modelos.py. La app lee estos resultados para la página de resumen
y para mostrar al instante la red del cuaderno y la receta elegida.
"""
import numpy as np

import regresion as R


def _lista(a, paso=1, dec=5):
    return [round(float(v), dec) for v in np.asarray(a)[::paso]]


def _hist(h):
    return {"loss": _lista(h["loss"], dec=7), "val_loss": _lista(h["val_loss"], dec=7)}


def compactar_lineal(r):
    return {**{k: r[k] for k in ("m_real", "b_real", "ruido", "m_aprendido", "b_aprendido",
                                 "mse_prueba", "parametros", "epocas_usadas", "segundos", "n_total")},
            "historial": _hist(r["historial"]),
            "x_te": _lista(r["x_te"], 2), "y_te": _lista(r["y_te"], 2), "pred_te": _lista(r["pred_te"], 2)}


def compactar_coseno(r, ajustes):
    return {**{k: r[k] for k in ("f", "mse_prueba", "mae_prueba", "varianza_senal", "parametros",
                                 "epocas_usadas", "mejor_epoca", "segundos")},
            "ajustes": ajustes, "historial": _hist(r["historial"]),
            "x_todo": _lista(r["x_todo"], 5), "y_todo": _lista(r["y_todo"], 5),
            "pred_todo": _lista(r["pred_todo"], 5)}


def entrenar_todo():
    print("\n" + "=" * 70 + "\n  Entrenando REGRESIÓN LINEAL\n" + "=" * 70)
    lineal = R.entrenar_lineal()
    print(f">>> lineal: m = {lineal['m_aprendido']:.4f} (real 3), b = {lineal['b_aprendido']:.4f} (real 1), "
          f"MSE prueba = {lineal['mse_prueba']:.2e}")

    resultados = {}
    for nombre, cfg in R.CONFIGS_COSENO.items():
        print("\n" + "=" * 70 + f"\n  Entrenando COSENO — {cfg['titulo']}\n" + "=" * 70)
        ajustes = {**cfg, "f": 2.0}
        r = R.entrenar_coseno(**{k: v for k, v in ajustes.items() if k != "titulo"})
        print(f">>> {nombre}: MSE prueba = {r['mse_prueba']:.5f} (varianza {r['varianza_senal']:.3f}), "
              f"{r['epocas_usadas']} épocas, {r['parametros']} parámetros")
        resultados[nombre] = compactar_coseno(r, ajustes)
    mejor = min((n for n in resultados if n != "cuaderno"), key=lambda n: resultados[n]["mse_prueba"])
    return {"lineal": compactar_lineal(lineal),
            "coseno": {"mejor": mejor, "resultados": resultados}}
