# Atelier neuronal: clasificación (MNIST) y regresión con redes neuronales

Aplicación en Streamlit con dos secciones, cada una con sus propios controles en la barra lateral:

- **Clasificación MNIST**: cómo dos redes "miran" un dígito escrito a mano.
- **Regresión**: una neurona que aprende una recta, y una red que intenta aprender un coseno.
- **Resumen de modelos**: tabla con todos los modelos entrenados y lo aprendido.

## Redes de MNIST

- **Red densa** (la de la clase): 784 → 64 → 64 → 10 (55 050 parámetros).
- **Red convolucional** (igual a la del cuaderno `scratchpad_4.ipynb`, 50 186 parámetros):

  ```
  Input 28×28×1 → conv_1 (32 filtros 3×3, ReLU, same) → pool_1 (2×2)   → 14×14×32
                → conv_2 (64 filtros 3×3, ReLU, same) → pool_2 (2×2)   →  7×7×64
                → aplanar (3136) → salida (Dense 10, softmax)
  ```

  Sin tercera capa convolucional y sin Dropout.
- **Red densa sin normalizar** (píxeles 0-255, como en el cuaderno), solo para comparar con la normalizada.

Las tres se entrenan con **EarlyStopping** (monitor `val_loss`, paciencia 3, `restore_best_weights=True`, máximo 30 épocas) y 6 000 imágenes de validación. La app muestra cuántas épocas usó cada una.

## Redes de regresión

- **Recta** `y = m·x + b`: una neurona sin activación (2 parámetros). Controles: pendiente, intercepto y ruido.
- **Coseno** (2 Hz, `t` entre −2 y 2 s): se reproduce la red del cuaderno (3 capas de 10 neuronas SELU, 100 épocas), que **falla**, y se comparan versiones mejoradas. Controles: frecuencia, neuronas, capas, activación y un botón **Entrenar** (el resultado se guarda en `st.session_state`).

## Qué se puede ver en la app

| Sección / página | Contenido |
|---|---|
| **Clasificación MNIST** → Predicción | El dígito que predice cada red y sus 10 probabilidades (softmax). |
| → Red densa | El dígito aplanado en 784 valores, las activaciones de las capas ocultas y los pesos de cada neurona vistos como imagen 28 × 28, con el cálculo de una neurona paso a paso. |
| → Red convolucional, capa por capa | Para cada capa convolucional (`conv_1` y `conv_2`): los filtros aprendidos, los mapas de características con tu dígito, la salida del pooling y el cálculo de un píxel hecho a mano y comparado con Keras. |
| → Laboratorio de filtros | Filtros clásicos (Sobel, Laplaciano, desenfoque…) o uno escrito por ti, y el filtro aprendido por `conv_1` que más se le parece. |
| → Comparación | Exactitud, épocas usadas, curvas de entrenamiento y validación con la época donde paró EarlyStopping, comparación sin normalizar contra normalizada, cómo detectar sobreajuste, matrices de confusión y dígitos que la densa falla. |
| **Regresión** → Regresión lineal | Datos contra predicción, curva MSE de entrenamiento y validación, y `m` y `b` aprendidos contra los reales. |
| → Regresión de un coseno | La red del cuaderno falla (y por qué), recetas mejoradas comparadas con números, y un laboratorio para entrenar tu propia red. |
| **Balance** → Resumen de modelos | Tabla con parámetros, exactitud o error en prueba y épocas usadas de todos los modelos. |

El dígito puede venir del conjunto de prueba de MNIST, de la carpeta `mis_imagenes/`, de una imagen subida o de la cámara. Las fotos reales pasan por `preprocesamiento.py` para quedar en el formato MNIST.

## Archivos

```
MNIST_CNN_Filtros/
├── app.py                 portero: carga el estilo y define las páginas (st.navigation)
├── paginas/
│   ├── mnist.py               clasificación MNIST (5 pestañas)
│   ├── regresion_lineal.py    regresión de una recta
│   ├── regresion_coseno.py    regresión de un coseno
│   └── resumen.py             tabla de todos los modelos
├── comun.py               rutas, formatos de número y carga de resultados
├── entrenar_modelos.py    entrena MNIST (3 redes) y la regresión, y guarda todo en modelos/
├── regresion.py           datos, redes y entrenamiento de la regresión
├── regresion_resumen.py   deja los resultados de regresión listos para metricas.json
├── visualizaciones.py     funciones de dibujo, curvas y la convolución hecha con numpy
├── preprocesamiento.py    convierte una foto real al formato MNIST
├── estilo.css             tipografías y colores de la página
├── .streamlit/config.toml colores base y servidor (address 0.0.0.0 para abrir desde Windows)
├── ejemplos/              10 dígitos de ejemplo
├── mis_imagenes/          pon aquí tus propias fotos
├── modelos/               (se llena al entrenar)
├── requirements.txt
└── GUIA.md                instrucciones paso a paso para la VM
```

## Uso rápido

```bash
pip install -r requirements.txt
python entrenar_modelos.py      # una sola vez (MNIST + regresión)
streamlit run app.py            # http://localhost:8501
```

Opciones útiles de `entrenar_modelos.py`: `--epocas 5` (más rápido, para probar) y `--sin-regresion` (solo MNIST).

Los pasos detallados están en [GUIA.md](GUIA.md).
