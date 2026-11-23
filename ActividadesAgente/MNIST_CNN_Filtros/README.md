# Atelier neuronal: red densa vs. red convolucional en MNIST

Aplicación en Streamlit que muestra, en vivo, cómo dos redes neuronales "miran" un dígito escrito a mano:

- **Red densa** (la de la clase): 784 → 64 → 64 → 10.
- **Red convolucional** con tres capas de filtros 3 × 3 (16, 32 y 64 filtros) y MaxPooling.

Las dos tienen casi el mismo número de parámetros (≈ 55 000), así que la comparación es justa: lo único que cambia es la arquitectura.

## Qué se puede ver en la app

| Pestaña | Contenido |
|---|---|
| Predicción | El dígito que predice cada red y sus 10 probabilidades (softmax). |
| Red densa | El dígito aplanado en 784 valores, las activaciones de las capas ocultas y los pesos de cada neurona vistos como imagen 28 × 28, con el cálculo de una neurona paso a paso. |
| Red convolucional, capa por capa | Para **cada capa convolucional**: los filtros aprendidos (pesos 3 × 3), los mapas de características que produce con tu dígito, la salida del pooling y el cálculo de un píxel hecho a mano y comparado con Keras. |
| Laboratorio de filtros | Filtros clásicos (Sobel, Laplaciano, desenfoque…) o un filtro escrito por ti, aplicados al dígito, y el filtro aprendido por la red que más se le parece. |
| Comparación | Exactitud, curvas de entrenamiento, matrices de confusión y dígitos que la densa falla y la convolucional acierta. |

El dígito puede venir del conjunto de prueba de MNIST, de la carpeta `mis_imagenes/`, de una imagen subida o de la cámara. Las fotos reales pasan por `preprocesamiento.py` para quedar en el formato MNIST.

## Archivos

```
MNIST_CNN_Filtros/
├── entrenar_modelos.py    entrena las dos redes y guarda todo en modelos/
├── app.py                 la aplicación de Streamlit
├── visualizaciones.py     funciones de dibujo y la convolución hecha con numpy
├── preprocesamiento.py    convierte una foto real al formato MNIST
├── estilo.css             tipografías y colores de la página
├── .streamlit/config.toml colores base de Streamlit
├── ejemplos/              10 dígitos de ejemplo
├── mis_imagenes/          pon aquí tus propias fotos
├── modelos/               (se llena al entrenar)
├── requirements.txt
└── GUIA.md                instrucciones paso a paso para la VM
```

## Uso rápido

```bash
pip install -r requirements.txt
python entrenar_modelos.py      # una sola vez
streamlit run app.py            # http://localhost:8501
```

Los pasos detallados están en [GUIA.md](GUIA.md).
