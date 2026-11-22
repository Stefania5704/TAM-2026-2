# Guía: Gentle Shell + Claude en Ubuntu y el reconocedor de dígitos MNIST

Esta guía tiene tres partes:

1. **Qué es cada cosa** (modelo, agente, Gentle AI, entrenamiento), porque es lo que más se confunde.
2. **Cómo instalar y configurar** Gentle Shell / Gentle AI con Claude en la máquina virtual Ubuntu.
3. **El proyecto**: entrenar el modelo del cuaderno, guardarlo y probarlo arrastrando imágenes propias, con la razón de cada paso.

---

## Parte 1 — El mapa: hay dos "modelos" distintos

En esta actividad aparece la palabra *modelo* para dos cosas que no tienen nada que ver entre sí:

| | **Claude** | **Tu red de MNIST** |
|---|---|---|
| Qué es | Un modelo de lenguaje (LLM) | Una red neuronal densa pequeña |
| Tamaño | Enorme | 55 050 parámetros |
| Quién lo entrena | Anthropic, una sola vez | **Tú**, en la CPU de tu máquina virtual |
| Dónde corre | En servidores de Anthropic (por internet) | En tu VM, con TensorFlow |
| Para qué sirve aquí | Leer tu cuaderno, escribir el código, ejecutarlo, explicarte | Recibir una imagen de 28×28 y decir qué dígito es |

**Claude no "se entrena" con tus imágenes.** Claude escribe y ejecuta el código que entrena *tu* red. El que aprende de los datos es tu modelo de TensorFlow.

### ¿Qué es un agente?

Un LLM por sí solo solo produce texto. Un **agente** es un programa que corre en tu computador y le da al LLM *herramientas*: leer archivos, escribirlos, ejecutar comandos en la terminal. El agente funciona en un ciclo:

```
  tú das una instrucción
          │
          ▼
  Claude decide qué hacer ──► usa una herramienta (p. ej. ejecutar "python entrenar_modelo.py")
          ▲                               │
          └──────── observa el resultado ◄┘   (repite hasta terminar)
```

Ejemplos de agentes: **Pi** (sobre el que está construido Gentle Shell) y **Claude Code**.

### ¿Qué son Gentle AI y Gentle Shell?

- **Gentle AI** (de Gentleman Programming) no es un agente ni un modelo: es un *configurador*. Le agrega a un agente que ya tienes cosas como memoria persistente entre sesiones (Engram), un flujo de trabajo ordenado (primero especificar, luego implementar), *skills* (instrucciones empaquetadas), una personalidad de "profesor" y permisos de seguridad.
- **Gentle Shell** es un espacio de trabajo listo para usar construido sobre el agente **Pi**, que ya trae los paquetes de Gentle AI incluidos.

Resumiendo en capas:

```
  Gentle Shell / Gentle AI   ← configuración: memoria, flujo, skills, persona
  Agente (Pi o Claude Code)  ← las "manos": lee, escribe, ejecuta en tu VM
  Claude (LLM)               ← el "cerebro": decide qué hacer
  ─────────────────────────────────────────────
  Python + TensorFlow        ← donde realmente se entrena TU red de MNIST
```

---

## Parte 2 — Instalación en Ubuntu

> Las herramientas de Gentleman Programming cambian rápido. Si el profesor les dio comandos distintos, usen los del profesor. Lo de abajo corresponde a la documentación pública de noviembre de 2026.

### 2.1 Prerrequisitos

```bash
sudo apt update
sudo apt install -y git curl python3 python3-venv python3-pip nodejs npm
node --version     # Pi necesita Node 18 o superior (idealmente 20+)
```

Si `node --version` muestra algo menor a 18, instala una versión nueva con `nvm`:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
source ~/.bashrc
nvm install --lts
```

### 2.2 Opción A — Gentle Shell (agente Pi)

```bash
git clone https://github.com/Gentleman-Programming/gentle-shell.git
cd gentle-shell
sh scripts/bootstrap.sh
```

o, si ya tienes Pi instalado:

```bash
npm i -g gentle-pi
gentle-shell
```

**Configurar Claude como IA:**

1. Abre el agente en la carpeta del proyecto (`cd ~/proyecto_mnist` y luego `gentle-shell` o `pi`).
2. Escribe `/login` y elige **Anthropic**:
   - **Claude Pro/Max** si tienes suscripción (abre el navegador para iniciar sesión). Ojo: el uso desde herramientas de terceros como Pi se cobra como *uso extra* por tokens, no contra el límite del plan.
   - o **API key**: créala en console.anthropic.com y expórtala antes de abrir el agente:
     ```bash
     export ANTHROPIC_API_KEY=sk-ant-...      # no la compartas ni la subas a GitHub
     ```
3. Escribe `/model` (o Ctrl+L) y elige un modelo de Claude.
4. Verifica con `/gentle:status` y `/gentle:doctor`.

### 2.3 Opción B — Claude Code + Gentle AI

```bash
curl -fsSL https://claude.ai/install.sh | bash          # instala Claude Code
curl -fsSL https://raw.githubusercontent.com/Gentleman-Programming/gentle-ai/main/scripts/install.sh | bash
gentle-ai            # menú interactivo: elige "Claude Code" y los componentes
gentle-ai doctor     # chequeo de salud
```

Luego entra a la carpeta del proyecto y ejecuta `claude`. La primera vez te pide iniciar sesión con tu cuenta de Claude.

---

## Parte 3 — El proyecto MNIST

### 3.1 Archivos

```
proyecto_mnist/
├── Actividad1Clase_con_app.ipynb   tu cuaderno + celdas nuevas (guardar, probar, interfaz)
├── entrenar_modelo.py              entrena la red del cuaderno y la guarda
├── modelo_mnist.keras              el modelo YA entrenado (exactitud en prueba ≈ 97 %)
├── preprocesamiento.py             convierte una foto real al formato MNIST
├── app_streamlit.py                interfaz con Streamlit (localhost:8501) — VERSIÓN PRINCIPAL
├── app.py                          interfaz alternativa con Gradio
├── requirements.txt                librerías necesarias
└── ejemplos/digito_0.png … 9.png   imágenes de prueba tipo "foto de cuaderno"
```

### 3.2 Cómo ejecutarlo (a mano o pidiéndoselo al agente)

```bash
cd ~/proyecto_mnist
python3 -m venv .venv              # entorno virtual: librerías aisladas del sistema
source .venv/bin/activate
pip install -r requirements.txt    # tensorflow, streamlit, pandas, pillow, numpy, gradio
python entrenar_modelo.py          # opcional: ya viene modelo_mnist.keras entrenado
streamlit run app_streamlit.py     # abre http://localhost:8501 en el navegador de la VM
```

**¿Por qué un entorno virtual?** Ubuntu 23.04 en adelante bloquea `pip install` sobre el Python del sistema (error `externally-managed-environment`) para no romper programas del sistema. El `.venv` es una carpeta con su propio Python y sus propias librerías.

### 3.3 Qué hace el cuaderno y por qué (paso a paso)

**a) Datos.** MNIST trae 60 000 imágenes de entrenamiento y 10 000 de prueba, de 28×28 píxeles en gris, con valores de 0 (negro) a 255 (blanco). El dígito es **blanco sobre fondo negro** y está centrado.

**b) Aplanar.** `reshape(…, 28*28)` convierte cada matriz 28×28 en un vector de 784 números, porque una capa densa recibe vectores. Se pierde la noción de "vecindad" entre píxeles; por eso este modelo depende tanto de que el dígito esté centrado y del tamaño correcto (una red convolucional sería más robusta).

**c) Normalizar** (mejora agregada). Dividir entre 255 deja los valores en [0, 1]. Con 0–255 las sumas ponderadas de la primera capa son muy grandes al inicio y el entrenamiento arranca inestable: en tu cuaderno la pérdida de la época 1 fue 1.56; normalizando baja a ~0.3.

**d) Arquitectura.** Cada capa densa calcula

$$\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$$

- Capa 1: 784 → 64, ReLU $\sigma(z)=\max(0,z)$. Parámetros: $784\cdot64 + 64 = 50\,240$.
- Capa 2: 64 → 64, ReLU. Parámetros: $64\cdot64 + 64 = 4\,160$.
- Salida: 64 → 10, **softmax**: $p_k = \dfrac{e^{z_k}}{\sum_{j=0}^{9} e^{z_j}}$. Parámetros: $64\cdot10+10 = 650$.
- Total: **55 050** parámetros (es lo que muestra `Model.summary()`).

La softmax convierte 10 números cualesquiera en 10 probabilidades que suman 1: una por cada dígito.

**e) Función de pérdida.** `sparse_categorical_crossentropy`: si la etiqueta real es $y$, la pérdida es $L = -\log p_y$. Si el modelo le da probabilidad 0.99 al dígito correcto, $L\approx0.01$; si le da 0.01, $L\approx4.6$. "Sparse" significa que la etiqueta llega como un entero (3) y no como vector one-hot.

**f) Entrenar = minimizar la pérdida.** El optimizador **Adam** calcula el gradiente $\partial L/\partial W$ con *backpropagation* y mueve cada peso un poco en la dirección que reduce la pérdida (con un paso que se adapta para cada parámetro).
- `batch_size=32`: se actualizan los pesos cada 32 imágenes. 60 000 / 32 = **1875** actualizaciones por época (el número que veías en el cuaderno).
- `epochs`: cuántas veces se recorre todo el conjunto.

**g) Entrenamiento vs. validación vs. prueba.** El cuaderno solo reportaba la exactitud de *entrenamiento* (97.97 %), que es optimista porque el modelo ya vio esas imágenes. En `entrenar_modelo.py`:
- se aparta un 10 % para *validación* (vigilar durante el entrenamiento),
- y al final se evalúa con las 10 000 de *prueba*, nunca vistas.

En mis corridas: entrenamiento ≈ 99.4 %, validación ≈ 97.6 %, prueba ≈ 97–97.5 %. Esa brecha es **sobreajuste**: el modelo empieza a memorizar el conjunto de entrenamiento en lugar de generalizar.

**h) Guardar el modelo.** `Model.save("modelo_mnist.keras")` guarda arquitectura y pesos. Después, `tf.keras.models.load_model(...)` lo recupera sin reentrenar. Usar un modelo ya entrenado se llama **inferencia**.

### 3.4 Por qué hace falta preprocesar tus imágenes

El modelo solo conoce imágenes con el formato de MNIST. Una foto de tu cuaderno es lo opuesto: tinta **oscura** sobre papel **claro**, grande, descentrada y con sombras. Si se le pasa tal cual (reducida a 28×28), falla casi siempre. `preprocesamiento.py` replica cómo se construyó MNIST:

1. Estima el color del papel (incluida la sombra) y lo resta: queda tinta clara sobre fondo negro.
2. Umbral de Otsu: separa tinta de ruido eligiendo el valor que maximiza la varianza entre las dos clases.
3. Recorta la caja que contiene el trazo.
4. Engrosa trazos muy finos (en MNIST la tinta ocupa ~30–55 % de la caja del dígito).
5. Escala para que el lado mayor mida 20 px, conservando la proporción.
6. Pega en un lienzo negro de 28×28 y centra por **centro de masa**, como en MNIST.
7. Normaliza a [0, 1], igual que en el entrenamiento.

La interfaz te muestra "Lo que ve el modelo (28×28)": si esa imagen no parece un dígito MNIST, la predicción no será confiable.

**Resultados de las pruebas** (imágenes MNIST de prueba convertidas en "fotos" con papel gris, sombra lateral, ruido y posición aleatoria):

| Caso | Exactitud |
|---|---|
| MNIST directo (referencia) | 98 % |
| Fotos simuladas, trazo normal | 97 % |
| Fotos simuladas, trazo muy fino | 95 % |
| Dígitos dibujados en el lienzo | correctos |

### 3.5 Consejos para tus fotos

- Escribe con **marcador o lapicero grueso**, un solo dígito por foto.
- Papel liso (sin cuadrícula ni renglones si es posible), buena luz y sin otros objetos en el encuadre.
- Recorta la foto alrededor del número antes de arrastrarla.
- Un 4, 7 o 1 escrito "a la europea" (con rayita) puede confundir: MNIST viene de escritura estadounidense.

### 3.6 Problemas comunes

| Síntoma | Causa y solución |
|---|---|
| `Illegal instruction (core dumped)` al importar TensorFlow | La VM no expone instrucciones AVX del procesador. En VirtualBox sobre Windows suele pasar si Hyper-V está activo. Desactivarlo o usar Colab para entrenar (el `.keras` se puede copiar a la VM). |
| `externally-managed-environment` | Falta activar el `.venv` (ver 3.2). |
| No puedo arrastrar archivos desde Windows/Mac a la VM | Activa "Arrastrar y soltar: bidireccional" en VirtualBox (requiere Guest Additions), usa una carpeta compartida, o abre la app desde el navegador del equipo anfitrión: cambia la última línea de `app.py` a `demo.launch(server_name="0.0.0.0")` y entra a `http://IP-de-la-VM:7860`. |
| `Address already in use` | Ya hay otra app en el puerto 7860: ciérrala o usa `demo.launch(server_port=7861)`. |

---

## Parte 4 — Prompt sugerido para el agente

Copia la carpeta `proyecto_mnist` en tu VM, abre el agente dentro de ella y pega:

```
Estoy aprendiendo machine learning. En esta carpeta hay un proyecto que entrena una red
densa con MNIST (entrenar_modelo.py), la guarda en modelo_mnist.keras y la usa en una
interfaz Gradio (app.py) para reconocer dígitos de imágenes que yo arrastro.

1. Lee GUIA.md y los archivos .py antes de hacer nada.
2. Crea un entorno virtual .venv e instala requirements.txt.
3. Ejecuta entrenar_modelo.py y explícame la salida: pérdida, exactitud de entrenamiento,
   validación y prueba, y si hay sobreajuste.
4. Ejecuta app.py y dime la dirección para abrirla.
5. Antes de cada comando, explícame en una o dos frases qué hace y por qué.
No modifiques la arquitectura del modelo sin preguntarme.
```

Como reto opcional (para comparar después), puedes pedirle al agente: *"Crea entrenar_cnn.py con una red convolucional pequeña, compara su exactitud con la red densa en mis propias fotos y explícame por qué la convolucional es más robusta a desplazamientos."*
