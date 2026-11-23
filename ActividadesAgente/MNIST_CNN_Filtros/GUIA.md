# Guía paso a paso (VM Ubuntu)

Esta guía te lleva desde el archivo `.zip` hasta la app funcionando en `localhost:8501` y la carpeta subida a GitHub. Haz los pasos en orden; cada uno dice qué hace cada comando.

> **Analogía general.** Piensa en tres "cajones" distintos:
> 1. `~/proyecto_mnist` es tu proyecto anterior (no lo tocamos, solo usamos su entorno virtual).
> 2. `~/proyecto_mnist_cnn` es el **taller** de esta tarea: aquí entrenas y pruebas.
> 3. `~/TAM-2026-2/ActividadesAgente/MNIST_CNN_Filtros` es la **vitrina**: la copia que se entrega en GitHub.

---

## Paso 1. Descargar el proyecto dentro de la VM

1. Dentro de la VM abre Firefox, entra a claude.ai, abre este mismo chat y descarga `MNIST_CNN_Filtros.zip`.
2. Abre una terminal (`Ctrl + Alt + T`) y busca dónde quedó:

```bash
ls ~/Descargas ~/Downloads
```

- `ls` lista lo que hay en una carpeta. Le damos las dos porque, según el idioma de Ubuntu, la carpeta se llama `Descargas` o `Downloads`. Una de las dos mostrará un error de "no existe"; es normal.

## Paso 2. Crear la carpeta del taller y descomprimir

```bash
cd ~
python3 -m zipfile -e ~/Descargas/MNIST_CNN_Filtros.zip ~/
mv ~/MNIST_CNN_Filtros ~/proyecto_mnist_cnn
cd ~/proyecto_mnist_cnn
ls -a
```

- `cd ~` te lleva a tu carpeta personal (`/home/vboxuser`). La `~` es un atajo para esa carpeta.
- `python3 -m zipfile -e ARCHIVO DESTINO` descomprime el zip usando Python (así no dependes de que `unzip` esté instalado). Si tu zip quedó en `~/Downloads`, cambia `Descargas` por `Downloads`.
- `mv ORIGEN DESTINO` mueve o **renombra**. Aquí renombra la carpeta a `proyecto_mnist_cnn`, separada de tu proyecto anterior.
- `ls -a` lista todo, incluso lo oculto (lo que empieza por punto). Deberías ver `.streamlit`, `app.py`, `entrenar_modelos.py`, etc.

## Paso 3. Activar el entorno virtual que ya tienes

No hace falta instalar TensorFlow otra vez: reutilizamos el `.venv` de tu proyecto anterior, que ya lo tiene.

```bash
source ~/proyecto_mnist/.venv/bin/activate
pip install matplotlib altair
```

- `source .../activate` "enciende" el entorno virtual. Es como entrar a un cuarto donde están guardadas tus librerías de Python (TensorFlow, Streamlit…). Sabrás que funcionó porque la línea de la terminal empieza con `(.venv)`.
- `pip install matplotlib altair` agrega las dos librerías de gráficas que usa esta app. Altair probablemente ya venía con Streamlit; si es así, pip solo dirá "Requirement already satisfied".

> Cada vez que abras una terminal nueva tendrás que repetir el `source`. El entorno no queda encendido para siempre.

## Paso 4. Entrenar las dos redes (una sola vez)

```bash
cd ~/proyecto_mnist_cnn
python entrenar_modelos.py
```

- Entrena primero la red densa y luego la convolucional, 6 épocas cada una. En la VM puede tardar unos 5 a 10 minutos (la convolucional es la lenta).
- Verás por cada época algo como `accuracy: 0.98 - val_accuracy: 0.98`. Es la nota de la red en cada vuelta al conjunto de datos.
- Al final imprime la exactitud de cada red y crea 4 archivos en `modelos/`.
- Si tienes afán: `python entrenar_modelos.py --epocas 3` (más rápido, un poco menos preciso).

## Paso 5. Abrir la app en vivo

```bash
streamlit run app.py
```

- Streamlit levanta un pequeño servidor web en tu VM. Abre Firefox en `http://localhost:8501` (normalmente se abre solo).
- `localhost` significa "este mismo computador"; `8501` es la "puerta" por donde Streamlit atiende.
- Mientras la app esté abierta, la terminal queda ocupada. Para apagarla: `Ctrl + C` en la terminal.
- Si cambias algo en el código, guarda el archivo y en la página pulsa **Rerun** (arriba a la derecha) o la tecla `R`.

**Recorrido sugerido:** en la barra lateral elige un dígito, mira la pestaña *Predicción*, luego *Red convolucional, capa por capa* y cambia entre Capa 1, 2 y 3. Para tus fotos, guárdalas en `~/proyecto_mnist_cnn/mis_imagenes/` y elige *Carpeta del proyecto* (útil porque el botón de subir de Firefox no funcionaba en la VM).

## Paso 6. Crear la subcarpeta en el repositorio y subirla a GitHub

Abre **otra** terminal (la primera está ocupada con Streamlit) y copia el taller a la vitrina:

```bash
cd ~/TAM-2026-2
git pull
mkdir -p ActividadesAgente/MNIST_CNN_Filtros
cp -r ~/proyecto_mnist_cnn/. ActividadesAgente/MNIST_CNN_Filtros/
git status
```

- `git pull` trae los últimos cambios de GitHub, para que tu copia local esté al día antes de agregar algo.
- `mkdir -p` crea la carpeta (y no se queja si ya existe).
- `cp -r ORIGEN/. DESTINO/` copia **todo** el contenido, incluidos los archivos ocultos como `.streamlit` y `.gitignore`. El `/.` al final significa "lo que hay adentro".
- `git status` muestra qué archivos son nuevos. Debe aparecer `ActividadesAgente/MNIST_CNN_Filtros/` en rojo (nuevo, aún sin guardar en git). Las carpetas `__pycache__` no aparecen porque el `.gitignore` las excluye.

Ahora guarda y sube:

```bash
git add ActividadesAgente/MNIST_CNN_Filtros
git commit -m "Agrega app de red densa vs. convolucional con visualización de filtros"
git push
```

- `git add` pone los archivos "en la caja" para el próximo guardado.
- `git commit -m "..."` cierra la caja con una etiqueta (el mensaje). Es una foto del proyecto en ese momento.
- `git push` envía esa foto a GitHub. Al recargar la página del repositorio verás la nueva carpeta junto a `MNIST_Reconocedor`.

Los modelos entrenados (`modelos/*.keras`, unos 2 MB en total) se suben también, así quien clone el repositorio puede abrir la app sin entrenar.

---

## Si algo falla

| Mensaje | Qué significa y qué hacer |
|---|---|
| `No module named 'matplotlib'` | El entorno virtual no está activo o falta instalar: repite el paso 3. |
| La app dice "Falta entrenar las redes" | No existe `modelos/`: corre el paso 4 desde `~/proyecto_mnist_cnn`. |
| `TypeError ... unexpected keyword argument 'width'` | Tu Streamlit es antiguo: `pip install --upgrade streamlit`. |
| `Port 8501 is already in use` | Ya hay otra app abierta (por ejemplo la anterior). Ciérrala con `Ctrl + C` o usa `streamlit run app.py --server.port 8502`. |
| `git push` pide usuario y contraseña | Usa el mismo método con el que subiste `MNIST_Reconocedor` (token de GitHub o la sesión de Claude Code). |
| Las letras de la página se ven distintas a las de la captura | Las tipografías se descargan de Google Fonts; sin internet se usa una de respaldo. Todo lo demás funciona igual. |
