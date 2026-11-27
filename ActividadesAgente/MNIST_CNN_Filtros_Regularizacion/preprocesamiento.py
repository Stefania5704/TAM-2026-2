"""
PASO 2 — Convertir una imagen "del mundo real" al formato que el modelo conoce.

El modelo SOLO vio imágenes MNIST durante el entrenamiento:
  * 28x28 píxeles, escala de grises
  * dígito CLARO sobre fondo OSCURO
  * el dígito ocupa una caja de ~20x20 en el centro (centrado por su "centro de masa")
  * valores normalizados a [0, 1]

Una foto de tu cuaderno es lo contrario (tinta oscura sobre papel claro, grande,
descentrada). Si se la pasas tal cual, el modelo se equivoca casi siempre.
Esta función replica el procedimiento con el que se construyó MNIST.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageOps


def _a_gris(img):
    """Cualquier imagen (RGB, RGBA, gris) -> PIL en escala de grises.
    Si tiene transparencia (p. ej. el lienzo de dibujo), se pinta sobre fondo blanco."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        fondo = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(fondo, img)
    return img.convert("L")


def _umbral_otsu(a):
    """Umbral de Otsu: el valor que mejor separa 'tinta' de 'fondo' en el histograma."""
    hist = np.bincount(a.ravel(), minlength=256).astype(float)
    total = a.size
    suma_total = np.dot(np.arange(256), hist)
    peso_f = suma_f = 0.0
    mejor, umbral = -1.0, 128
    for t in range(256):
        peso_f += hist[t]
        if peso_f == 0 or peso_f == total:
            continue
        suma_f += t * hist[t]
        m_f = suma_f / peso_f
        m_d = (suma_total - suma_f) / (total - peso_f)
        var = peso_f * (total - peso_f) * (m_f - m_d) ** 2
        if var > mejor:
            mejor, umbral = var, t
    return umbral


def preparar_imagen(img):
    """
    Recibe una imagen PIL o un arreglo numpy.
    Devuelve:
        vector  -> arreglo (1, 784) float32 en [0, 1], listo para Model.predict
        vista   -> arreglo (28, 28) uint8, lo que "ve" el modelo (para mostrarlo)
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = ImageOps.exif_transpose(img)          # respeta la rotación de fotos de celular
    gris = _a_gris(img)

    # Reducir fotos enormes para que todo sea rápido
    gris.thumbnail((600, 600))
    a = np.asarray(gris).astype(np.uint8)

    # 1) Estimar el fondo (el papel) y restarlo. En una foto real la luz no es
    #    pareja: un lado del papel sale más oscuro. Si no se corrige, esa sombra
    #    se confunde con tinta. Truco: en una versión pequeña de la imagen, un
    #    filtro de máximo "borra" los trazos oscuros y deja solo el papel.
    #    Se mira el borde de la imagen para saber si el fondo es claro u oscuro.
    borde = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
    fondo_claro = borde.mean() > 127
    H, W = a.shape
    peq = gris.resize((max(1, W // 8), max(1, H // 8)), Image.BILINEAR)
    peq = peq.filter(ImageFilter.MaxFilter(5) if fondo_claro else ImageFilter.MinFilter(5))
    fondo = np.asarray(peq.filter(ImageFilter.GaussianBlur(2)).resize((W, H), Image.BILINEAR), float)
    # 2) Tinta = diferencia con el papel. Queda trazo CLARO sobre fondo NEGRO, como MNIST.
    tinta = (fondo - a) if fondo_claro else (a - fondo)
    a = np.clip(tinta, 0, 255).astype(np.uint8)

    # 3) Umbral de Otsu: separa tinta de lo que queda de ruido/textura del papel
    t = max(_umbral_otsu(a), int(0.2 * a.max()))
    a = np.where(a > t, a, 0).astype(np.uint8)
    if a.max() == 0:
        vacia = np.zeros((28, 28), np.uint8)
        return vacia.reshape(1, 784).astype("float32"), vacia

    # 4) Recortar a la caja que contiene el trazo
    filas = np.where(a.max(axis=1) > 0)[0]
    cols = np.where(a.max(axis=0) > 0)[0]
    a = a[filas[0]:filas[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = a.shape

    # 5) Engrosar trazos finos. En MNIST la tinta ocupa ~30-55% de la caja del dígito
    #    (trazo grueso). Un lapicero en una foto grande queda muy delgado y, al
    #    reducir a 20x20, casi desaparecería. Se "dilata" el trazo hasta llegar a ~30%.
    recorte = Image.fromarray(a)
    k = max(3, int(round(max(h, w) / 40)) | 1)        # tamaño del filtro (impar)
    for _ in range(12):
        if (np.asarray(recorte) > 0).mean() >= 0.30:
            break
        recorte = recorte.filter(ImageFilter.MaxFilter(k))
    a = np.asarray(recorte)
    filas = np.where(a.max(axis=1) > 0)[0]
    cols = np.where(a.max(axis=0) > 0)[0]
    a = a[filas[0]:filas[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = a.shape
    recorte = Image.fromarray(a)

    # 6) Escalar para que el lado mayor mida 20 px, conservando la proporción
    escala = 20.0 / max(h, w)
    nw, nh = max(1, round(w * escala)), max(1, round(h * escala))
    recorte = recorte.resize((nw, nh), Image.LANCZOS)

    # 7) Pegar en un lienzo negro de 28x28
    lienzo = np.zeros((28, 28), np.float32)
    y0, x0 = (28 - nh) // 2, (28 - nw) // 2
    lienzo[y0:y0 + nh, x0:x0 + nw] = np.asarray(recorte, np.float32)

    # 8) Centrar por centro de masa (así se centró MNIST originalmente)
    ys, xs = np.indices(lienzo.shape)
    masa = lienzo.sum()
    cy, cx = (ys * lienzo).sum() / masa, (xs * lienzo).sum() / masa
    dy, dx = int(round(14 - cy)), int(round(14 - cx))
    lienzo = np.roll(lienzo, (dy, dx), axis=(0, 1))

    # 9) Contraste: que el trazo más intenso valga 255, como en MNIST
    lienzo = lienzo / lienzo.max() * 255.0
    vista = lienzo.astype(np.uint8)

    # 10) Aplanar y normalizar EXACTAMENTE igual que en el entrenamiento
    vector = (lienzo / 255.0).reshape(1, 784).astype("float32")
    return vector, vista
