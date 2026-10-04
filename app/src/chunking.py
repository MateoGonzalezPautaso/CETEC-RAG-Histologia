"""
División del texto de una página en chunks para indexar.

Antes cada página se cortaba cada 500 caracteres, sin mirar el contenido: el
corte caía a mitad de oración o de lista ("Por el número de prolongaciones
p" | "ueden ser: unipolares...") y la respuesta quedaba repartida entre dos
chunks, sin las palabras de la consulta en el que la contenía.

Ahora el texto se agrupa en segmentos (una oración, un ítem de lista, una
línea de ficha como "Laminilla No:") y los chunks se arman con segmentos
enteros:

- Cada chunk apunta a CHUNK_OBJETIVO caracteres y nunca pasa de CHUNK_MAX.
  Uno de menos de CHUNK_MIN caracteres no se cierra: se le suma el segmento
  siguiente (un título suelto no sirve como chunk).
- El último segmento de un chunk se repite al principio del siguiente
  (solapamiento) si mide hasta CHUNK_SOLAPAMIENTO caracteres, para que una
  definición no quede sin su encabezado. Si el chunk venía de una ficha
  ("Imagen 1: Arteria muscular..."), el siguiente también empieza con ese
  título: la descripción de la foto no queda sin el nombre de lo que muestra.
- "Imagen N" y "Práctica N" empiezan un chunk nuevo: así la ficha de una
  imagen o el título de una práctica no se mezclan con el texto anterior. La
  descripción de la foto ("Foto 1: ...") queda con la ficha de su imagen.
"""

import re
from typing import List

# MiniLM lee hasta 256 tokens (unos 750 caracteres de español): un chunk más
# largo quedaría representado solo por su principio.
CHUNK_OBJETIVO = 650
CHUNK_MIN = 300
CHUNK_MAX = 850
CHUNK_SOLAPAMIENTO = 250
# Cambia cuando cambia la forma de dividir: la indexación lo usa para saber si
# los chunks guardados en Qdrant son de otra versión y hay que rehacerlos.
CHUNKING_VERSION = "2-oraciones"

# Glifos de viñeta que PyMuPDF devuelve solos en una línea (Symbol/Wingdings).
_VINETAS = {"\uf0b7", "\uf0a7", "\uf0d8", "\u2022", "\u25cf", "\u25cb", "\u25aa", "\u25a0", "\u25e6", "-", "o"}
_INICIO_ITEM = re.compile(r"^(?:[a-z]\)|\d{1,2}[.)]\s|[\u2022\u25cf\u25cb\u25aa\u25a0\u25e6\uf0b7\uf0a7]\s*)")
_INICIO_BLOQUE = re.compile(r"^(?:Imagen|Práctica|Practica)\s*\d", re.IGNORECASE)
_FIN_SEGMENTO = (".", ":", ";", "?", "!")
_FIN_ORACION = re.compile(r"(?<=[.!?])\s+")


def _segmentos(texto: str) -> List[str]:
    """Agrupa las líneas de una página en segmentos que no conviene partir."""
    segmentos: List[str] = []
    actual = ""
    vineta_pendiente = False
    for cruda in texto.splitlines():
        linea = cruda.strip()
        if not linea:
            continue
        if linea in _VINETAS:
            # La viñeta viene sola en su línea: el ítem empieza en la siguiente.
            if actual:
                segmentos.append(actual)
            actual = ""
            vineta_pendiente = True
            continue
        nuevo = (
            not actual
            or vineta_pendiente
            or actual.endswith(_FIN_SEGMENTO)
            or _INICIO_ITEM.match(linea)
            or _INICIO_BLOQUE.match(linea)
        )
        if nuevo:
            if actual:
                segmentos.append(actual)
            actual = linea
        elif actual.endswith("-") and linea[:1].islower():
            actual = actual[:-1] + linea  # "fractu-" + "rado"
        else:
            actual = f"{actual} {linea}"
        vineta_pendiente = False
    if actual:
        segmentos.append(actual)
    return segmentos


def _partir_largo(segmento: str, maximo: int) -> List[str]:
    """Parte un segmento más largo que `maximo` por oraciones y, si hace falta, por palabras."""
    partes: List[str] = []
    actual = ""
    for oracion in _FIN_ORACION.split(segmento):
        if len(oracion) > maximo:
            palabras = oracion.split(" ")
            oracion = ""
            for palabra in palabras:
                if len(oracion) + len(palabra) + 1 > maximo and oracion:
                    if actual:
                        partes.append(actual)
                        actual = ""
                    partes.append(oracion)
                    oracion = palabra
                else:
                    oracion = f"{oracion} {palabra}" if oracion else palabra
        if actual and len(actual) + len(oracion) + 1 > maximo:
            partes.append(actual)
            actual = oracion
        else:
            actual = f"{actual} {oracion}" if actual else oracion
    if actual:
        partes.append(actual)
    return partes


def dividir_en_chunks(
    texto: str,
    objetivo: int = CHUNK_OBJETIVO,
    maximo: int = CHUNK_MAX,
    solapamiento: int = CHUNK_SOLAPAMIENTO,
    minimo: int = CHUNK_MIN,
) -> List[str]:
    """Chunks de una página, armados con segmentos enteros (ver docstring del módulo)."""
    segmentos: List[str] = []
    for seg in _segmentos(texto):
        segmentos.extend(_partir_largo(seg, maximo) if len(seg) > maximo else [seg])

    chunks: List[str] = []
    actual: List[str] = []
    largo = 0
    solo_solape = False  # el chunk en curso tiene solo segmentos repetidos
    titulo = ""  # título de la ficha o práctica en curso ("Imagen 1: ...")

    def cerrar():
        nonlocal actual, largo, solo_solape
        if actual and not solo_solape:
            chunks.append("\n".join(actual))
        actual, largo, solo_solape = [], 0, False

    for seg in segmentos:
        entra = largo + len(seg) + 1 <= maximo
        if _INICIO_BLOQUE.match(seg):
            if largo >= minimo or not entra:
                cerrar()
            titulo = seg if len(seg) <= solapamiento else ""
        elif actual and largo + len(seg) + 1 > objetivo and (largo >= minimo or not entra):
            ultimo = actual[-1]
            cerrar()
            repetir = [titulo] if titulo else []
            if len(ultimo) <= solapamiento and ultimo != titulo:
                repetir.append(ultimo)
            while repetir and sum(len(r) + 1 for r in repetir) + len(seg) > maximo:
                repetir.pop()
            if repetir:
                actual, solo_solape = repetir, True
                largo = sum(len(r) for r in repetir) + len(repetir) - 1
        actual.append(seg)
        largo += len(seg) + (1 if largo else 0)
        solo_solape = False
    cerrar()
    return chunks
