"""
Puntaje léxico BM25 sobre los chunks del manual.

La búsqueda por embeddings (MiniLM) confunde chunks que comparten vocabulario
general ("tejido", "arteria", "células") y no distingue bien un término
puntual: "osteona", "laringe 43" o "lámina elástica interna" quedaban fuera de
los 10 primeros. BM25 pesa cada término por su rareza en el corpus (IDF), así
que una palabra que aparece en pocos chunks decide el orden y una que aparece
en todos casi no suma.

El corpus son unos cien chunks: el índice se arma en memoria en milisegundos y
se rehace cuando cambia la colección.
"""

import math
import re
from typing import Dict, List

from .config import normalizar

# Palabras que no distinguen un chunk de otro (ya sin tildes).
_STOP = set("""
a al algo algun alguna algunas algunos ante como con cual cuales cuyo cuya de del desde donde
e el ella ellas ellos en entre es esa ese eso esta estan estas este esto estos fue ha hacia
hasta hay la las le les lo los mas muy no o para pero por que se segun ser si sin sobre son
su sus tambien tiene tienen un una unas uno unos y ya
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def _raiz(palabra: str) -> str:
    """Quita el plural para que "células" y "célula" cuenten como el mismo término."""
    if len(palabra) > 4 and palabra.endswith("es") and not palabra.endswith("ses"):
        return palabra[:-2]
    if len(palabra) > 3 and palabra.endswith("s"):
        return palabra[:-1]
    return palabra


def tokenizar(texto: str) -> List[str]:
    return [
        _raiz(p) for p in _TOKEN.findall(normalizar(texto))
        if p not in _STOP and (len(p) > 2 or p.isdigit())
    ]


class IndiceBM25:
    def __init__(self, textos: List[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenizar(t) for t in textos]
        self.largos = [len(d) for d in self.docs]
        self.largo_medio = (sum(self.largos) / len(self.docs)) if self.docs else 0.0
        self.frecuencias: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        for doc in self.docs:
            tf: Dict[str, int] = {}
            for termino in doc:
                tf[termino] = tf.get(termino, 0) + 1
            self.frecuencias.append(tf)
            for termino in tf:
                df[termino] = df.get(termino, 0) + 1
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def puntajes(self, consulta: str) -> List[float]:
        """Puntaje BM25 de cada chunk para la consulta (0 si no comparte ningún término)."""
        terminos = set(tokenizar(consulta)) & self.idf.keys()
        salida = []
        for tf, largo in zip(self.frecuencias, self.largos):
            s = 0.0
            for t in terminos:
                f = tf.get(t)
                if f:
                    norma = self.k1 * (1 - self.b + self.b * largo / (self.largo_medio or 1))
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norma)
            salida.append(s)
        return salida
