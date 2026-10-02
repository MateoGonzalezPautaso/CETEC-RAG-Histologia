"""
Rotación de API keys gratuitas de Groq.

Con varias keys (una por integrante del grupo) en GROQ_API_KEYS, cada llamada
usa la siguiente key disponible (round-robin). Si una key devuelve un límite de
uso (429), queda en cooldown y la llamada se reintenta con la próxima:

- límite por minuto (TPM/RPM): cooldown corto (GROQ_COOLDOWN_MINUTO_S, 60 s).
- límite diario (TPD/RPD): cooldown largo (GROQ_COOLDOWN_DIARIO_S, 1 h).

Si todas las keys están en cooldown se usa la que se libera antes y el error
sigue su curso normal (src/llm.py decide si reintenta o corta por cuota).

Con una sola key (GROQ_API_KEY) el comportamiento es el mismo de antes.
"""

import os
import threading
import time
from typing import Callable, List, Optional


def _es_limite(error: Exception) -> bool:
    # status_code: SDK de Groq/LangChain; status: errores de DSPy (LMRateLimitError).
    if 429 in (getattr(error, "status_code", None), getattr(error, "status", None)):
        return True
    if "ratelimit" in type(error).__name__.lower():
        return True
    raw = str(error).lower()
    return any(token in raw for token in ["429", "rate limit", "rate_limit", "too many requests"])


def _es_limite_diario(error: Exception) -> bool:
    raw = str(error).lower()
    return any(token in raw for token in ["per day", "tpd", "rpd", "daily"])


def _sufijo(clave: str) -> str:
    return f"...{clave[-4:]}" if len(clave) > 4 else "..."


class RotadorClaves:
    """Pool de keys con round-robin y cooldown por key. Es thread-safe."""

    def __init__(self, claves: List[str], cooldown_minuto: float = 60.0, cooldown_diario: float = 3600.0):
        # Sin duplicados y en el orden dado.
        self.claves = list(dict.fromkeys(c.strip() for c in claves if c and c.strip()))
        self.cooldown_minuto = cooldown_minuto
        self.cooldown_diario = cooldown_diario
        self._libre_desde = {}
        self._siguiente = 0
        self._lock = threading.Lock()

    @classmethod
    def desde_entorno(cls, var: str = "GROQ_API_KEYS", respaldo: str = "GROQ_API_KEY") -> "RotadorClaves":
        claves = os.getenv(var, "").split(",")
        claves.append(os.getenv(respaldo, ""))
        return cls(
            claves,
            cooldown_minuto=float(os.getenv("GROQ_COOLDOWN_MINUTO_S", "60")),
            cooldown_diario=float(os.getenv("GROQ_COOLDOWN_DIARIO_S", "3600")),
        )

    def __len__(self) -> int:
        return len(self.claves)

    def __deepcopy__(self, memo):
        # DSPy copia los LM con deepcopy: el pool (y su estado) se comparte.
        return self

    def orden_de_intento(self) -> List[str]:
        """Keys disponibles empezando por la que toca; si no hay, la que se libera antes."""
        with self._lock:
            if not self.claves:
                return []
            ahora = time.time()
            n = len(self.claves)
            inicio = self._siguiente
            self._siguiente = (inicio + 1) % n
            rotadas = [self.claves[(inicio + i) % n] for i in range(n)]
            libres = [c for c in rotadas if self._libre_desde.get(c, 0) <= ahora]
            if libres:
                return libres
            return [min(rotadas, key=lambda c: self._libre_desde.get(c, 0))]

    def reportar_limite(self, clave: str, error: Exception) -> None:
        diario = _es_limite_diario(error)
        espera = self.cooldown_diario if diario else self.cooldown_minuto
        with self._lock:
            self._libre_desde[clave] = time.time() + espera
        tipo = "diario" if diario else "por minuto"
        print(f"   🔑 Key {_sufijo(clave)} con límite {tipo} — en pausa {int(espera)}s")

    def ejecutar(self, llamada: Callable[[str], object]):
        """Ejecuta llamada(clave) rotando ante límites de uso."""
        ultimo: Optional[Exception] = None
        for clave in self.orden_de_intento():
            try:
                return llamada(clave)
            except Exception as e:
                if not _es_limite(e):
                    raise
                self.reportar_limite(clave, e)
                ultimo = e
        if ultimo is None:
            raise RuntimeError("No hay API keys de Groq configuradas (GROQ_API_KEYS o GROQ_API_KEY).")
        raise ultimo

    async def aejecutar(self, llamada):
        """Versión async: llamada(clave) devuelve un awaitable."""
        ultimo: Optional[Exception] = None
        for clave in self.orden_de_intento():
            try:
                return await llamada(clave)
            except Exception as e:
                if not _es_limite(e):
                    raise
                self.reportar_limite(clave, e)
                ultimo = e
        if ultimo is None:
            raise RuntimeError("No hay API keys de Groq configuradas (GROQ_API_KEYS o GROQ_API_KEY).")
        raise ultimo


class ChatRotativo:
    """
    Envuelve un chat model de LangChain (uno por key) y rota las keys en
    invoke/ainvoke. Expone model_name para las trazas.
    """

    def __init__(self, fabrica: Callable[[str], object], rotador: RotadorClaves):
        self.rotador = rotador
        self._modelos = {clave: fabrica(clave) for clave in rotador.claves}
        primero = next(iter(self._modelos.values()), None)
        self.model_name = getattr(primero, "model_name", None)

    def invoke(self, messages, **kwargs):
        return self.rotador.ejecutar(lambda clave: self._modelos[clave].invoke(messages, **kwargs))

    async def ainvoke(self, messages, **kwargs):
        return await self.rotador.aejecutar(lambda clave: self._modelos[clave].ainvoke(messages, **kwargs))


def opciones_razonamiento(modelo: str) -> dict:
    """
    Qwen en Groq razona antes de responder ("thinking"). Para el RAG no hace
    falta y gasta cuota: por defecto se desactiva (LLM_REASONING_EFFORT=none) y,
    si igual razona, el razonamiento no se incluye en la respuesta
    (LLM_REASONING_FORMAT=hidden). Un valor vacío no envía el parámetro.
    """
    if "qwen" not in modelo.lower():  # "qwen/..." (LangChain) o "groq/qwen/..." (DSPy)
        return {}
    opciones = {}
    formato = os.getenv("LLM_REASONING_FORMAT", "hidden").strip()
    esfuerzo = os.getenv("LLM_REASONING_EFFORT", "none").strip()
    if formato:
        opciones["reasoning_format"] = formato
    if esfuerzo:
        opciones["reasoning_effort"] = esfuerzo
    return opciones


def crear_chat_groq(modelo: str, temperature: float = 0, max_retries: int = 1):
    """ChatGroq con rotación de keys (GROQ_API_KEYS) o una sola key (GROQ_API_KEY)."""
    from langchain_groq import ChatGroq

    opciones = opciones_razonamiento(modelo)

    def fabrica(clave):
        return ChatGroq(model=modelo, api_key=clave, temperature=temperature, max_retries=max_retries, **opciones)

    rotador = RotadorClaves.desde_entorno()
    if len(rotador) <= 1:
        return fabrica(rotador.claves[0] if rotador.claves else None)
    print(f"🔑 Rotación de keys de Groq: {len(rotador)} keys")
    return ChatRotativo(fabrica, rotador)
