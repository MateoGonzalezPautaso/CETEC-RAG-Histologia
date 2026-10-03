"""
Rotación de API keys gratuitas de Groq.

Con varias keys (una por integrante del grupo) en GROQ_API_KEYS, cada llamada
usa la siguiente key disponible (round-robin). Si una key devuelve un límite de
uso (429), queda en pausa el tiempo que indica Groq ("Please try again in
7.5s") y la llamada sigue con la próxima. Sin ese dato:

- límite por minuto (TPM/RPM): pausa corta (GROQ_COOLDOWN_MINUTO_S, 60 s).
- límite diario (TPD/RPD): pausa larga (GROQ_COOLDOWN_DIARIO_S, 1 h).

Si todas las keys están en pausa y la primera se libera en menos de
GROQ_ESPERA_MAX_S (90 s), se espera y se reintenta: así un límite por minuto
no termina en error. Si la espera es más larga (límite diario), el error sigue
su curso (src/llm.py decide si reintenta o corta por cuota).

Con una sola key (GROQ_API_KEY) se aplica lo mismo sobre esa key.
"""

import asyncio
import os
import re
import threading
import time
from typing import Callable, List, Optional


def _es_limite(error: Exception) -> bool:
    # 413 "Request too large": el pedido excede el límite por minuto de
    # cualquier key; rotar no sirve y dejaría todas las keys en pausa.
    if 413 in (getattr(error, "status_code", None), getattr(error, "status", None)):
        return False
    if "request too large" in str(error).lower():
        return False
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


_UNIDADES = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}


def _espera_sugerida(error: Exception) -> Optional[float]:
    """Segundos de "Please try again in 1h2m3.5s" / "in 820ms" del mensaje de Groq."""
    m = re.search(r"try again in ((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", str(error))
    if not m:
        return None
    return sum(float(v) * _UNIDADES[u] for v, u in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1)))


def _sufijo(clave: str) -> str:
    return f"...{clave[-4:]}" if len(clave) > 4 else "..."


class RotadorClaves:
    """Pool de keys con round-robin y pausa por key. Es thread-safe."""

    def __init__(self, claves: List[str], cooldown_minuto: float = 60.0, cooldown_diario: float = 3600.0,
                 espera_max: float = 90.0, rondas: int = 4):
        # Sin duplicados y en el orden dado.
        self.claves = list(dict.fromkeys(c.strip() for c in claves if c and c.strip()))
        self.cooldown_minuto = cooldown_minuto
        self.cooldown_diario = cooldown_diario
        self.espera_max = espera_max
        self.rondas = rondas
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
            espera_max=float(os.getenv("GROQ_ESPERA_MAX_S", "90")),
        )

    def __len__(self) -> int:
        return len(self.claves)

    def __deepcopy__(self, memo):
        # DSPy copia los LM con deepcopy: el pool (y su estado) se comparte.
        return self

    def orden_de_intento(self) -> List[str]:
        """Keys disponibles empezando por la que toca (vacío si todas están en pausa)."""
        with self._lock:
            if not self.claves:
                return []
            ahora = time.time()
            n = len(self.claves)
            inicio = self._siguiente
            self._siguiente = (inicio + 1) % n
            rotadas = [self.claves[(inicio + i) % n] for i in range(n)]
            return [c for c in rotadas if self._libre_desde.get(c, 0) <= ahora]

    def _proxima_libre(self) -> tuple:
        """(key, segundos hasta que se libera) de la key que se libera antes."""
        with self._lock:
            clave = min(self.claves, key=lambda c: self._libre_desde.get(c, 0))
            return clave, max(0.0, self._libre_desde.get(clave, 0) - time.time())

    def reportar_limite(self, clave: str, error: Exception) -> None:
        diario = _es_limite_diario(error)
        sugerida = _espera_sugerida(error)
        if sugerida is not None:
            espera = sugerida + 0.5
        else:
            espera = self.cooldown_diario if diario else self.cooldown_minuto
        with self._lock:
            self._libre_desde[clave] = time.time() + espera
        tipo = "diario" if diario else "por minuto"
        print(f"   🔑 Key {_sufijo(clave)} con límite {tipo} — en pausa {espera:.0f}s")

    def _plan(self, ronda: int, intento_hecho: bool):
        """Qué hacer cuando no quedan keys libres: (esperar_s, clave_forzada) o None para cortar."""
        clave, espera = self._proxima_libre()
        if espera <= self.espera_max and ronda < self.rondas - 1:
            return espera, None
        if not intento_hecho:
            # Todas venían en pausa larga (estimada): se prueba la que se libera
            # antes por si la cuota ya se renovó.
            return 0.0, clave
        return None

    def ejecutar(self, llamada: Callable[[str], object]):
        """Ejecuta llamada(clave) rotando ante límites de uso."""
        if not self.claves:
            raise RuntimeError("No hay API keys de Groq configuradas (GROQ_API_KEYS o GROQ_API_KEY).")
        ultimo: Optional[Exception] = None
        for ronda in range(self.rondas):
            claves = self.orden_de_intento()
            if not claves:
                plan = self._plan(ronda, ultimo is not None)
                if plan is None:
                    break
                espera, forzada = plan
                if espera:
                    print(f"   ⏳ Todas las keys en pausa: espero {espera:.0f}s")
                    time.sleep(espera)
                claves = [forzada] if forzada else self.orden_de_intento()
            for clave in claves:
                try:
                    return llamada(clave)
                except Exception as e:
                    if not _es_limite(e):
                        raise
                    self.reportar_limite(clave, e)
                    ultimo = e
        if ultimo is None:
            raise RuntimeError("Todas las API keys de Groq están en pausa por límite de uso.")
        raise ultimo

    async def aejecutar(self, llamada):
        """Versión async: llamada(clave) devuelve un awaitable."""
        if not self.claves:
            raise RuntimeError("No hay API keys de Groq configuradas (GROQ_API_KEYS o GROQ_API_KEY).")
        ultimo: Optional[Exception] = None
        for ronda in range(self.rondas):
            claves = self.orden_de_intento()
            if not claves:
                plan = self._plan(ronda, ultimo is not None)
                if plan is None:
                    break
                espera, forzada = plan
                if espera:
                    print(f"   ⏳ Todas las keys en pausa: espero {espera:.0f}s")
                    await asyncio.sleep(espera)
                claves = [forzada] if forzada else self.orden_de_intento()
            for clave in claves:
                try:
                    return await llamada(clave)
                except Exception as e:
                    if not _es_limite(e):
                        raise
                    self.reportar_limite(clave, e)
                    ultimo = e
        if ultimo is None:
            raise RuntimeError("Todas las API keys de Groq están en pausa por límite de uso.")
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
