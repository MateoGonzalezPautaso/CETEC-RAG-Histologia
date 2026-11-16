"""
LLM helpers: retry logic, quota management, LangSmith setup.
"""

import asyncio
import os
import re
import time
from typing import Optional


# ── Env shim (compatible with .env and Colab userdata) ───────────────────────

class userdata:
    @staticmethod
    def get(key: str) -> Optional[str]:
        return os.environ.get(key)


# ── Quota state ───────────────────────────────────────────────────────────────

_llm_quota_blocked_until = 0.0


def _quota_blocked() -> bool:
    return time.time() < _llm_quota_blocked_until


def _mark_quota_blocked() -> None:
    global _llm_quota_blocked_until
    seconds = int(os.getenv("LLM_QUOTA_BLOCK_SECONDS", "300"))
    _llm_quota_blocked_until = time.time() + seconds


def _is_quota_error(error: Exception) -> bool:
    raw = str(error).lower()
    return any(token in raw for token in [
        "tokens per day", "tpd", "daily", "cuota diaria",
        "quota exceeded", "insufficient_quota", "resource_exhausted",
        "sin cuota", "cupo diario",
    ])


def _quota_message() -> str:
    return (
        "El proveedor del modelo está temporalmente ocupado o sin cuota disponible. "
        "Probá de nuevo en unos minutos."
    )


# ── Razonamiento visible ──────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)\s*", re.DOTALL | re.IGNORECASE)


def _sin_razonamiento(resp):
    """Quita bloques <think>…</think> si el modelo (Qwen) los deja en la respuesta."""
    contenido = getattr(resp, "content", None)
    if isinstance(contenido, str) and "<think>" in contenido.lower():
        resp.content = _THINK_RE.sub("", contenido).strip()
    return resp


# ── Retry wrappers ────────────────────────────────────────────────────────────

async def invoke_con_reintento(llm, messages, max_retries=None):
    if _quota_blocked():
        raise RuntimeError(_quota_message())

    max_retries = max_retries or int(os.getenv("LLM_MAX_RETRIES", "2"))
    retry_base = int(os.getenv("LLM_RETRY_BASE_SECONDS", "8"))

    for attempt in range(max_retries):
        try:
            return _sin_razonamiento(await llm.ainvoke(messages))
        except Exception as e:
            err_str = str(e)
            if _is_quota_error(e):
                _mark_quota_blocked()
                raise RuntimeError(_quota_message()) from e
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    wait = retry_base * (attempt + 1)
                    print(f"   ⚠️ Rate limit (429/503) — reintentando en {wait}s... ({attempt+1}/{max_retries})")
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(_quota_message()) from e
            else:
                raise

    # Si max_retries fuese 0 (p. ej. LLM_MAX_RETRIES=0) el bucle no se ejecuta:
    # nunca devolvemos None silenciosamente al llamador.
    raise RuntimeError(_quota_message())


def invoke_con_reintento_sync(llm, messages, max_retries=None):
    if _quota_blocked():
        raise RuntimeError(_quota_message())

    max_retries = max_retries or int(os.getenv("LLM_MAX_RETRIES", "2"))
    retry_base = int(os.getenv("LLM_RETRY_BASE_SECONDS", "8"))

    for attempt in range(max_retries):
        try:
            return _sin_razonamiento(llm.invoke(messages))
        except Exception as e:
            err_str = str(e)
            if _is_quota_error(e):
                _mark_quota_blocked()
                raise RuntimeError(_quota_message()) from e
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    wait = retry_base * (attempt + 1)
                    print(f"   ⚠️ Rate limit (429/503) — reintentando en {wait}s... ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    raise RuntimeError(_quota_message()) from e
            else:
                raise

    raise RuntimeError(_quota_message())


def embed_query_con_reintento(embeddings, texto: str, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            return embeddings.embed_query(texto)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    wait = 15 * (attempt + 1)
                    print(f"   ⚠️ Rate limit en embeddings — reintentando en {wait}s... ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    raise
            else:
                raise

    raise RuntimeError("No se pudo generar el embedding tras varios reintentos.")


def embed_documents_con_reintento(embeddings, textos: list, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            return embeddings.embed_documents(textos)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    wait = 15 * (attempt + 1)
                    print(f"   ⚠️ Rate limit en embeddings — reintentando en {wait}s... ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    raise
            else:
                raise

    raise RuntimeError("No se pudieron generar los embeddings tras varios reintentos.")


# ── LangSmith ─────────────────────────────────────────────────────────────────

def setup_langsmith():
    def _dummy_traceable(*args, **kwargs):
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    api_key = userdata.get("LANGSMITH_API_KEY")
    if not api_key:
        # Sin API key no activamos el tracing: si no, LangChain intenta enviar
        # trazas a LangSmith sin credenciales y spamea errores 401.
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        print("ℹ️ LangSmith deshabilitado (sin LANGSMITH_API_KEY)")
        return False, _dummy_traceable, None

    config = {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY": api_key,
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": userdata.get("LANGSMITH_PROJECT") or "rag-histologia",
    }
    for key, value in config.items():
        if value:
            os.environ[key] = value
    try:
        from langsmith import traceable, Client
        client = Client()
        print(f"✅ LangSmith — Proyecto: {os.environ.get('LANGCHAIN_PROJECT')}")
        return True, traceable, client
    except Exception as e:
        print(f"⚠️ LangSmith no disponible: {e}")
        return False, _dummy_traceable, None


LANGSMITH_ENABLED, traceable, langsmith_client = setup_langsmith()
