"""
Registro de trazas de interacción (Etapa 2 — captura de trazas).

Cada consulta que pasa por el pipeline se guarda como una línea JSON en
`TRAZAS_DIR/trazas-AAAA-MM-DD.jsonl` (un archivo por día, UTC). Una traza
contiene todo lo necesario para reconstruir la interacción sin volver a
ejecutar el pipeline: la consulta (original y reescrita), el contexto
recuperado que vio el LLM, la respuesta y la trayectoria por nodo.

El formato está definido en un único lugar (`construir_traza`) para poder
adaptarlo al esquema común del proyecto sin tocar el resto del código.
Registrar una traza nunca debe romper una respuesta: los errores se loguean
y se descartan.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import VERSION_PIPELINE

SCHEMA_VERSION = 1


# ── Construcción ──────────────────────────────────────────────────────────────

def _resultados(resultados: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    salida = []
    for r in resultados or []:
        img = r.get("imagen_path")
        salida.append({
            "id": r.get("id"),
            "fuente": r.get("fuente"),
            "pagina": r.get("pagina"),
            "tipo": r.get("tipo"),
            "similitud": r.get("similitud"),
            "texto": r.get("texto"),
            "imagen": os.path.basename(img) if img else None,
        })
    return salida


def construir_traza(
    final: Dict[str, Any],
    consulta_original: str,
    session_id: str,
    imagen_subida: Optional[str],
    duracion_s: float,
    modelo: Optional[str] = None,
    origen: str = "api",
    error: Optional[str] = None,
    prompt: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Arma el registro de una interacción a partir del estado final del grafo.

    `final` puede venir vacío si el grafo falló; en ese caso la traza igual se
    registra con el error, para no perder las consultas que fallan.
    """
    final = final or {}
    respuesta = final.get("respuesta_final", "") or ""
    if error is None and respuesta.startswith("Error:"):
        error = respuesta

    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origen": origen,
        "session_id": session_id,
        "version_pipeline": VERSION_PIPELINE,
        "modelo": modelo,
        # Prompt de respuesta activo (default u optimizado por GEPA, ver src/prompts.py).
        "prompt_respuesta": prompt or {"fuente": "default"},
        "consulta": {
            "original": consulta_original,
            "reescrita": final.get("consulta_texto"),
            "busqueda_texto": final.get("consulta_busqueda_texto"),
            "historial": final.get("historial_conversacional"),
        },
        "imagen": {
            # Solo el nombre de archivo: la imagen en sí no se copia a la traza.
            "subida": os.path.basename(imagen_subida) if imagen_subida else None,
            "tiene_imagen": final.get("tiene_imagen", False),
            "es_nueva": final.get("imagen_es_nueva", False),
            "analisis_visual": final.get("analisis_visual"),
            "analisis_comparativo": final.get("analisis_comparativo"),
            "estructura_identificada": final.get("estructura_identificada"),
        },
        "clasificacion": {
            "tema_valido": final.get("tema_valido"),
            "tema_encontrado": final.get("tema_encontrado"),
            "similitud_dominio": final.get("similitud_semantica_dominio"),
            "mostrar_imagenes": final.get("mostrar_imagenes", False),
        },
        "recuperacion": {
            "n_resultados": len(final.get("resultados_busqueda", []) or []),
            "contexto_suficiente": final.get("contexto_suficiente"),
            "resultados_validos": _resultados(final.get("resultados_validos", [])),
            # Lo que efectivamente se le pasó al LLM como SECCIONES DEL MANUAL.
            "contexto_documentos": final.get("contexto_documentos"),
        },
        "respuesta": {
            "texto": respuesta,
            "imagenes_mostradas": [
                img.get("nombre_archivo") for img in final.get("imagenes_para_mostrar", []) or []
            ],
        },
        "trayectoria": final.get("trayectoria", []),
        "duracion_s": round(duracion_s, 3),
        "error": error,
    }


# ── Persistencia ──────────────────────────────────────────────────────────────

class RegistroTrazas:
    """Escribe trazas en archivos JSONL diarios. Seguro entre hilos."""

    def __init__(self, directorio: str, habilitado: bool = True):
        self.directorio = directorio
        self.habilitado = habilitado
        self._lock = threading.Lock()

    def _ruta_del_dia(self, timestamp: str) -> str:
        return os.path.join(self.directorio, f"trazas-{timestamp[:10]}.jsonl")

    def registrar(self, traza: Dict[str, Any]) -> Optional[str]:
        """Agrega la traza al archivo del día. Devuelve la ruta, o None si no se escribió."""
        if not self.habilitado:
            return None
        try:
            # default=str: un valor no serializable (p. ej. un numpy.float32 que
            # se cuele desde Qdrant) no debe hacer perder la traza entera.
            linea = json.dumps(traza, ensure_ascii=False, default=str)
            ruta = self._ruta_del_dia(traza.get("timestamp", ""))
            with self._lock:
                os.makedirs(self.directorio, exist_ok=True)
                with open(ruta, "a", encoding="utf-8") as f:
                    f.write(linea + "\n")
            return ruta
        except Exception as e:
            print(f"   ⚠️ No se pudo registrar la traza: {e}")
            return None


def leer_trazas(directorio: str) -> List[Dict[str, Any]]:
    """Lee todas las trazas de `directorio`, en orden cronológico de archivo.

    Las líneas corruptas (p. ej. un corte a mitad de escritura) se saltean.
    """
    trazas: List[Dict[str, Any]] = []
    if not os.path.isdir(directorio):
        return trazas
    for nombre in sorted(os.listdir(directorio)):
        if not (nombre.startswith("trazas-") and nombre.endswith(".jsonl")):
            continue
        with open(os.path.join(directorio, nombre), encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    trazas.append(json.loads(linea))
                except json.JSONDecodeError:
                    continue
    return trazas
