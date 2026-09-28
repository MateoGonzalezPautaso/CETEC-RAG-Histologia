"""
Servidor FastAPI para RAG Histología Qdrant — Fullstack A2UI
============================================================
Wrappea AsistenteHistologiaQdrant y expone endpoints REST.
"""

import base64
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from src.assistant import AsistenteHistologiaQdrant
from src.config import COLLECTION_CHUNKS, DIRECTORIO_PDFS

# ── Estado global ────────────────────────────────────────────────────
asistente: Optional[AsistenteHistologiaQdrant] = None
_init_complete = False
_init_error: Optional[str] = None


# ── Modelos Pydantic ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    image_base64: Optional[str] = None
    image_filename: Optional[str] = None
    # Identificador de sesión del navegador: aísla conversación / imagen activa
    # por usuario. Si no llega, se usa una sesión por defecto (compatibilidad).
    session_id: Optional[str] = None


class LimpiarRequest(BaseModel):
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    respuesta: str
    estructura_identificada: Optional[str] = None
    imagenes_recuperadas: list = []
    imagenes_base64: list = []  # Lista de {filename, base64, mime_type}
    trayectoria: list = []
    imagen_activa: Optional[str] = None
    mostrar_imagenes: bool = False
    # Id de la traza registrada para esta interacción (ver src/trazas.py).
    trace_id: Optional[str] = None


# ── Lifecycle ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global asistente, _init_complete, _init_error
    print("🚀 Iniciando servidor RAG Histología Qdrant + A2UI...")

    try:
        asistente = AsistenteHistologiaQdrant()
        await asistente.inicializar_componentes()

        print("📚 Leyendo PDFs...")
        asistente.procesar_contenido_base(DIRECTORIO_PDFS)

        print("📋 Extrayendo temario...")
        await asistente.extraer_y_preparar_temario()
        n_temas = len(asistente.extractor_temario.temas) if asistente.extractor_temario else 0
        print(f"   → {n_temas} temas")

        print("💾 Verificando e indexando base de datos Qdrant (si está vacía)...")
        await asistente.indexar_en_qdrant(DIRECTORIO_PDFS, forzar=False)

        _init_complete = True
        print("✅ Servidor listo")
    except Exception as e:
        import traceback
        traceback.print_exc()
        _init_error = str(e)
        print(f"❌ Error inicializando: {e}")

    yield

    # Shutdown
    if asistente:
        await asistente.cerrar()
    print("👋 Servidor apagado")


# ── App FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Histología Qdrant + A2UI",
    description="Sistema RAG Multimodal de Histología — Fullstack",
    version="5.0.0",
    lifespan=lifespan,
)

# Orígenes permitidos. La app se sirve same-origin (API_BASE=''), así que por
# defecto solo habilitamos orígenes locales para CORS; se puede ampliar con la
# env ALLOWED_ORIGINS (lista separada por comas, o "*" para abrir todo).
_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
if _origins_env == "*":
    _allow_origins, _allow_origin_regex = ["*"], None
elif _origins_env:
    _allow_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
    _allow_origin_regex = None
else:
    _allow_origins = []
    _allow_origin_regex = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_origin_regex=_allow_origin_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Archivos estáticos del cliente
CLIENT_DIR = Path(__file__).parent / "client"

# Directorio de imágenes extraídas (para servir al frontend)
IMAGENES_DIR = Path(__file__).parent / "imagenes_extraidas"


def _check_ready():
    if not _init_complete:
        raise HTTPException(503, detail=_init_error or "Sistema inicializándose...")


def _prune_chat_images(chat_dir: Path, keep: int = 20, protect: Optional[str] = None):
    """Keep only the most recent uploads so imagenes_chat/ doesn't grow unbounded.

    The currently-active image (reused across turns) is never deleted.
    """
    try:
        files = sorted(
            chat_dir.glob("upload_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for p in files[keep:]:
            if protect and os.path.abspath(str(p)) == os.path.abspath(protect):
                continue
            try:
                p.unlink()
            except Exception:
                pass
    except Exception:
        pass


def _friendly_error_message(error: Exception) -> str:
    raw = str(error).lower()
    if any(token in raw for token in ["rate limit", "quota", "429", "resource_exhausted", "sin cuota"]):
        return (
            "El asistente está temporalmente ocupado porque se agotó el cupo diario del modelo. "
            "Probá de nuevo en unos minutos o avisá al equipo docente si el problema continúa."
        )
    if any(token in raw for token in ["503", "temporarily", "ocupado", "connection", "timeout"]):
        return (
            "El asistente no pudo completar la respuesta por una demora del servicio. "
            "Probá nuevamente en unos minutos."
        )
    return (
        "No pude completar la respuesta en este momento. "
        "Probá nuevamente o reformulá la pregunta."
    )


def _status_for_error(error: Exception) -> int:
    """503 solo para cuota/servicio ocupado; el resto es 500 (bug real).

    Antes todo se devolvía como 503 'ocupado/sin cuota', enmascarando errores
    de programación y haciendo creer a los operadores que era un problema de
    cuota inexistente.
    """
    raw = str(error).lower()
    transitorios = [
        "rate limit", "quota", "429", "resource_exhausted", "sin cuota",
        "503", "temporarily", "ocupado", "connection", "timeout",
    ]
    return 503 if any(token in raw for token in transitorios) else 500


# Límite de tamaño y extensiones permitidas para imágenes subidas por el chat.
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_MB", "8")) * 1024 * 1024
_EXT_IMAGEN_OK = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def _guardar_imagen_subida(image_base64: str, image_filename: Optional[str],
                           chat_img_dir: Path) -> str:
    """Decodifica y guarda la imagen subida, validando tamaño y extensión.

    - La extensión se restringe a una lista blanca (no se confía en el nombre
      provisto por el cliente, que podría pedir .py/.html).
    - Se rechazan payloads mayores a MAX_IMAGE_BYTES para evitar DoS de memoria.
    """
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise ValueError(f"Imagen inválida: no es base64 válido ({exc}).") from exc

    if not raw:
        raise ValueError("Imagen vacía.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"La imagen supera el límite de {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )

    ext = ".png"
    if image_filename:
        _, cand = os.path.splitext(image_filename)
        if cand.lower() in _EXT_IMAGEN_OK:
            ext = cand.lower()

    chat_img_dir.mkdir(exist_ok=True)
    nombre_archivo = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    imagen_path = str(chat_img_dir / nombre_archivo)
    with open(imagen_path, "wb") as f:
        f.write(raw)
    return imagen_path


# ── Rutas de archivos del frontend ───────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(str(CLIENT_DIR / "index.html"))


@app.get("/app.js")
async def serve_js():
    return FileResponse(str(CLIENT_DIR / "app.js"), media_type="application/javascript")


@app.get("/style.css")
async def serve_css():
    return FileResponse(str(CLIENT_DIR / "style.css"), media_type="text/css")


# ── Favicon / iconos del cliente ─────────────────────────────────────
# Servidos en la raíz porque index.html los referencia con rutas absolutas.
_FAVICONS = {
    "favicon.ico": "image/x-icon",
    "favicon.svg": "image/svg+xml",
    "favicon-16.png": "image/png",
    "favicon-32.png": "image/png",
    "favicon-180.png": "image/png",
}

def _make_favicon_route(fname: str, mime: str):
    async def _serve():
        return FileResponse(str(CLIENT_DIR / fname), media_type=mime)
    return _serve

for _fname, _mime in _FAVICONS.items():
    app.add_api_route(f"/{_fname}", _make_favicon_route(_fname, _mime), methods=["GET"])


# ── API: Estado ──────────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    if not _init_complete:
        return {
            "ready": False,
            "error": _init_error,
            "diagnostico": {
                "sistema": "Inicializando",
                "qdrant": "No verificado",
                "modelo": "No disponible",
                "cuota_modelo": "No verificada",
            },
        }

    qdrant_ok = False
    try:
        if asistente and asistente.qdrant_store and asistente.qdrant_store.client:
            asistente.qdrant_store.client.count(collection_name=COLLECTION_CHUNKS)
            qdrant_ok = True
    except Exception:
        qdrant_ok = False

    modelo_ok = bool(getattr(asistente, "llm", None))

    return {
        "ready": True,
        "n_temas": len(asistente.extractor_temario.temas) if asistente.extractor_temario else 0,
        # La imagen activa y el turno son por sesión; el endpoint de estado es
        # global, así que no los expone aquí (el /api/chat ya devuelve imagen_activa).
        "imagen_activa": None,
        "turno": 0,
        "device": asistente.device,
        "diagnostico": {
            "sistema": "Sistema listo",
            "qdrant": "Qdrant conectado" if qdrant_ok else "Qdrant no disponible",
            "modelo": "Modelo disponible" if modelo_ok else "Modelo no disponible",
            "cuota_modelo": "Cuota del modelo: no verificada / limitada",
        },
    }


# ── API: Temario ─────────────────────────────────────────────────────
@app.get("/api/temario")
async def get_temario():
    _check_ready()
    temas = asistente.extractor_temario.temas if asistente.extractor_temario else []
    return {"temas": temas, "total": len(temas)}


# ── API: Chat (texto plano) ─────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def post_chat(req: ChatRequest):
    _check_ready()

    user_id = req.session_id or "default_user"
    memoria = asistente._get_memoria(user_id)

    imagen_path = None
    try:
        # Si hay imagen, validarla y guardarla en un directorio persistente.
        if req.image_base64:
            chat_img_dir = Path(__file__).parent / "imagenes_chat"
            imagen_path = _guardar_imagen_subida(
                req.image_base64, req.image_filename, chat_img_dir,
            )
            print(f"📷 Imagen guardada para chat: {imagen_path}")
            _prune_chat_images(chat_img_dir, keep=20, protect=memoria.get_imagen_activa())

        # Ejecutar consulta RAG. consultar() devuelve el resultado completo, así
        # evitamos leer estado compartido que otra request concurrente podría pisar.
        resultado = await asistente.consultar(
            consulta_texto=req.query,
            imagen_path=imagen_path,
            user_id=user_id,
        )

        estructura = resultado.get("estructura_identificada")

        # Imágenes de la BD para mostrar al usuario
        imagenes_para_mostrar = resultado.get("imagenes_para_mostrar", [])
        imagenes_response = []
        mostrar_imgs = resultado.get("mostrar_imagenes", False)

        if mostrar_imgs and imagenes_para_mostrar:
            for img_info in imagenes_para_mostrar:
                nombre = img_info.get("nombre_archivo", "")
                if nombre:
                    imagenes_response.append({
                        "url": f"/imagenes_extraidas/{nombre}",
                        "caption": img_info.get("caption", ""),
                        "etiqueta": img_info.get("etiqueta", ""),
                        "nombre_archivo": nombre,
                        "similitud": img_info.get("similitud_semantica", 0),
                    })
            print(f"🖼️ {len(imagenes_response)} imágenes para mostrar al usuario")

        return ChatResponse(
            respuesta=resultado.get("respuesta", ""),
            estructura_identificada=estructura,
            imagenes_recuperadas=imagenes_response,
            imagenes_base64=[],
            trayectoria=resultado.get("trayectoria", []),
            imagen_activa=resultado.get("imagen_activa"),
            mostrar_imagenes=mostrar_imgs and len(imagenes_response) > 0,
            trace_id=resultado.get("trace_id"),
        )

    except ValueError as e:
        # Entrada inválida (imagen corrupta / demasiado grande / etc.).
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(_status_for_error(e), detail=_friendly_error_message(e))


# ── API: Limpiar imagen ──────────────────────────────────────────────
@app.post("/api/imagen/limpiar")
async def limpiar_imagen(req: Optional[LimpiarRequest] = None):
    _check_ready()
    user_id = (req.session_id if req else None) or "default_user"
    asistente._get_memoria(user_id).set_imagen(None)
    return {"ok": True, "mensaje": "Imagen activa eliminada"}


# ── Ruta estática: imágenes extraídas ────────────────────────────────
IMAGENES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/imagenes_extraidas", StaticFiles(directory=str(IMAGENES_DIR)), name="imagenes_extraidas")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    port = int(os.getenv("PORT", "10007"))
    print(f"🌐 Servidor en http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
