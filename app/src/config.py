"""
Global constants, paths, and utility functions.
"""

import os
import unicodedata


# ── Thresholds ────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.70

# ── Embedding dimensions ──────────────────────────────────────────────────────
DIM_TEXTO = 384
DIM_IMG_UNI = 1024
DIM_IMG_PLIP = 512

# ── Directories ───────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRECTORIO_IMAGENES = os.path.join(_BASE, "imagenes_extraidas")

_PDFS_APP = os.path.join(_BASE, "pdf")
_PDFS_REPO = os.path.abspath(os.path.join(_BASE, os.pardir, "data", "pdf"))


def _tiene_pdfs(path: str) -> bool:
    try:
        return os.path.isdir(path) and any(
            nombre.lower().endswith(".pdf") for nombre in os.listdir(path)
        )
    except Exception:
        return False


DIRECTORIO_PDFS = _PDFS_APP if _tiene_pdfs(_PDFS_APP) or not _tiene_pdfs(_PDFS_REPO) else _PDFS_REPO

_QDRANT_PATH_ENV = os.getenv("QDRANT_PATH")
if _QDRANT_PATH_ENV:
    QDRANT_PATH = (
        _QDRANT_PATH_ENV if os.path.isabs(_QDRANT_PATH_ENV)
        else os.path.abspath(os.path.join(_BASE, _QDRANT_PATH_ENV))
    )
else:
    QDRANT_PATH = os.path.join(_BASE, "qdrant_data")

# ── Trazas de interacción (Etapa 2) ───────────────────────────────────────────
# Cada consulta se registra como una línea JSON en TRAZAS_DIR, un archivo por
# día (UTC). Son la materia prima para el dataset de evaluación/optimización.
VERSION_PIPELINE = "5.0"
TRAZAS_HABILITADAS = os.getenv("TRAZAS_HABILITADAS", "true").strip().lower() not in ("0", "false", "no")
_TRAZAS_DIR_ENV = os.getenv("TRAZAS_DIR")
if _TRAZAS_DIR_ENV:
    TRAZAS_DIR = (
        _TRAZAS_DIR_ENV if os.path.isabs(_TRAZAS_DIR_ENV)
        else os.path.abspath(os.path.join(_BASE, _TRAZAS_DIR_ENV))
    )
else:
    TRAZAS_DIR = os.path.join(_BASE, "trazas")

# ── Qdrant collections ────────────────────────────────────────────────────────
COLLECTION_CHUNKS = "histo_chunks"
COLLECTION_IMAGENES = "histo_imagenes"

INDEX_TEXTO = "histo_text"
INDEX_UNI = "histo_img_uni"
INDEX_PLIP = "histo_img_plip"

# ── Histology domain knowledge ────────────────────────────────────────────────
FEATURES_DISCRIMINATORIAS = [
    "presencia/ausencia de lumen central",
    "estratificación celular (capas concéntricas vs difusa)",
    "tipo de queratinización (parakeratosis, ortoqueratosis, ninguna)",
    "aspecto del núcleo (picnótico, fantasma, ausente, vesicular)",
    "células fantasma (sí/no)",
    "material amorfo central (sí/no y aspecto)",
    "patrón de tinción H&E (eosinofilia, basofilia)",
    "tamaño estimado de la estructura",
    "tejido circundante (estroma, epitelio, piel, otro)",
    "reacción inflamatoria perilesional (sí/no, tipo)",
    "tipo de matriz extracelular: homogénea/vítrea (cartílago) vs calcificada con canalículos (hueso)",
    "células en lagunas: condrocitos (redondeados, en nidos/isógenos) vs osteocitos (estrellados, aislados con prolongaciones)",
    "presencia de pericondrio (cartílago) vs periostio (hueso)",
    "sistemas de Havers / osteonas (exclusivo de hueso compacto)",
    "nidos isógenos / grupos celulares (exclusivo de cartílago)",
    "tinción de la matriz: basófila-azulada (cartílago hialino) vs eosinófila-rosada (hueso)",
    "fibras visibles en la matriz: ausentes (hialino), elásticas (elástico), colágenas gruesas (fibroso)",
    "vascularización: avascular (cartílago, excepto fibrocartílago) vs muy vascularizado (hueso)",
]

ANCLAS_SEMANTICAS_HISTOLOGIA = [
    "histología tejido celular microscopía",
    "tipos de tejido epitelial conectivo muscular nervioso",
    "coloración hematoxilina eosina H&E tinción histológica",
    "estructuras celulares núcleo citoplasma membrana",
    "diagnóstico diferencial patología biopsia",
    "glándulas epitelio estratificado cilíndrico simple",
    "identificar tejido muestra microscópica",
    "¿qué tipo de tejido es este?",
    "¿cuál es la estructura observada en la imagen?",
    "clasificar célula estructura histológica",
    "tumor quiste folículo cuerpo lúteo albicans",
    "corte histológico preparación muestra lámina",
    "tejido epitelial simple cilíndrico estratificado pseudoestratificado",
    "tejido conectivo laxo denso adiposo cartilaginoso óseo",
    "tejido muscular liso estriado cardíaco esquelético",
    "tejido nervioso neurona glía axón dendrita",
]


# ── Reglas de extracción de entidades (fuente única) ──────────────────────────
# Usadas tanto por el pipeline en vivo (extractors.ExtractorEntidades) como por
# el backfill (backfill_metadata_payloads). Antes estaban duplicadas en ambos
# lados y divergían al editar solo uno.
REGLAS_ENTIDADES = {
    "tejidos": {
        "musculo liso": ["musculo liso", "musculares lisas", "musculares lisos"],
        "tejido conectivo": ["tejido conectivo", "conectivo colageno", "fibras colagenas"],
        "epitelio seminifero": ["epitelio seminifero"],
        "endotelio": ["endotelio", "endoteliales"],
    },
    "estructuras": {
        "tunica intima": ["tunica intima"],
        "tunica media": ["tunica media"],
        "tunica adventicia": ["tunica adventicia", "adventicia", "tunica externa"],
        "lamina elastica interna": ["lamina elastica interna"],
        "lamina elastica externa": ["lamina elastica externa"],
        "tubulo seminifero": ["tubulo seminifero", "tubulos seminiferos"],
        "intersticio testicular": ["intersticio testicular", "intersticio"],
        "membrana basal": ["membrana basal"],
    },
    "tinciones": {
        "hematoxilina-eosina": ["hematoxilina", "eosina", "h&e", "he"],
    },
    "dominios": {
        "vasos sanguineos": ["arteria", "arterial", "arteriola", "vena", "venula",
                             "vaso sanguineo", "vascular", "tunica intima", "tunica media"],
        "testiculo": ["testiculo", "testicular", "seminifero", "seminiferos",
                      "sertoli", "leydig", "espermatogonia", "espermatide",
                      "peritubular", "mioide"],
    },
    "organos": {
        "arteria muscular": ["arteria muscular"],
        "testiculo": ["testiculo", "testicular"],
        "tubulo seminifero": ["tubulo seminifero", "tubulos seminiferos"],
    },
    "celulas": {
        "celula de sertoli": ["sertoli"],
        "celula de leydig": ["leydig"],
        "espermatogonia": ["espermatogonia"],
        "espermatide": ["espermatide", "espermátide"],
        "celula peritubular": ["peritubular", "mioide"],
        "celula endotelial": ["endotelial", "endoteliales"],
    },
    "temas": {
        "arteria muscular": ["arteria muscular"],
        "capas arteriales": ["tunica intima", "tunica media", "tunica adventicia", "tunica externa"],
        "laminas elasticas": ["lamina elastica interna", "lamina elastica externa"],
        "espermatogenesis": ["espermatogenesis", "espermatogonia", "espermatide"],
        "epitelio seminifero": ["epitelio seminifero", "tubulo seminifero"],
        "intersticio testicular": ["intersticio", "leydig"],
    },
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _safe(value, default: str = "") -> str:
    return value if isinstance(value, str) and value else default


def normalizar(texto: str) -> str:
    texto = str(texto or "").lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def extraer_entidades_por_reglas(texto: str) -> dict:
    """Extracción determinística de entidades según REGLAS_ENTIDADES."""
    texto_norm = normalizar(texto)
    entidades = {categoria: [] for categoria in REGLAS_ENTIDADES}
    for categoria, valores in REGLAS_ENTIDADES.items():
        for etiqueta, patrones in valores.items():
            if any(normalizar(patron) in texto_norm for patron in patrones):
                entidades[categoria].append(etiqueta)
    return entidades
