"""
Data extractors:
- ExtractorImagenesPDF: pulls images from PDFs, applies preprocessing + magnification.
- ExtractorTemario: asks the LLM to enumerate histology topics from the corpus.
- ExtractorEntidades: extracts histological entities (tissues, structures, stains, etc.).
"""

import glob
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image

from .config import (
    DIRECTORIO_IMAGENES, DIM_TEXTO, extraer_entidades_por_reglas,
    normalizar as _normalizar,
)
from .embeddings import ImageExtractionConfig
from .llm import invoke_con_reintento


# ── PDF Image Extractor ───────────────────────────────────────────────────────

class ExtractorImagenesPDF:
    """
    Extracts the largest image per PDF page, applies contrast/brightness
    enhancement, magnifies if below target size, and runs OCR for caption text.
    Falls back to rendering the full page when no embedded image is found.
    """

    def __init__(
        self,
        directorio_salida: str = DIRECTORIO_IMAGENES,
        config: Optional[ImageExtractionConfig] = None,
    ):
        self.directorio_salida = directorio_salida
        self.config = config or ImageExtractionConfig()
        os.makedirs(directorio_salida, exist_ok=True)

    def _apply_preprocessing(self, image: Image.Image) -> Image.Image:
        try:
            from PIL import ImageEnhance
            enhanced = image
            if self.config.ENHANCE_CONTRAST:
                enhanced = ImageEnhance.Contrast(enhanced).enhance(self.config.CONTRAST_FACTOR)
            if self.config.ENHANCE_BRIGHTNESS:
                enhanced = ImageEnhance.Brightness(enhanced).enhance(self.config.BRIGHTNESS_FACTOR)
            return enhanced
        except Exception as e:
            print(f"  ⚠️ Preprocessing failed: {e}")
            return image

    def _apply_magnification(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        target = self.config.TARGET_MAGNIFICATION_SIZE
        if w >= target and h >= target:
            return image
        if w < target:
            new_w, new_h = target, int(h * (target / w))
            print(f"  🔍 Magnifying: {w}x{h} → {new_w}x{new_h} (width-based)")
        else:
            new_h, new_w = target, int(w * (target / h))
            print(f"  🔍 Magnifying: {w}x{h} → {new_w}x{new_h} (height-based)")
        return image.resize((new_w, new_h), self.config.RESAMPLING_ALGORITHM)

    def _fallback_render_page(self, pdf_path: str, page_num: int) -> Optional[Image.Image]:
        # PyMuPDF renderiza la página sin depender de Poppler (que en Windows no
        # suele estar instalado); pdf2image queda como segunda opción.
        try:
            print(f"  🔄 Fallback: rendering page {page_num} with PyMuPDF")
            with fitz.open(pdf_path) as doc:
                pix = doc[page_num - 1].get_pixmap(dpi=self.config.FALLBACK_DPI)
                return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        except Exception as e:
            print(f"  ⚠️ PyMuPDF no pudo renderizar la página {page_num}: {e}")
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=self.config.FALLBACK_DPI)
            return pages[0] if pages else None
        except Exception as e:
            print(f"  ❌ Fallback rendering failed for page {page_num}: {e}")
            return None

    @staticmethod
    def _extraer_etiqueta(texto: str) -> str:
        if not texto:
            return ""
        for patron in [
            r'(Imagen\s+\d+[\.]?\d*[A-Za-z]?)',
            r'(Fig(?:ura)?\.?\s*\d+[\.]?\d*[A-Za-z]?)',
            r'(Lámina\s+\d+[\.]?\d*[A-Za-z]?)',
            r'(Foto(?:grafía)?\s+\d+[\.]?\d*[A-Za-z]?)',
        ]:
            m = re.search(patron, texto, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return ""

    @staticmethod
    def extraer_caption_imagen(page_fitz, img_bbox, texto_pagina: str) -> str:
        """Extract caption by searching text below the image bounding box."""
        caption = ""
        try:
            page_rect = page_fitz.rect
            area = fitz.Rect(0, max(0, img_bbox[3] - 10), page_rect.width, page_rect.height)
            caption = page_fitz.get_text("text", clip=area).strip()
            if not caption:
                area = fitz.Rect(0, img_bbox[3], page_rect.width, page_rect.height)
                caption = page_fitz.get_text("text", clip=area).strip()
        except Exception:
            pass
        if caption:
            caption = re.sub(r'\n\s*\d{1,3}\s*$', '', caption).strip()
            return caption
        return texto_pagina[:500] if texto_pagina else ""

    def extraer_de_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        imagenes_extraidas = []
        nombre_pdf = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Error abriendo {pdf_path}: {e}")
            return []

        for num_pagina, pagina in enumerate(doc, start=1):
            idx_0 = num_pagina - 1
            valid_images = []

            try:
                img_info_list = pagina.get_image_info(xrefs=True)
                xref_to_bbox = {info["xref"]: info.get("bbox") for info in img_info_list if info.get("xref")}
                for xref, bbox in xref_to_bbox.items():
                    try:
                        base_image = doc.extract_image(xref)
                        if not base_image:
                            continue
                        from io import BytesIO
                        pil_temp = Image.open(BytesIO(base_image["image"]))
                        w, h = pil_temp.size
                        if w >= self.config.MIN_WIDTH and h >= self.config.MIN_HEIGHT:
                            valid_images.append({"pil": pil_temp, "area": w * h, "bbox": bbox})
                    except Exception:
                        continue
            except Exception:
                pass

            texto_pagina = ""
            try:
                texto_pagina = doc[idx_0].get_text().strip()
            except Exception:
                pass

            if valid_images:
                mejor = max(valid_images, key=lambda x: x["area"])
                pil_img = self._apply_magnification(self._apply_preprocessing(mejor["pil"]))
                nombre_archivo = f"{nombre_pdf}_pag{num_pagina}.png"
                ruta = os.path.join(self.directorio_salida, nombre_archivo)
                try:
                    pil_img.save(ruta, format="PNG")
                    try:
                        import pytesseract
                        ocr_text = pytesseract.image_to_string(pil_img).strip()[:300]
                    except Exception:
                        ocr_text = ""
                    caption = self.extraer_caption_imagen(pagina, mejor["bbox"], texto_pagina) if mejor.get("bbox") else ""
                    etiqueta = self._extraer_etiqueta(caption) or self._extraer_etiqueta(texto_pagina)
                    imagenes_extraidas.append({
                        "path": ruta, "fuente_pdf": os.path.basename(pdf_path),
                        "pagina": num_pagina, "indice": 1, "ocr_text": ocr_text,
                        "texto_pagina": texto_pagina, "caption": caption,
                        "nombre_archivo": nombre_archivo, "etiqueta": etiqueta,
                    })
                except Exception as e:
                    print(f"  ⚠️ Error guardando pág {num_pagina}: {e}")
            else:
                pil_full = self._fallback_render_page(pdf_path, num_pagina)
                if pil_full:
                    pil_full = self._apply_magnification(self._apply_preprocessing(pil_full))
                    nombre_archivo = f"{nombre_pdf}_pag{num_pagina}_full.png"
                    ruta = os.path.join(self.directorio_salida, nombre_archivo)
                    try:
                        pil_full.save(ruta, format="PNG")
                        try:
                            import pytesseract
                            ocr_text = pytesseract.image_to_string(pil_full).strip()[:300]
                        except Exception:
                            ocr_text = ""
                        refs = re.findall(
                            r'(?:Fig(?:ura)?\.?\s*\d+[A-Za-z]?[^.]*\.)|(?:Imagen\s+\d+[\.\d]*[^.]*\.)',
                            texto_pagina, re.IGNORECASE,
                        )
                        caption_fb = "\n".join(refs[:3]) if refs else ""
                        etiqueta_fb = self._extraer_etiqueta(caption_fb) or self._extraer_etiqueta(texto_pagina)
                        imagenes_extraidas.append({
                            "path": ruta, "fuente_pdf": os.path.basename(pdf_path),
                            "pagina": num_pagina, "indice": 1, "ocr_text": ocr_text,
                            "texto_pagina": texto_pagina, "caption": caption_fb,
                            "nombre_archivo": nombre_archivo, "etiqueta": etiqueta_fb,
                        })
                    except Exception as e:
                        print(f"  ⚠️ Error guardando fallback pág {num_pagina}: {e}")

        doc.close()
        print(f"  📸 {len(imagenes_extraidas)} imágenes procesadas de {os.path.basename(pdf_path)}")
        return imagenes_extraidas

    def extraer_de_directorio(self, directorio: str) -> List[Dict[str, str]]:
        todas = []
        pdfs = glob.glob(os.path.join(directorio, "*.pdf"))
        print(f"📂 Extrayendo {len(pdfs)} PDFs...")
        for pdf_path in pdfs:
            todas.extend(self.extraer_de_pdf(pdf_path))
        print(f"✅ Total imágenes: {len(todas)}")
        return todas


# ── Syllabus Extractor ────────────────────────────────────────────────────────

class ExtractorTemario:
    """
    Uses the LLM to enumerate histology topics from the corpus text.
    Caches the result on disk and only re-generates when the corpus changes.
    """

    def __init__(self, llm):
        self.llm = llm
        self.temas: List[str] = []

    @staticmethod
    def _muestra_representativa(texto: str, presupuesto: int = 24000, ventanas: int = 6) -> str:
        """
        Sample evenly-spaced windows across the whole corpus instead of only the
        first N characters, so the syllabus reflects the entire manual (not just
        the opening pages of the first PDF).
        """
        if len(texto) <= presupuesto:
            return texto
        win = max(1, presupuesto // ventanas)
        paso = max(1, (len(texto) - win) // max(1, ventanas - 1))
        partes = [texto[i * paso: i * paso + win] for i in range(ventanas)]
        return "\n[...]\n".join(partes)

    def _limpiar(self, temas_raw: List[str]) -> List[str]:
        FRASES_MODELO = [
            "aqui te dejo", "aqui tienes", "lista exhaustiva",
            "espero que", "si necesitas", "no dudes", "de ayuda",
            "claro", "por supuesto",
        ]
        temas = []
        for item in temas_raw:
            tema = str(item or "").strip()
            if not tema:
                continue
            tema = re.sub(r"^[-*•\s]+", "", tema)
            tema = re.sub(r"^\d+[\.)]\s*", "", tema).strip()
            tema_norm = _normalizar(tema)
            if any(f in tema_norm for f in FRASES_MODELO):
                continue
            if len(tema) < 3 or len(tema) > 80 or tema.endswith(":"):
                continue
            if tema not in temas:
                temas.append(tema)
        return temas

    async def extraer_temario(self, texto_completo: str) -> List[str]:
        print("📋 Extrayendo temario...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_path = os.path.join(base_dir, "temario_histologia.json")
        hash_path = os.path.join(base_dir, "temario_histologia.sha256")
        corpus_hash = hashlib.sha256(texto_completo.encode("utf-8")).hexdigest()

        if os.path.exists(cache_path):
            try:
                cached_hash = open(hash_path).read().strip() if os.path.exists(hash_path) else ""
                if cached_hash == corpus_hash or not cached_hash:
                    temas_cache = json.load(open(cache_path, encoding="utf-8"))
                    if isinstance(temas_cache, list) and temas_cache:
                        self.temas = self._limpiar(temas_cache)
                        if not cached_hash:
                            open(hash_path, "w").write(corpus_hash)
                            json.dump(self.temas, open(cache_path, "w", encoding="utf-8"),
                                      ensure_ascii=False, indent=2)
                        print(f"✅ Temario desde cache: {len(self.temas)} temas")
                        return self.temas
            except Exception as e:
                print(f"⚠️ No se pudo leer cache de temario: {e}")

        muestra = self._muestra_representativa(texto_completo)
        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=(
                    "Eres un experto en histología. Genera una lista EXHAUSTIVA de temas, "
                    "estructuras, tejidos, células, tinciones del manual.\n"
                    "Un tema por línea, sin bullets. Solo la lista."
                )),
                HumanMessage(content=f"TEXTO:\n{muestra}"),
            ])
            self.temas = self._limpiar(resp.content.strip().split("\n"))
            json.dump(self.temas, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            open(hash_path, "w").write(corpus_hash)
            print(f"✅ Temario: {len(self.temas)} temas")
        except Exception as e:
            print(f"❌ Error extrayendo temario: {e}")
        return self.temas

    def get_temario_texto(self) -> str:
        return "\n".join(f"- {t}" for t in self.temas[:100]) if self.temas else "No disponible."


# ── Entity Extractor ──────────────────────────────────────────────────────────

class ExtractorEntidades:
    """
    Extracts histological entities (tissues, structures, stains, etc.) from text.
    Combines LLM output with deterministic rule-based extraction for reliability.
    """

    def __init__(self, llm):
        self.llm = llm

    def _merge_unicos(self, *listas) -> List[str]:
        vistos: set = set()
        salida = []
        for lista in listas:
            for item in lista or []:
                valor = str(item).strip().lower()
                if valor and valor not in vistos:
                    vistos.add(valor)
                    salida.append(valor)
        return salida

    def _extraer_reglas(self, texto: str) -> Dict[str, List[str]]:
        """Deterministic entity extraction to supplement LLM output.

        Las reglas viven en config.REGLAS_ENTIDADES (fuente única compartida con
        el script de backfill) para no duplicar el diccionario.
        """
        return extraer_entidades_por_reglas(texto)

    async def extraer_de_texto(self, texto: str) -> Dict[str, List[str]]:
        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=(
                    "Extrae entidades histológicas del texto. "
                    'Responde SOLO en JSON: {"tejidos": [...], "estructuras": [...], "tinciones": [...]}\n'
                    "Máximo 3 items por categoría. Si no hay, lista vacía."
                )),
                HumanMessage(content=texto[:500]),
            ])
            texto_resp = re.sub(r"```json\s*|\s*```", "", resp.content.strip())
            resultado = json.loads(texto_resp)
            reglas = self._extraer_reglas(texto)
            texto_norm = _normalizar(texto)
            # Nombres que las reglas ya ubicaron en otra categoría (p. ej.
            # "arteria muscular" es un órgano, no un tejido).
            ya_clasificados = {_normalizar(v) for k in ("dominios", "organos", "celulas", "temas")
                               for v in reglas.get(k, [])}

            def anclados(items):
                # Solo entidades que están en el texto: el LLM a veces agrega
                # términos inferidos ("tejido conectivo" para una arteria) que
                # la búsqueda por keywords puntúa como coincidencia exacta y
                # traen chunks de otros temas. Así la recuperación no depende
                # de qué tan "creativo" sea el modelo generador.
                salida = []
                for item in items[:3]:
                    item_norm = _normalizar(item).strip()
                    if item_norm and item_norm in texto_norm and item_norm not in ya_clasificados:
                        salida.append(str(item).lower())
                return salida

            return {
                "tejidos": self._merge_unicos(anclados(resultado.get("tejidos", [])), reglas.get("tejidos", [])),
                "estructuras": self._merge_unicos(anclados(resultado.get("estructuras", [])), reglas.get("estructuras", [])),
                "tinciones": self._merge_unicos(anclados(resultado.get("tinciones", [])), reglas.get("tinciones", [])),
                "dominios": reglas.get("dominios", []),
                "organos": reglas.get("organos", []),
                "celulas": reglas.get("celulas", []),
                "temas": reglas.get("temas", []),
            }
        except Exception:
            return self._extraer_reglas(texto)

    def extraer_de_texto_sync(self, texto: str) -> Dict[str, List[str]]:
        """Synchronous extraction using keyword lists (no LLM call)."""
        TEJIDOS = [
            "epitelio", "conectivo", "muscular", "nervioso", "cartílago", "hueso",
            "sangre", "linfoide", "hepático", "renal", "pulmonar", "dérmico",
            "epitelial", "estroma", "mucosa", "serosa",
        ]
        ESTRUCTURAS = [
            "célula", "núcleo", "citoplasma", "membrana", "gránulo", "fibra",
            "canalículo", "vellosidad", "cripta", "glomérulo", "túbulo", "alvéolo",
            "folículo", "sinusoide", "perla córnea", "cuerpo de albicans",
            "cuerpo de councilman", "queratina", "colágeno",
        ]
        TINCIONES = [
            "h&e", "hematoxilina", "eosina", "pas", "tricrómico", "grocott",
            "ziehl", "giemsa", "reticulina", "alcian blue", "von kossa",
        ]
        texto_lower = texto.lower()
        entidades: Dict[str, List[str]] = {
            "tejidos": [t for t in TEJIDOS if t in texto_lower][:3],
            "estructuras": [e for e in ESTRUCTURAS if e in texto_lower][:3],
            "tinciones": [t for t in TINCIONES if t in texto_lower][:3],
        }
        reglas = self._extraer_reglas(texto)
        for key in ("tejidos", "estructuras", "tinciones"):
            entidades[key] = self._merge_unicos(entidades[key], reglas.get(key, []))
        for key in ("dominios", "organos", "celulas", "temas"):
            entidades[key] = reglas.get(key, [])
        return entidades

