"""
Qdrant vector store: schema creation, upsert, and all search strategies.
"""

import asyncio
import os
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchAny, MatchValue, PointStruct, VectorParams,
)

from .config import (
    COLLECTION_CHUNKS, COLLECTION_IMAGENES,
    DIM_IMG_PLIP, DIM_IMG_UNI, DIM_TEXTO,
    INDEX_PLIP, INDEX_TEXTO, INDEX_UNI, QDRANT_PATH,
    normalizar as _sin_tildes,
)
from .llm import embed_query_con_reintento


class QdrantVectorStore:
    """
    Wraps a Qdrant instance and exposes:
    - Schema creation (two collections: chunks + images)
    - Upsert for chunks and images
    - Hybrid, vectorial, entity-based, and text-based search
    """

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None, path: Optional[str] = None):
        self.url = url
        self.api_key = api_key
        self.path = path or QDRANT_PATH
        if url:
            self.client = QdrantClient(url=url, api_key=api_key or None, timeout=60)
            self.location = url
        else:
            os.makedirs(self.path, exist_ok=True)
            self.client = QdrantClient(path=self.path, timeout=60)
            self.location = self.path

    # ── Connection ────────────────────────────────────────────────────────────

    async def connect(self):
        try:
            self.client.get_collections()
            print(f"✅ Qdrant conectado: {self.location}")
        except Exception as e:
            raise ConnectionError(f"No se pudo conectar a Qdrant: {e}")

    async def close(self):
        try:
            self.client.close()
        except Exception:
            pass

    # ── Schema ────────────────────────────────────────────────────────────────

    async def crear_esquema(self):
        print("🏗️ Creando esquema Qdrant (chunks e imágenes)...")
        self._ensure_chunks_collection()
        self._ensure_imagenes_collection()
        self._create_payload_indexes()
        print("✅ Esquema Qdrant listo (2 colecciones + payload index)")

    async def reiniciar_chunks(self):
        """Vacía la colección de chunks (antes de reindexar con otro chunking)."""
        try:
            self.client.delete_collection(COLLECTION_CHUNKS)
        except Exception:
            pass
        self._ensure_chunks_collection()
        self._create_payload_indexes()
        print(f"   🧹 Colección '{COLLECTION_CHUNKS}' vaciada para reindexar")

    def _ensure_chunks_collection(self):
        try:
            info = self.client.get_collection(COLLECTION_CHUNKS)
            config = info.config.params.vectors
            dims = 0
            if isinstance(config, dict):
                vec_config = config.get("texto", config.get("texto_emb"))
                dims = getattr(vec_config, "size", 0) if vec_config else 0
            else:
                dims = getattr(config, "size", 0)

            if dims != DIM_TEXTO:
                print(f"   ⚠️ Dimensión incorrecta en {COLLECTION_CHUNKS}. Recreando...")
                self.client.delete_collection(COLLECTION_CHUNKS)
                raise Exception("Recreate")
            print(f"   ✅ Colección '{COLLECTION_CHUNKS}' ya existe")
        except Exception as e:
            if "Recreate" not in str(e) and "Not found" not in str(e):
                pass
            try:
                self.client.create_collection(
                    collection_name=COLLECTION_CHUNKS,
                    vectors_config=VectorParams(size=DIM_TEXTO, distance=Distance.COSINE),
                )
                print(f"   ✅ Colección '{COLLECTION_CHUNKS}' creada")
            except Exception as e2:
                if "already exists" not in str(e2):
                    raise e2

    def _ensure_imagenes_collection(self):
        try:
            col_info = self.client.get_collection(COLLECTION_IMAGENES)
            config = col_info.config.params.vectors
            recrear = not isinstance(config, dict)
            if not recrear:
                uni_cfg = config.get("uni")
                plip_cfg = config.get("plip")
                text_cfg = config.get("texto_emb")
                recrear = (
                    not uni_cfg or getattr(uni_cfg, "size", 0) != DIM_IMG_UNI
                    or not plip_cfg or getattr(plip_cfg, "size", 0) != DIM_IMG_PLIP
                    or not text_cfg or getattr(text_cfg, "size", 0) != DIM_TEXTO
                )
            if recrear:
                print(f"   ⚠️ Estructura incorrecta en {COLLECTION_IMAGENES}. Recreando...")
                self.client.delete_collection(COLLECTION_IMAGENES)
                raise Exception("Recreate")
            print(f"   ✅ Colección '{COLLECTION_IMAGENES}' ya existe")
        except Exception as e:
            if "Recreate" not in str(e) and "Not found" not in str(e):
                pass
            try:
                self.client.create_collection(
                    collection_name=COLLECTION_IMAGENES,
                    vectors_config={
                        "uni": VectorParams(size=DIM_IMG_UNI, distance=Distance.COSINE),
                        "plip": VectorParams(size=DIM_IMG_PLIP, distance=Distance.COSINE),
                        "texto_emb": VectorParams(size=DIM_TEXTO, distance=Distance.COSINE),
                    },
                )
                print(f"   ✅ Colección '{COLLECTION_IMAGENES}' creada")
            except Exception as e2:
                if "already exists" not in str(e2):
                    raise e2

    def _create_payload_indexes(self):
        chunk_fields = ["tejidos", "estructuras", "tinciones", "dominios", "organos", "celulas", "temas", "fuente"]
        img_fields = ["fuente", "pagina_str", "tejidos", "estructuras", "tinciones", "dominios", "organos", "celulas", "temas"]

        for field in chunk_fields:
            try:
                self.client.create_payload_index(COLLECTION_CHUNKS, field, models.PayloadSchemaType.KEYWORD)
            except Exception:
                pass

        for field in img_fields:
            try:
                self.client.create_payload_index(COLLECTION_IMAGENES, field, models.PayloadSchemaType.KEYWORD)
            except Exception:
                pass

        try:
            self.client.create_payload_index(COLLECTION_CHUNKS, "pagina", models.PayloadSchemaType.INTEGER)
        except Exception:
            pass

    # ── Upsert ────────────────────────────────────────────────────────────────

    async def upsert_chunk(
        self, chunk_id: str, texto: str, fuente: str,
        chunk_idx: int, embedding: list, entidades: dict,
        pagina: int = 0, imagenes_pagina: list = None,
    ):
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)),
            vector=embedding,
            payload={
                "id": chunk_id, "texto": texto, "fuente": fuente,
                "chunk_idx": chunk_idx, "pagina": pagina,
                "imagenes_pagina": [p for p in (imagenes_pagina or []) if os.path.exists(p)],
                "tipo": "texto",
                "tejidos": entidades.get("tejidos", []),
                "estructuras": entidades.get("estructuras", []),
                "tinciones": entidades.get("tinciones", []),
                "dominios": entidades.get("dominios", []),
                "organos": entidades.get("organos", []),
                "celulas": entidades.get("celulas", []),
                "temas": entidades.get("temas", []),
            },
        )
        self.client.upsert(collection_name=COLLECTION_CHUNKS, points=[point])

    async def upsert_imagen(
        self, imagen_id: str, path: str, fuente: str,
        pagina: int, ocr_text: str, texto_pagina: str,
        emb_uni: list, emb_plip: list, emb_texto: list = None,
        caption: str = "", nombre_archivo: str = "", etiqueta: str = "",
    ):
        vectors = {
            "uni": emb_uni,
            "plip": emb_plip,
            "texto_emb": emb_texto if (emb_texto and any(v != 0.0 for v in emb_texto))
                         else [0.0] * DIM_TEXTO,
        }
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, imagen_id)),
            vector=vectors,
            payload={
                "id": imagen_id, "path": path, "fuente": fuente,
                "pagina": pagina, "pagina_str": str(pagina),
                "ocr_text": ocr_text, "texto_pagina": texto_pagina[:3000],
                "caption": caption, "nombre_archivo": nombre_archivo,
                "etiqueta": etiqueta, "tipo": "imagen",
            },
        )
        self.client.upsert(collection_name=COLLECTION_IMAGENES, points=[point])

    # ── Search ────────────────────────────────────────────────────────────────

    async def _scroll_all(self, collection_name: str, scroll_filter=None, batch: int = 500) -> list:
        """Scroll an entire collection (paginated) off the event loop."""
        puntos: list = []
        next_offset = None
        while True:
            lote, next_offset = await asyncio.to_thread(
                lambda off=next_offset: self.client.scroll(
                    collection_name=collection_name, scroll_filter=scroll_filter,
                    limit=batch, offset=off, with_payload=True, with_vectors=False,
                )
            )
            puntos.extend(lote)
            if next_offset is None:
                break
        return puntos

    async def busqueda_vectorial(self, embedding: list, index_name: str, top_k: int = 10) -> list:
        try:
            if index_name == INDEX_TEXTO:
                results = (await asyncio.to_thread(
                    self.client.query_points,
                    collection_name=COLLECTION_CHUNKS, query=embedding, limit=top_k,
                )).points
                return [{
                    "id": str(r.id), "texto": r.payload.get("texto", ""),
                    "fuente": r.payload.get("fuente", ""), "tipo": "texto",
                    "imagen_path": None, "similitud": r.score,
                    "nombre_archivo": "", "etiqueta": "",
                    "imagenes_pagina": r.payload.get("imagenes_pagina", []),
                    "pagina": r.payload.get("pagina"),
                } for r in results]
            else:
                using_vector = "uni" if index_name == INDEX_UNI else "plip"
                results = (await asyncio.to_thread(
                    self.client.query_points,
                    collection_name=COLLECTION_IMAGENES, query=embedding,
                    using=using_vector, limit=top_k,
                )).points
                out = []
                for r in results:
                    nombre_archivo = r.payload.get("nombre_archivo", "").lower()
                    if "_full." in nombre_archivo:
                        continue
                    caption = r.payload.get("caption", "")
                    texto_pag = r.payload.get("texto_pagina", "")
                    ocr = r.payload.get("ocr_text", "")
                    texto = caption or texto_pag or ocr
                    out.append({
                        "id": str(r.id), "texto": texto,
                        "fuente": r.payload.get("fuente", ""), "tipo": "imagen",
                        "imagen_path": r.payload.get("path"), "similitud": r.score,
                        "nombre_archivo": r.payload.get("nombre_archivo", ""),
                        "etiqueta": r.payload.get("etiqueta", ""),
                        "texto_pagina": texto_pag, "pagina": r.payload.get("pagina"),
                    })
                return out
        except Exception as e:
            print(f"⚠️ Error búsqueda vectorial Qdrant {index_name}: {e}")
            return []

    async def busqueda_chunks_por_pagina(self, fuente: str, pagina: int, top_k: int = 3) -> list:
        try:
            results, _ = await asyncio.to_thread(
                self.client.scroll,
                collection_name=COLLECTION_CHUNKS,
                scroll_filter=Filter(must=[
                    FieldCondition(key="fuente", match=MatchValue(value=fuente)),
                    FieldCondition(key="pagina", match=MatchValue(value=pagina)),
                ]),
                limit=top_k,
            )
            return [{
                "id": str(r.id), "texto": r.payload.get("texto", ""),
                "fuente": r.payload.get("fuente", ""), "tipo": "texto",
                "imagen_path": None, "similitud": 0.80,
                "nombre_archivo": "", "etiqueta": "",
                "imagenes_pagina": r.payload.get("imagenes_pagina", []),
                "pagina": r.payload.get("pagina"),
            } for r in results]
        except Exception as e:
            print(f"⚠️ Error búsqueda chunks por página: {e}")
            return []

    async def busqueda_por_entidades(self, entidades: dict, top_k: int = 10) -> list:
        tejidos = entidades.get("tejidos", [])
        estructuras = entidades.get("estructuras", [])
        tinciones = entidades.get("tinciones", [])
        dominios = entidades.get("dominios", [])
        organos = entidades.get("organos", [])
        celulas = entidades.get("celulas", [])
        temas = entidades.get("temas", [])

        if not any([tejidos, estructuras, tinciones, dominios, organos, celulas, temas]):
            return []

        conditions = []
        if tejidos: conditions.append(FieldCondition(key="tejidos", match=MatchAny(any=tejidos)))
        if estructuras: conditions.append(FieldCondition(key="estructuras", match=MatchAny(any=estructuras)))
        if tinciones: conditions.append(FieldCondition(key="tinciones", match=MatchAny(any=tinciones)))
        if dominios: conditions.append(FieldCondition(key="dominios", match=MatchAny(any=dominios)))
        if organos: conditions.append(FieldCondition(key="organos", match=MatchAny(any=organos)))
        if celulas: conditions.append(FieldCondition(key="celulas", match=MatchAny(any=celulas)))
        if temas: conditions.append(FieldCondition(key="temas", match=MatchAny(any=temas)))

        try:
            results, _ = await asyncio.to_thread(
                self.client.scroll,
                collection_name=COLLECTION_CHUNKS,
                scroll_filter=Filter(should=conditions),
                limit=top_k,
            )
            # Score must clear umbral_texto (0.30) after the text-mode weight (×0.60)
            # in busqueda_hibrida, otherwise entity-only hits are always filtered out.
            return [{
                "id": str(r.id), "texto": r.payload.get("texto", ""),
                "fuente": r.payload.get("fuente", ""), "tipo": "texto",
                "imagen_path": None, "similitud": 0.55,
                "nombre_archivo": "", "etiqueta": "",
                "imagenes_pagina": r.payload.get("imagenes_pagina", []),
                "pagina": r.payload.get("pagina"),
                "dominios": r.payload.get("dominios", []),
                "organos": r.payload.get("organos", []),
                "celulas": r.payload.get("celulas", []),
                "temas": r.payload.get("temas", []),
            } for r in results]
        except Exception as e:
            print(f"⚠️ Error búsqueda entidades: {e}")
            return []

    async def busqueda_chunks_por_texto(self, terminos: list, top_k: int = 10) -> list:
        """Keyword fallback when vector search returns weak results."""
        if not terminos:
            return []

        terminos_lower = list(set(
            [t.lower() for t in terminos] + [_sin_tildes(t.lower()) for t in terminos]
        ))
        terminos_largos = [t for t in terminos_lower if len(t.split()) >= 2]

        try:
            all_chunks = await self._scroll_all(COLLECTION_CHUNKS)
            resultados = []
            for r in all_chunks:
                texto = (r.payload.get("texto", "") or "").lower()
                texto_norm = _sin_tildes(texto)
                matches_largos = sum(1 for t in terminos_largos if t in texto or t in texto_norm)
                matches = sum(1 for t in terminos_lower if t in texto or t in texto_norm)
                if matches:
                    similitud = 0.95 if matches_largos else min(0.85, 0.50 + 0.08 * matches)
                    resultados.append({
                        "id": str(r.id), "texto": r.payload.get("texto", ""),
                        "fuente": r.payload.get("fuente", ""), "tipo": "texto",
                        "imagen_path": None, "similitud": similitud,
                        "nombre_archivo": "", "etiqueta": "",
                        "imagenes_pagina": r.payload.get("imagenes_pagina", []),
                        "pagina": r.payload.get("pagina"),
                        "dominios": r.payload.get("dominios", []),
                        "organos": r.payload.get("organos", []),
                        "celulas": r.payload.get("celulas", []),
                        "temas": r.payload.get("temas", []),
                    })
            if resultados:
                print(f"   📝 {len(resultados)} chunks encontrados (keyword fallback)")
            return sorted(resultados, key=lambda x: x.get("similitud", 0), reverse=True)[:top_k]
        except Exception as e:
            print(f"⚠️ Error búsqueda chunks por texto: {e}")
            return []

    async def busqueda_imagenes_por_texto(self, entidades: dict, top_k: int = 5) -> list:
        """Search images whose caption/page text contains query terms."""
        tejidos = entidades.get("tejidos", [])
        estructuras = entidades.get("estructuras", [])
        consulta = entidades.get("_consulta", [])
        terminos = tejidos + estructuras + consulta
        if not terminos:
            return []

        terminos_lower = list(set(
            [t.lower() for t in terminos] + [_sin_tildes(t.lower()) for t in terminos]
        ))

        try:
            all_imgs = await self._scroll_all(COLLECTION_IMAGENES)
            resultados = []
            for r in all_imgs:
                combined = _sin_tildes(" ".join([
                    (r.payload.get("caption", "") or "").lower(),
                    (r.payload.get("texto_pagina", "") or "").lower(),
                    (r.payload.get("ocr_text", "") or "").lower(),
                ]))
                if not any(t in combined for t in terminos_lower):
                    continue
                img_path = r.payload.get("path", "")
                nombre = r.payload.get("nombre_archivo", "").lower()
                if not img_path or not os.path.exists(img_path) or "_full." in nombre:
                    continue
                caption = r.payload.get("caption", "")
                texto_pag = r.payload.get("texto_pagina", "")
                resultados.append({
                    "id": str(r.id),
                    "texto": caption.strip() or texto_pag.strip() or r.payload.get("ocr_text", ""),
                    "fuente": r.payload.get("fuente", ""), "tipo": "imagen",
                    "imagen_path": img_path, "similitud": 0.50,
                    "nombre_archivo": r.payload.get("nombre_archivo", ""),
                    "etiqueta": r.payload.get("etiqueta", ""),
                    "caption_raw": caption,
                })
            if resultados:
                print(f"   🖼️ {len(resultados)} imágenes encontradas (texto directo)")
            return resultados[:top_k]
        except Exception as e:
            print(f"⚠️ Error búsqueda imágenes por texto: {e}")
            return []

    async def buscar_imagenes_por_referencia(self, patrones: list, top_k: int = 10) -> list:
        """Search images by matching etiqueta, caption, or filename against patterns."""
        if not patrones:
            return []

        patrones_lower = [p.lower() for p in patrones]
        patrones_norm = [_sin_tildes(p) for p in patrones_lower]

        try:
            all_imgs = await self._scroll_all(COLLECTION_IMAGENES)
            resultados = []
            vistas: set = set()
            for r in all_imgs:
                nombre = (r.payload.get("nombre_archivo", "") or "").lower()
                if "_full." in nombre:
                    continue
                combined = _sin_tildes(" ".join([
                    (r.payload.get("etiqueta", "") or "").lower(),
                    (r.payload.get("caption", "") or "").lower(),
                    nombre,
                ]))
                if not any(p in combined for p in patrones_norm):
                    continue
                img_path = r.payload.get("path", "")
                if not img_path or not os.path.exists(img_path) or img_path in vistas:
                    continue
                vistas.add(img_path)
                caption = r.payload.get("caption", "") or ""
                resultados.append({
                    "id": str(r.id),
                    "texto": caption.strip() or r.payload.get("texto_pagina", "") or "",
                    "fuente": r.payload.get("fuente", ""), "tipo": "imagen",
                    "imagen_path": img_path, "similitud": 0.95,
                    "nombre_archivo": r.payload.get("nombre_archivo", ""),
                    "etiqueta": r.payload.get("etiqueta", ""),
                    "caption_raw": caption, "origen": "referencia_respuesta",
                })
            if resultados:
                print(f"   🖼️ {len(resultados)} imágenes encontradas por referencia")
            return resultados[:top_k]
        except Exception as e:
            print(f"⚠️ Error búsqueda imágenes por referencia: {e}")
            return []

    def extraer_imagenes_de_resultados(self, resultados: list, top_k: int = 5) -> list:
        """Filter, validate, and deduplicate image results."""
        imagenes = [r for r in resultados if r.get("tipo") == "imagen"]
        if not imagenes:
            return []

        vistas: set = set()
        validas = []
        for img in imagenes:
            img_path = img.get("imagen_path", "")
            if not img_path or not os.path.exists(img_path):
                continue
            nombre = os.path.basename(img_path).lower()
            if "_full." in nombre or img_path in vistas:
                continue
            vistas.add(img_path)
            caption = img.get("caption_raw") or img.get("texto", "") or ""
            validas.append({
                "id": img.get("id", ""), "path": img_path, "caption": caption[:500],
                "nombre_archivo": img.get("nombre_archivo", os.path.basename(img_path)),
                "etiqueta": img.get("etiqueta", ""), "fuente": img.get("fuente", ""),
                "similitud_semantica": img.get("similitud", 0),
            })
        return validas[:top_k]

    async def busqueda_imagenes_semantica(
        self, texto_embedding: list, entidades: dict,
        embeddings_model, top_k: int = 5,
    ) -> list:
        """
        Semantic re-ranking of image candidates:
        1. Retrieve via texto_emb vector
        2. Re-rank by cosine similarity between query and caption embeddings
        3. Filter by source coherence and keyword match
        """
        if not texto_embedding:
            return []

        print("   🔎 Búsqueda semántica de imágenes en Qdrant...")
        candidatas: Dict[str, dict] = {}
        try:
            results = (await asyncio.to_thread(
                self.client.query_points,
                collection_name=COLLECTION_IMAGENES, query=texto_embedding,
                using="texto_emb", limit=top_k * 3,
            )).points
            for r in results:
                caption = r.payload.get("caption", "")
                texto_pag = r.payload.get("texto_pagina", "")
                ocr = r.payload.get("ocr_text", "")
                texto = caption or texto_pag or ocr
                candidatas[str(r.id)] = {
                    "id": str(r.id), "texto": texto, "fuente": r.payload.get("fuente", ""),
                    "imagen_path": r.payload.get("path"), "similitud": r.score,
                    "nombre_archivo": r.payload.get("nombre_archivo", ""),
                    "etiqueta": r.payload.get("etiqueta", ""), "caption_raw": caption,
                }
        except Exception as e:
            print(f"⚠️ Error en texto_emb de Qdrant: {e}")

        if not candidatas:
            print("   ℹ️ Sin imágenes candidatas")
            return []

        print(f"   📋 {len(candidatas)} candidatas — re-ranking semántico...")
        q_emb = np.array(texto_embedding)

        def _rerank(cands: list) -> list:
            salida = []
            for img_dict in cands:
                caption_full = img_dict.get("texto", "") or img_dict.get("caption_raw", "")
                if not caption_full or len(caption_full.strip()) < 10:
                    continue
                img_path = img_dict.get("imagen_path", "")
                nombre = img_dict.get("nombre_archivo", "").lower()
                if not img_path or not os.path.exists(img_path) or "_full." in nombre:
                    continue
                try:
                    caption_emb = np.array(embed_query_con_reintento(embeddings_model, caption_full[:500]))
                    titulo = caption_full.split("\n")[0].strip()
                    sim_caption = float(q_emb @ caption_emb / (np.linalg.norm(q_emb) * np.linalg.norm(caption_emb) + 1e-10))
                    sim_titulo = 0.0
                    if titulo and len(titulo) >= 5:
                        titulo_emb = np.array(embed_query_con_reintento(embeddings_model, titulo))
                        sim_titulo = float(q_emb @ titulo_emb / (np.linalg.norm(q_emb) * np.linalg.norm(titulo_emb) + 1e-10))
                    sim = max(sim_titulo, sim_caption) * 0.7 + min(sim_titulo, sim_caption) * 0.3
                    salida.append({
                        "id": img_dict["id"], "path": img_path, "caption": caption_full[:500],
                        "nombre_archivo": img_dict.get("nombre_archivo", os.path.basename(img_path)),
                        "etiqueta": img_dict.get("etiqueta", ""), "fuente": img_dict.get("fuente", ""),
                        "similitud_semantica": sim,
                    })
                except Exception:
                    continue
            return salida

        # Embedding inference is CPU/GPU-bound — keep it off the event loop.
        rankeadas = await asyncio.to_thread(_rerank, list(candidatas.values()))
        rankeadas.sort(key=lambda x: x["similitud_semantica"], reverse=True)
        if rankeadas:
            top3 = [(r["nombre_archivo"], round(r["similitud_semantica"], 3)) for r in rankeadas[:3]]
            print(f"   📊 Re-ranking top-3: {top3}")

        # Umbral de similitud caption↔consulta (MiniLM). 0.55 era demasiado
        # exigente: matches válidos rondan ~0.4 y nunca pasaban. Los filtros
        # posteriores (fuente dominante + keywords en el caption) limpian el
        # resto, así que un umbral más bajo es seguro. Tunable por env.
        UMBRAL = float(os.getenv("IMG_RERANK_UMBRAL", "0.35"))
        filtradas = [r for r in rankeadas if r["similitud_semantica"] >= UMBRAL]

        # Filter by dominant source
        consulta_kw = entidades.get("_consulta", [])
        if filtradas and len(filtradas) > 1:
            top_sim = filtradas[0]["similitud_semantica"]
            top_fuente = filtradas[0].get("fuente", "")
            if top_fuente:
                coherentes = [
                    r for r in filtradas
                    if r.get("fuente") == top_fuente or r["similitud_semantica"] >= top_sim - 0.01
                ]
                if coherentes:
                    descartadas = len(filtradas) - len(coherentes)
                    if descartadas:
                        print(f"   🔍 Filtro de fuente: descartadas {descartadas} de otras fuentes")
                    filtradas = coherentes

        # Filter by keyword presence in caption
        if filtradas and consulta_kw and len(filtradas) > 1:
            kw_lower = [_sin_tildes(k.lower()) for k in consulta_kw if len(k) > 4]
            if kw_lower:
                con_match = [r for r in filtradas if any(k in _sin_tildes((r.get("caption", "") or "").lower()) for k in kw_lower)]
                sin_match = [r for r in filtradas if r not in con_match]
                if con_match:
                    if sin_match:
                        print(f"   🔍 Filtro keywords: descartadas {len(sin_match)} sin coincidencia")
                    filtradas = con_match

        if filtradas:
            print(f"   ✅ {len(filtradas)} imágenes con similitud >= {UMBRAL}")
        else:
            print(f"   ⚠️ Ninguna imagen superó umbral ({UMBRAL})")

        return filtradas[:top_k]

    async def busqueda_hibrida(
        self,
        texto_embedding,
        imagen_embedding_uni,
        imagen_embedding_plip,
        entidades,
        top_k: int = 10,
        incluir_imagenes_texto: bool = False,
    ) -> list:
        res_pag_chunks = []
        res_img_texto = []
        res_keyword = []

        # Independent read-only searches run concurrently; each offloads its
        # blocking qdrant-client call via asyncio.to_thread.
        async def _buscar_texto():
            return await self.busqueda_vectorial(texto_embedding, INDEX_TEXTO, top_k) if texto_embedding else []

        async def _buscar_uni():
            if not imagen_embedding_uni:
                return []
            return [r for r in await self.busqueda_vectorial(imagen_embedding_uni, INDEX_UNI, top_k) if r.get("similitud", 0) >= 0.80]

        async def _buscar_plip():
            if not imagen_embedding_plip:
                return []
            return [r for r in await self.busqueda_vectorial(imagen_embedding_plip, INDEX_PLIP, top_k) if r.get("similitud", 0) >= 0.80]

        res_texto, res_uni, res_plip, res_ent = await asyncio.gather(
            _buscar_texto(), _buscar_uni(), _buscar_plip(),
            self.busqueda_por_entidades(entidades, top_k),
        )

        tiene_imagen = imagen_embedding_uni is not None or imagen_embedding_plip is not None

        if tiene_imagen:
            # res_uni / res_plip ya vienen filtrados por similitud >= 0.80,
            # así que tomamos directamente los mejores resultados.
            top_img = res_uni + res_plip
            for img_r in top_img[:3]:
                fuente = img_r.get("fuente", "")
                pagina = img_r.get("pagina")
                if fuente and pagina is not None:
                    res_pag_chunks.extend(await self.busqueda_chunks_por_pagina(fuente, pagina))

        if texto_embedding and incluir_imagenes_texto:
            try:
                raw_img = (await asyncio.to_thread(
                    self.client.query_points,
                    collection_name=COLLECTION_IMAGENES, query=texto_embedding,
                    using="texto_emb", limit=top_k,
                )).points
                for r in raw_img:
                    if "_full." in (r.payload.get("nombre_archivo", "").lower()):
                        continue
                    if r.score > 0.1:
                        caption = r.payload.get("caption", "")
                        texto_pag = r.payload.get("texto_pagina", "")
                        res_img_texto.append({
                            "id": str(r.id),
                            "texto": caption or texto_pag or r.payload.get("ocr_text", ""),
                            "fuente": r.payload.get("fuente", ""), "tipo": "imagen",
                            "imagen_path": r.payload.get("path"), "similitud": r.score,
                            "nombre_archivo": r.payload.get("nombre_archivo", ""),
                            "etiqueta": r.payload.get("etiqueta", ""),
                            "pagina": r.payload.get("pagina"),
                        })
            except Exception:
                pass

        # Keyword fallback for text-only queries with weak vector results
        if not tiene_imagen:
            consulta_kw = entidades.get("_consulta", [])
            tejidos = entidades.get("tejidos", [])
            estructuras = entidades.get("estructuras", [])
            terminos = tejidos + estructuras + consulta_kw
            top_vector_sim = max((r.get("similitud", 0) for r in res_texto), default=0)
            estructuras_especificas = [
                t for t in estructuras
                if len(str(t).split()) >= 2 or any(
                    clave in str(t).lower()
                    for clave in ("lamina", "lámina", "tunica", "túnica", "sertoli", "leydig")
                )
            ]
            if terminos and (top_vector_sim < 0.50 or estructuras_especificas):
                res_keyword = await self.busqueda_chunks_por_texto(terminos, top_k)
                dominios = set(entidades.get("dominios", []) or [])
                if dominios and res_keyword:
                    res_keyword = [
                        r for r in res_keyword
                        if not r.get("dominios") or dominios.intersection(set(r.get("dominios", [])))
                    ]

        # Weighted merge
        combined: Dict[str, dict] = {}

        def agregar(resultados, peso, es_visual=False):
            for r in resultados:
                key = r.get("id") or f"{r.get('fuente')}_{str(r.get('texto', ''))[:40]}"
                if not r.get("texto") and not r.get("imagen_path"):
                    continue
                sim_ponderada = r.get("similitud", 0) * peso
                if es_visual and r.get("similitud", 0) > 0.95:
                    sim_ponderada += 2.0
                if key not in combined:
                    combined[key] = {**r, "similitud": sim_ponderada}
                else:
                    combined[key]["similitud"] += sim_ponderada

        if tiene_imagen:
            agregar(res_texto, 0.80)
            agregar(res_uni, 0.60, es_visual=True)
            agregar(res_plip, 0.60, es_visual=True)
            agregar(res_ent, 0.50)
            agregar(res_pag_chunks, 0.15)
        else:
            agregar(res_texto, 0.80)
            agregar(res_uni, 0.20, es_visual=True)
            agregar(res_plip, 0.20, es_visual=True)
            agregar(res_img_texto, 0.40)
            agregar(res_ent, 0.60)
            agregar(res_keyword, 1.00)

        final = sorted(combined.values(), key=lambda x: x["similitud"], reverse=True)

        print(
            f"   📊 Híbrida: Txt={len(res_texto)} | UNI={len(res_uni)} | "
            f"PLIP={len(res_plip)} | Ent={len(res_ent)} | "
            f"ImgTxt={len(res_img_texto)} | Keyword={len(res_keyword)} → {len(final)}"
        )
        return final[:15]
