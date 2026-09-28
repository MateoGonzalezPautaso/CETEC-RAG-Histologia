"""
AsistenteHistologiaQdrant — main RAG orchestrator.

Owns the LangGraph pipeline and all node implementations.
All public entry points are async.
"""

import base64
import glob
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import torch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .classifier import ClasificadorSemantico
from .config import (
    COLLECTION_CHUNKS, COLLECTION_IMAGENES, DIRECTORIO_IMAGENES, DIRECTORIO_PDFS,
    FEATURES_DISCRIMINATORIAS, PROMPTS_OPTIMIZADOS, PROMPTS_OPTIMIZADOS_PATH,
    QDRANT_PATH, SIMILARITY_THRESHOLD, TRAZAS_DIR, TRAZAS_HABILITADAS,
    _safe, normalizar,
)
from .embeddings import PlipWrapper, UniWrapper
from .extractors import ExtractorEntidades, ExtractorImagenesPDF, ExtractorTemario
from .graph import AgentState
from .llm import (
    embed_query_con_reintento, invoke_con_reintento,
    userdata,
)
from .memory import SemanticMemory
from .prompts import CLAVE_RESPUESTA_TEXTO, INSTRUCCION_BASE_TEXTO, INSTRUCCION_PROSA, PromptsOptimizados
from .qdrant_store import QdrantVectorStore
from .trazas import RegistroTrazas, construir_traza


class AsistenteHistologiaQdrant:

    SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLD

    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.uni: Optional[UniWrapper] = None
        self.plip: Optional[PlipWrapper] = None

        # Memoria por sesión: cada user_id tiene su propia conversación / imagen
        # activa. Comparten un único cliente Qdrant (_memoria_qdrant) porque el
        # Qdrant embebido toma un lock exclusivo sobre el directorio.
        self._memorias: Dict[str, SemanticMemory] = {}
        self._memoria_qdrant = None
        self.qdrant_store: Optional[QdrantVectorStore] = None
        self.extractor_imagenes = ExtractorImagenesPDF(DIRECTORIO_IMAGENES)
        self.extractor_temario: Optional[ExtractorTemario] = None
        self.extractor_entidades: Optional[ExtractorEntidades] = None
        self.clasificador_semantico: Optional[ClasificadorSemantico] = None

        self.graph = None
        self.compiled_graph = None
        self.memory_saver = None
        self.contenido_base = ""
        self.registro_trazas = RegistroTrazas(TRAZAS_DIR, habilitado=TRAZAS_HABILITADAS)
        self.prompts = PromptsOptimizados(PROMPTS_OPTIMIZADOS_PATH, habilitado=PROMPTS_OPTIMIZADOS)

        self.device = self._detect_device()
        print(f"✅ AsistenteHistologiaQdrant v5.0 inicializado en {self.device}")

    def _detect_device(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        try:
            cap = torch.cuda.get_device_capability(0)
            if cap[0] < 7:
                print(f"⚠️ GPU incompatible (sm_{cap[0]}{cap[1]}). Forzando CPU.")
                return "cpu"
            return "cuda"
        except Exception:
            return "cpu"

    # ── Initialization ────────────────────────────────────────────────────────

    async def inicializar_componentes(self):
        self._init_modelos()
        # Prepara la colección de memoria y el cliente compartido (sesión default).
        self._get_memoria("default_user")
        self.extractor_temario = ExtractorTemario(llm=self.llm)
        self.extractor_entidades = ExtractorEntidades(llm=self.llm)
        self.clasificador_semantico = ClasificadorSemantico(
            llm=self.llm, embeddings=self.embeddings,
            device=self.device, temario=[],
        )
        self.qdrant_store = QdrantVectorStore(
            url=userdata.get("QDRANT_URL") or os.getenv("QDRANT_URL"),
            api_key=userdata.get("QDRANT_KEY") or os.getenv("QDRANT_KEY"),
            path=QDRANT_PATH,
        )
        await self.qdrant_store.connect()
        await self.qdrant_store.crear_esquema()

        self.memory_saver = MemorySaver()
        self._crear_grafo()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory_saver)
        print("✅ Todos los componentes inicializados")

    def _init_modelos(self):
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=userdata.get("GROQ_API_KEY"),
            temperature=0, max_retries=1,
        )
        print("✅ Groq inicializado")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": self.device},
        )
        print("✅ Embeddings HuggingFace inicializados")

        self.plip = PlipWrapper(self.device)
        self.plip.load()
        self.uni = UniWrapper(self.device)
        self.uni.load()

    # ── LangGraph ─────────────────────────────────────────────────────────────

    def _crear_grafo(self):
        g = StateGraph(AgentState)

        g.add_node("inicializar",          self._nodo_inicializar)
        g.add_node("procesar_imagen",      self._nodo_procesar_imagen)
        g.add_node("clasificar",           self._nodo_clasificar)
        g.add_node("generar_consulta",     self._nodo_generar_consulta)
        g.add_node("buscar_qdrant",        self._nodo_buscar_qdrant)
        g.add_node("filtrar_contexto",     self._nodo_filtrar_contexto)
        g.add_node("analisis_comparativo", self._nodo_analisis_comparativo)
        g.add_node("generar_respuesta",    self._nodo_generar_respuesta)
        g.add_node("finalizar",            self._nodo_finalizar)
        g.add_node("fuera_temario",        self._nodo_fuera_temario)

        g.add_edge(START, "inicializar")
        g.add_conditional_edges("inicializar", self._route_por_modo,
                                {"con_imagen": "procesar_imagen", "solo_texto": "clasificar"})
        g.add_edge("procesar_imagen", "clasificar")
        g.add_conditional_edges("clasificar", self._route_por_temario,
                                {"en_temario": "generar_consulta", "fuera_temario": "fuera_temario"})
        g.add_edge("fuera_temario", "finalizar")
        g.add_edge("generar_consulta", "buscar_qdrant")
        g.add_edge("buscar_qdrant", "filtrar_contexto")
        g.add_conditional_edges("filtrar_contexto", self._route_analisis_comparativo,
                                {"con_imagen": "analisis_comparativo", "sin_imagen": "generar_respuesta"})
        g.add_edge("analisis_comparativo", "generar_respuesta")
        g.add_edge("generar_respuesta", "finalizar")
        g.add_edge("finalizar", END)

        self.graph = g

    # ── Memoria por sesión ──────────────────────────────────────────────────────

    def _get_memoria(self, user_id: Optional[str]) -> SemanticMemory:
        """Devuelve (creándola si hace falta) la memoria de la sesión `user_id`.

        Cada sesión tiene su propio historial e imagen activa; todas comparten un
        único cliente Qdrant para no chocar con el lock del Qdrant embebido.
        """
        user_id = user_id or "default_user"
        mem = self._memorias.get(user_id)
        if mem is None:
            mem = SemanticMemory(
                llm=self.llm, embeddings=self.embeddings,
                uni=self.uni, plip=self.plip,
                qdrant_client=self._memoria_qdrant,
            )
            # La primera instancia crea el cliente local; lo reusamos en las demás.
            if self._memoria_qdrant is None:
                self._memoria_qdrant = mem._qdrant
            self._memorias[user_id] = mem
        return mem

    # ── Routers ───────────────────────────────────────────────────────────────

    async def _route_por_modo(self, state: AgentState) -> str:
        imagen_path = state.get("imagen_path")
        tiene_imagen_nueva = imagen_path and os.path.exists(imagen_path)
        tiene_imagen_memoria = self._get_memoria(state.get("user_id")).tiene_imagen_previa()

        if tiene_imagen_nueva:
            print("🖼️ Modo multimodal (imagen nueva)")
            return "con_imagen"

        if tiene_imagen_memoria:
            consulta = state.get("consulta_texto", "")
            try:
                resp = await invoke_con_reintento(self.llm, [
                    SystemMessage(content=(
                        "Sos un clasificador estricto. El usuario tiene una imagen histológica "
                        "subida previamente. Determiná si su consulta ACTUAL necesita analizar "
                        "esa imagen, o si es una pregunta teórica.\n\n"
                        "REGLA: En caso de duda, respondé TEXTO. Solo respondé IMAGEN si la "
                        "consulta CLARAMENTE necesita ver/analizar la imagen subida.\n"
                        "Respondé SOLO con 'IMAGEN' o 'TEXTO'."
                    )),
                    HumanMessage(content=f"CONSULTA: {consulta}"),
                ])
                resultado = resp.content.strip().upper()
                print(f"   🤖 Clasificador de modo: '{consulta}' → {resultado}")
                if resultado.startswith("IMAGEN"):
                    print("🖼️ Modo multimodal (imagen en memoria + referencia en consulta)")
                    return "con_imagen"
            except Exception as e:
                print(f"   ⚠️ Error clasificador de modo, fallback a texto: {e}")

        print("📝 Modo solo texto")
        return "solo_texto"

    def _route_por_temario(self, state: AgentState) -> str:
        return "en_temario" if state.get("tema_valido", True) else "fuera_temario"

    def _route_analisis_comparativo(self, state: AgentState) -> str:
        if state.get("tiene_imagen") and state.get("imagen_path"):
            return "con_imagen"
        return "sin_imagen"

    # ── Nodes ─────────────────────────────────────────────────────────────────

    async def _nodo_inicializar(self, state: AgentState) -> AgentState:
        print("📝 Inicializando flujo v5.0...")
        memoria = self._get_memoria(state.get("user_id"))
        consulta_original = state.get("consulta_texto", "")
        historial = memoria.get_history_for_prompt(5)
        state["historial_conversacional"] = historial

        consulta = await self._reescribir_consulta_con_contexto(consulta_original, historial)
        if consulta != consulta_original:
            state["consulta_texto"] = consulta

        state["mostrar_imagenes"] = await self._detectar_solicitud_imagen(consulta_original)
        if state["mostrar_imagenes"]:
            print("   🖼️ Solicitud de imagen detectada")
            consulta_imagen = self._resolver_pedido_imagen_ambiguo(state.get("consulta_texto", consulta_original), historial)
            if consulta_imagen != state.get("consulta_texto", consulta_original):
                state["consulta_texto"] = consulta_imagen

        state.update({
            "contexto_memoria": memoria.get_context(state.get("consulta_texto", "")),
            "contenido_base": self.contenido_base,
            "tiempo_inicio": time.time(),
            "tiene_imagen": False,
            "imagen_es_nueva": False,
            "contexto_suficiente": False,
            "resultados_validos": [],
            "terminos_busqueda": "",
            "entidades_consulta": {"tejidos": [], "estructuras": [], "tinciones": []},
            "imagenes_recuperadas": [],
            "tema_valido": True,
            "tema_encontrado": None,
            "temario": self.extractor_temario.temas if self.extractor_temario else [],
            "analisis_comparativo": None,
            "estructura_identificada": None,
            "texto_embedding": None,
            "similitud_semantica_dominio": 0.0,
            "trayectoria": [{"nodo": "Inicializar", "tiempo": 0}],
        })
        return state

    async def _nodo_procesar_imagen(self, state: AgentState) -> AgentState:
        t0 = time.time()
        print("🖼️ Procesando imagen...")

        memoria = self._get_memoria(state.get("user_id"))
        imagen_path_nuevo = state.get("imagen_path")
        imagen_es_nueva = False

        if imagen_path_nuevo and os.path.exists(imagen_path_nuevo):
            imagen_path_activo = imagen_path_nuevo
            imagen_es_nueva = True
            print(f"   🆕 Nueva imagen: {imagen_path_activo}")
        elif memoria.tiene_imagen_previa():
            imagen_path_activo = memoria.get_imagen_activa()
            state["imagen_path"] = imagen_path_activo
            state["analisis_visual"] = memoria.analisis_visual_activo
            print(f"   ♻️  Reutilizando imagen del turno {memoria.imagen_turno_subida}")
        else:
            imagen_path_activo = None

        if imagen_path_activo and os.path.exists(imagen_path_activo):
            try:
                emb_u = self.uni.embed_image(imagen_path_activo, preprocess=True)
                emb_p = self.plip.embed_image(imagen_path_activo, preprocess=True)
                state["imagen_embedding_uni"] = emb_u.tolist()
                state["imagen_embedding_plip"] = emb_p.tolist()
                state["tiene_imagen"] = True
                state["imagen_es_nueva"] = imagen_es_nueva

                if imagen_es_nueva or not state.get("analisis_visual"):
                    state["analisis_visual"] = await self._describir_imagen(imagen_path_activo)
                    memoria.set_imagen(imagen_path_activo, state["analisis_visual"])
                    print(f"   🔬 Análisis visual generado ({len(state['analisis_visual'])} chars)")
                else:
                    print("   ♻️  Reutilizando análisis visual previo")

                print(f"✅ Imagen lista | nueva={imagen_es_nueva}")
            except Exception as e:
                print(f"❌ Error imagen: {e}")
                state["imagen_embedding_uni"] = None
                state["imagen_embedding_plip"] = None
                state["analisis_visual"] = None
                state["tiene_imagen"] = False
        else:
            print("ℹ️ Sin imagen — modo texto")
            state["analisis_visual"] = None
            state["tiene_imagen"] = False
            state["imagen_es_nueva"] = False

        state["trayectoria"].append({
            "nodo": "ProcesarImagen", "tiene_imagen": state["tiene_imagen"],
            "imagen_es_nueva": imagen_es_nueva, "tiempo": round(time.time() - t0, 2),
        })
        return state

    async def _nodo_clasificar(self, state: AgentState) -> AgentState:
        t0 = time.time()
        print("🔍 Clasificando consulta...")

        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=(
                    "Extrae términos técnicos histológicos de la consulta.\n"
                    "Devuelve:\nTEJIDO: [...]\nESTRUCTURA: [...]\nCONCEPTO: [...]\n"
                    "TINCIÓN: [...]\nTÉRMINOS_CLAVE: [...]"
                )),
                HumanMessage(content="\n\n".join(filter(None, [
                    f"CONSULTA:\n{state['consulta_texto']}",
                    (f"CONTEXTO:\n{state.get('contexto_memoria', '')[:300]}"
                     if state.get('contexto_memoria') and state.get('contexto_memoria') != "No hay consultas previas." else ""),
                ]))),
            ])
            state["terminos_busqueda"] = resp.content
        except Exception:
            state["terminos_busqueda"] = state["consulta_texto"]

        state["entidades_consulta"] = await self.extractor_entidades.extraer_de_texto(state["consulta_texto"])
        print(f"   🏷️ Entidades: {state['entidades_consulta']}")

        try:
            state["texto_embedding"] = embed_query_con_reintento(self.embeddings, state["consulta_texto"])
        except Exception as e:
            print(f"⚠️ Error embedding texto: {e}")
            state["texto_embedding"] = None

        verificacion = await self.clasificador_semantico.clasificar(
            consulta=state["consulta_texto"],
            analisis_visual=state.get("analisis_visual"),
            imagen_activa=state.get("tiene_imagen", False),
            temario_muestra=state.get("temario", [])[:60],
        )
        state["tema_valido"] = verificacion.get("valido", True)
        state["tema_encontrado"] = verificacion.get("tema_encontrado")
        state["similitud_semantica_dominio"] = verificacion.get("similitud_dominio", 0.0)

        print(f"   📚 Válido: {state['tema_valido']} | "
              f"Tema: {state['tema_encontrado'] or 'N/A'} | "
              f"Sim: {state['similitud_semantica_dominio']:.3f} | "
              f"Método: {verificacion.get('metodo')}")

        state["trayectoria"].append({
            "nodo": "Clasificar", "tema_valido": state["tema_valido"],
            "tema_encontrado": state["tema_encontrado"],
            "entidades": state["entidades_consulta"],
            "similitud_dominio": state["similitud_semantica_dominio"],
            "metodo_clasificacion": verificacion.get("metodo"),
            "tiempo": round(time.time() - t0, 2),
        })
        return state

    async def _nodo_fuera_temario(self, state: AgentState) -> AgentState:
        t0 = time.time()
        print("🚫 Consulta fuera del dominio histológico")
        temario = state.get("temario") or []
        muestra = "\n".join(f"  • {t}" for t in temario[:20])
        if len(temario) > 20:
            muestra += f"\n  ... y {len(temario)-20} más"
        state["respuesta_final"] = (
            "⚠️ **Consulta fuera del dominio disponible**\n\n"
            "Tu consulta no parece estar relacionada con histología, patología "
            "o morfología tisular/celular.\n\n"
            f"**Temas disponibles (muestra):**\n{muestra}\n\n"
            "Si tenés una imagen histológica, subila y reformulá tu pregunta."
        )
        state["contexto_suficiente"] = False
        state["trayectoria"].append({"nodo": "FueraTemario", "tiempo": round(time.time() - t0, 2)})
        return state

    async def _nodo_generar_consulta(self, state: AgentState) -> AgentState:
        t0 = time.time()
        tema_extra = f"\nTEMA: {state['tema_encontrado']}" if state.get("tema_encontrado") else ""
        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=(
                    "Genera consultas cortas (≤8 palabras) para histología.\n"
                    "Formato:\nCONSULTA_TEXTO: <texto>\n"
                    + ("CONSULTA_VISUAL: <visual>" if state.get("tiene_imagen") else "")
                )),
                HumanMessage(content=(
                    f"TÉRMINOS:\n{_safe(state.get('terminos_busqueda'))}"
                    f"{tema_extra}\nCONSULTA: {state['consulta_texto']}"
                )),
            ])
            contenido = resp.content
            ct = state["consulta_texto"][:77]
            cv = ""
            if "CONSULTA_TEXTO:" in contenido:
                after = contenido.split("CONSULTA_TEXTO:")[1]
                if "CONSULTA_VISUAL:" in after:
                    ct = after.split("CONSULTA_VISUAL:")[0].strip()[:77]
                    cv = after.split("CONSULTA_VISUAL:")[1].strip()[:77]
                else:
                    ct = after.strip()[:77]
            state["consulta_busqueda_texto"] = ct
            state["consulta_busqueda_visual"] = cv
        except Exception:
            state["consulta_busqueda_texto"] = state["consulta_texto"][:77]
            state["consulta_busqueda_visual"] = ""

        print(f"   📝 query='{state['consulta_busqueda_texto']}'")
        state["trayectoria"].append({
            "nodo": "GenerarConsulta", "query": state["consulta_busqueda_texto"],
            "tiempo": round(time.time() - t0, 2),
        })
        return state

    async def _nodo_buscar_qdrant(self, state: AgentState) -> AgentState:
        t0 = time.time()
        print("📚 Búsqueda híbrida Qdrant...")

        entidades = dict(state.get("entidades_consulta", {}))
        consulta_texto = state.get("consulta_busqueda_texto") or state.get("consulta_texto", "")
        stopwords = {
            "para", "como", "sobre", "este", "esta", "estos", "estas",
            "podes", "puede", "mostrar", "mostrame", "dame", "quiero",
            "tipo", "tipos", "explicar", "describir", "decir", "imagen",
            "imagenes", "ejemplo", "favor", "hablar", "habla",
            "cuál", "cual", "cuáles", "cuales", "qué", "que", "cómo",
            "donde", "dónde", "tiene",
        }
        palabras_consulta = [
            w.strip("¿?¡!.,;:()\"'") for w in consulta_texto.lower().split()
            if len(w.strip("¿?¡!.,;:()\"'")) > 3
            and w.strip("¿?¡!.,;:()\"'") not in stopwords
        ]
        entidades["_consulta"] = palabras_consulta

        pide_visual = await self._detectar_retrieval_visual(state.get("consulta_texto", ""), state)
        if pide_visual and not state.get("mostrar_imagenes", False):
            print("   🖼️ Consulta de reconocimiento visual: se incluye texto de imágenes")

        resultados = await self.qdrant_store.busqueda_hibrida(
            texto_embedding=state.get("texto_embedding"),
            imagen_embedding_uni=state.get("imagen_embedding_uni"),
            imagen_embedding_plip=state.get("imagen_embedding_plip"),
            entidades=entidades,
            top_k=10,
            incluir_imagenes_texto=pide_visual,
        )

        state["resultados_busqueda"] = resultados
        print(f"✅ {len(resultados)} resultados")

        # Image search when user explicitly requests images
        if state.get("mostrar_imagenes", False):
            print("   🖼️ Búsqueda semántica de imágenes...")
            imgs = await self.qdrant_store.busqueda_imagenes_semantica(
                # La clave siempre existe (se inicializa en None), así que el
                # default [] de .get nunca aplicaba; lo forzamos a [] si es None.
                texto_embedding=state.get("texto_embedding") or [],
                entidades=entidades,
                embeddings_model=self.embeddings,
                top_k=3,
            )
            if not imgs or len(imgs) < 2:
                print("   🔤 Fallback: keyword search en etiquetas/captions...")
                imgs_kw = await self.qdrant_store.buscar_imagenes_por_referencia(palabras_consulta, top_k=3)
                paths_ya = {img.get("path", "") for img in imgs}
                for img_kw in imgs_kw:
                    kw_path = img_kw.get("imagen_path", "")
                    if kw_path and kw_path not in paths_ya:
                        imgs.append({
                            "id": img_kw.get("id", ""), "path": kw_path,
                            "caption": img_kw.get("caption_raw", img_kw.get("texto", ""))[:500],
                            "nombre_archivo": img_kw.get("nombre_archivo", os.path.basename(kw_path)),
                            "etiqueta": img_kw.get("etiqueta", ""),
                            "fuente": img_kw.get("fuente", ""),
                            "similitud_semantica": 0.90,
                        })
                        paths_ya.add(kw_path)
                imgs = imgs[:3]

            state["imagenes_para_mostrar"] = imgs
            if imgs and not state.get("contexto_suficiente"):
                state["contexto_suficiente"] = True

        state["trayectoria"].append({
            "nodo": "BuscarQdrant", "hits": len(resultados),
            "imagenes_para_mostrar": len(state.get("imagenes_para_mostrar", [])),
            "tiempo": round(time.time() - t0, 2),
        })
        return state

    async def _nodo_filtrar_contexto(self, state: AgentState) -> AgentState:
        t0 = time.time()
        es_solo_texto = not state.get("tiene_imagen", False)
        umbral_texto = 0.30 if es_solo_texto else 0.5
        umbral_imagen = self.SIMILARITY_THRESHOLD

        validos = []
        for r in state["resultados_busqueda"]:
            sim = r.get("similitud", 0)
            if r.get("tipo") == "texto" and sim < umbral_texto:
                continue
            if r.get("tipo") == "imagen" and sim < umbral_imagen:
                continue
            if r.get("tipo") == "imagen":
                img_p = r.get("imagen_path")
                if not img_p or not os.path.exists(img_p):
                    continue
            validos.append(r)

        if es_solo_texto and validos:
            validos = self._filtrar_por_fuente_dominante(validos, state)
            validos = sorted(validos, key=lambda x: x.get("similitud", 0), reverse=True)[:6]

        state["resultados_validos"] = validos
        tiene_imgs = len(state.get("imagenes_para_mostrar", [])) > 0
        state["contexto_suficiente"] = len(validos) > 0 or tiene_imgs

        vistas: set = set()
        imagenes_unicas: List[str] = []
        imagenes_texto: Dict[str, str] = {}
        for r in validos:
            img_path = r.get("imagen_path")
            if img_path and os.path.exists(img_path) and img_path not in vistas:
                vistas.add(img_path)
                imagenes_unicas.append(img_path)
                imagenes_texto[img_path] = _safe(r.get("texto", ""))[:500]
        state["imagenes_recuperadas"] = imagenes_unicas
        state["imagenes_texto_map"] = imagenes_texto

        if validos:
            validos_sorted = sorted(validos, key=lambda x: x.get("similitud", 0), reverse=True)
            bloques = []
            for i, r in enumerate(validos_sorted, 1):
                es_top_img = i == 1 and r.get("tipo") == "imagen"
                marcador = " ⭐ MEJOR MATCH VISUAL" if es_top_img else ""
                enc = (
                    f"[Sección {i}{marcador} | Fuente: {r.get('fuente','N/A')} | "
                    f"Tipo: {r.get('tipo','?')} | Sim: {r.get('similitud',0):.3f}"
                )
                if r.get("imagen_path"):
                    enc += f" | Imagen: {os.path.basename(r['imagen_path'])}"
                enc += "]"
                bloques.append(f"{enc}\n{_safe(r.get('texto',''))[:700]}")
            state["contexto_documentos"] = "\n\n".join(bloques)
            modo_str = "TEXTO" if es_solo_texto else "IMAGEN+TEXTO"
            print(f"✅ {len(validos)} válidos | {len(imagenes_unicas)} imgs | Modo: {modo_str}")
        else:
            state["contexto_documentos"] = ""
            print(f"⚠️ Ningún resultado supera umbral (texto={umbral_texto}, img={umbral_imagen})")

        state["trayectoria"].append({
            "nodo": "FiltrarContexto", "hits_validos": len(validos),
            "imgs": len(imagenes_unicas),
            "modo": "solo_texto" if es_solo_texto else "multimodal",
            "tiempo": round(time.time() - t0, 2),
        })
        return state

    def _filtrar_por_fuente_dominante(self, resultados: list, state: AgentState) -> list:
        """Keep results from the dominant source; discard cross-source noise."""
        if len(resultados) <= 6:
            return resultados

        ordenados = sorted(resultados, key=lambda x: x.get("similitud", 0), reverse=True)
        top = [r for r in ordenados[:6] if r.get("fuente")]
        conteo: Dict[str, int] = {}
        for r in top:
            f = r.get("fuente", "")
            conteo[f] = conteo.get(f, 0) + 1

        if not conteo:
            return ordenados

        fuente_dom, n_dom = max(conteo.items(), key=lambda x: x[1])
        if n_dom < 3 or n_dom / max(len(top), 1) < 0.6:
            return ordenados

        consulta_norm = normalizar(" ".join([
            state.get("consulta_texto", ""),
            state.get("consulta_busqueda_texto", ""),
        ]))
        stop = {"como", "cuales", "cual", "tiene", "tienen", "para", "sobre",
                "entre", "donde", "esta", "este", "estas", "estos"}
        keywords = [
            p.strip("¿?¡!.,;:()\"'") for p in consulta_norm.split()
            if len(p.strip("¿?¡!.,;:()\"'")) > 4 and p.strip("¿?¡!.,;:()\"'") not in stop
        ]

        filtrados = []
        descartados = 0
        for r in ordenados:
            if r.get("fuente") == fuente_dom:
                filtrados.append(r)
            else:
                texto = normalizar(r.get("texto", ""))
                if sum(1 for k in keywords if k in texto) >= 2:
                    filtrados.append(r)
                else:
                    descartados += 1
        if descartados:
            print(f"   🔎 Filtro fuente dominante: {fuente_dom} | descartados={descartados}")
        return filtrados or ordenados

    async def _nodo_analisis_comparativo(self, state: AgentState) -> AgentState:
        t0 = time.time()
        if not state.get("tiene_imagen") or not state.get("imagen_path"):
            state["trayectoria"].append({"nodo": "AnalisisComparativo", "motivo": "sin imagen", "tiempo": round(time.time() - t0, 2)})
            return state

        imagenes_ref = [p for p in state.get("imagenes_recuperadas", [])[:3] if os.path.exists(p)]
        if not imagenes_ref:
            print("ℹ️ Sin referencias — análisis comparativo omitido")
            state["analisis_comparativo"] = None
            state["contexto_documentos"] = ""
            state["contexto_suficiente"] = False
            state["imagenes_recuperadas"] = []
            state["trayectoria"].append({"nodo": "AnalisisComparativo", "motivo": "sin referencias", "tiempo": round(time.time() - t0, 2)})
            return state

        print(f"🔬 Análisis comparativo vs {len(imagenes_ref)} referencias...")
        content_parts = [{"type": "text", "text": (
            "Compara la imagen de consulta con las referencias del manual para "
            "determinar si corresponden a la misma estructura histológica.\n\n"
            "=== IMAGEN DE CONSULTA ==="
        )}]

        try:
            with open(state["imagen_path"], "rb") as f:
                data_u = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(state["imagen_path"])[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data_u}"}})
        except Exception as e:
            print(f"⚠️ No se pudo cargar imagen usuario: {e}")
            state["analisis_comparativo"] = None
            return state

        analisis_previo = _safe(state.get("analisis_visual"))
        if analisis_previo:
            content_parts.append({"type": "text", "text": f"\nAnálisis previo:\n{analisis_previo[:600]}\n"})

        imagenes_texto = state.get("imagenes_texto_map", {})
        for i, ref_path in enumerate(imagenes_ref, 1):
            texto_ref = imagenes_texto.get(ref_path, "Sin descripción disponible")
            content_parts.append({"type": "text", "text": (
                f"\n=== REFERENCIA #{i} ({os.path.basename(ref_path)}) ===\n"
                f"DESCRIPCIÓN DEL MANUAL: {texto_ref}"
            )})
            try:
                with open(ref_path, "rb") as f:
                    data_r = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(ref_path)[1].lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data_r}"}})
            except Exception as e:
                print(f"  ⚠️ No se pudo cargar {ref_path}: {e}")

        features_lista = "\n".join(f"  - {f}" for f in FEATURES_DISCRIMINATORIAS)
        content_parts.append({"type": "text", "text": (
            "\n=== INSTRUCCIONES ===\n"
            f"Compara rigurosamente basándote en:\n{features_lista}\n\n"
            "IMPORTANTE: La DESCRIPCIÓN DEL MANUAL es la fuente de verdad.\n\n"
            "1. TABLA COMPARATIVA (Markdown): | Feature | Consulta | Ref#1 | Ref#2 |\n"
            "2. VEREDICTO DE IDENTIDAD: Evalúa si son el MISMO TEJIDO o TEJIDOS DIFERENTES.\n"
            "3. Terminá con UNA de estas frases EXACTAS:\n"
            "   - 'CONCLUSIÓN: SÍ son la misma estructura histológica'\n"
            "   - 'CONCLUSIÓN: TEJIDOS DIFERENTES'"
        )})

        try:
            resp = await invoke_con_reintento(self.llm, [HumanMessage(content=content_parts)])
            state["analisis_comparativo"] = resp.content
            print(f"✅ Análisis comparativo: {len(resp.content)} chars")
        except Exception as e:
            print(f"❌ Error análisis comparativo: {e}")
            state["analisis_comparativo"] = None

        analisis_str = state.get("analisis_comparativo", "")
        son_diferentes = bool(analisis_str and (
            "TEJIDOS DIFERENTES" in analisis_str or "NO son la misma" in analisis_str
        ))

        estructura_db = None
        if not son_diferentes:
            imagenes_rec = state.get("imagenes_recuperadas", [])
            if imagenes_rec and imagenes_texto:
                top_caption = imagenes_texto.get(imagenes_rec[0], "")
                if top_caption:
                    primera_linea = top_caption.split("\n")[0].strip()
                    m = re.match(r'Imagen\s+[\d.]+[A-Za-z]?\.\s*(.*)', primera_linea)
                    estructura_db = m.group(1).rstrip(".") if m else primera_linea
                    print(f"   → Estructura (DB): {estructura_db}")
        else:
            print("   → Tejidos DIFERENTES. Limpiando contexto.")
            state["contexto_documentos"] = ""
            state["contexto_suficiente"] = False
            state["imagenes_recuperadas"] = []

        state["estructura_identificada"] = estructura_db
        state["trayectoria"].append({
            "nodo": "AnalisisComparativo", "refs": len(imagenes_ref),
            "estructura": estructura_db, "tiempo": round(time.time() - t0, 2),
        })
        return state

    async def _nodo_generar_respuesta(self, state: AgentState) -> AgentState:
        t0 = time.time()
        es_solo_texto = not state.get("tiene_imagen", False)
        modo_str = "TEXTO" if es_solo_texto else "MULTIMODAL"
        print(f"💭 Generando respuesta v5.0 [{modo_str}]...")

        imagenes_para_mostrar = state.get("imagenes_para_mostrar", [])
        tiene_imgs_mostrar = state.get("mostrar_imagenes", False) and len(imagenes_para_mostrar) > 0

        if not state["contexto_suficiente"] and not tiene_imgs_mostrar:
            state["respuesta_final"] = self._respuesta_sin_contexto(es_solo_texto, state)
            state["trayectoria"].append({
                "nodo": "GenerarRespuesta", "contexto_suficiente": False,
                "modo": "solo_texto" if es_solo_texto else "multimodal",
                "tiempo": round(time.time() - t0, 2),
            })
            return state

        # Build context from images if no text context available
        if tiene_imgs_mostrar and not state.get("contexto_documentos"):
            bloques = []
            for i, img_info in enumerate(imagenes_para_mostrar, 1):
                bloques.append(
                    f"[Sección {i} | Fuente: {img_info.get('fuente','')} | Tipo: imagen | "
                    f"Sim: {img_info.get('similitud_semantica',0):.3f} | "
                    f"Imagen: {img_info.get('nombre_archivo','')}]\n"
                    f"Etiqueta: {img_info.get('etiqueta','')}\n"
                    f"Descripción: {img_info.get('caption','')[:500]}"
                )
            state["contexto_documentos"] = "\n\n".join(bloques)

        system_prompt = self._build_system_prompt(es_solo_texto, state)
        content_parts = self._build_content_parts(es_solo_texto, state)

        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content_parts),
            ])
            state["respuesta_final"] = resp.content

            # Auto-append source citation if missing
            respuesta_norm = state["respuesta_final"].lower()
            tiene_cita = "[manual:" in respuesta_norm or "fuente:" in respuesta_norm or re.search(r"\[[^\]]*\.pdf[^\]]*\]", respuesta_norm)
            es_sin_contexto = (
                respuesta_norm.startswith("este tema no se encuentra")
                or "consulta fuera del dominio" in respuesta_norm
                or respuesta_norm.startswith("error:")
            )
            if es_solo_texto and not tiene_cita and not es_sin_contexto:
                fuentes = list(dict.fromkeys(
                    r.get("fuente") for r in state.get("resultados_validos", []) if r.get("fuente")
                ))
                if fuentes:
                    state["respuesta_final"] += f"\n\n[Manual: {', '.join(fuentes[:3])}]"

            print(f"✅ Respuesta: {len(resp.content)} chars")
        except Exception as e:
            print(f"❌ Error: {e}")
            state["respuesta_final"] = f"Error: {e}"

        # Replace image references with exact matches from Qdrant
        if state.get("mostrar_imagenes") and state.get("respuesta_final"):
            state = await self._actualizar_imagenes_desde_respuesta(state)

        imagenes_usadas = len([
            p for p in state.get("imagenes_recuperadas", [])[:3]
            if not es_solo_texto and os.path.exists(p)
        ])
        state["trayectoria"].append({
            "nodo": "GenerarRespuesta", "contexto_suficiente": True,
            "imagenes_usadas": imagenes_usadas,
            "tiene_comparativo": bool(_safe(state.get("analisis_comparativo"))),
            "tiempo": round(time.time() - t0, 2),
        })
        return state

    def _respuesta_sin_contexto(self, es_solo_texto: bool, state: AgentState) -> str:
        temario = state.get("temario") or []
        if es_solo_texto:
            muestra = "\n".join(f"  • {t}" for t in temario[:15])
            if len(temario) > 15:
                muestra += f"\n  ... y {len(temario)-15} más"
            return (
                "⚠️ **No encontré información específica sobre eso en el manual**\n\n"
                "La consulta es válida pero no encontré contenido suficiente en la "
                "base de datos.\n\n"
                f"**Temas disponibles (muestra):**\n{muestra}\n\n"
                "Podés intentar:\n"
                "- Reformular con términos más específicos\n"
                "- Subir una imagen histológica para análisis visual"
            )
        else:
            muestra = "\n".join(f"  • {t}" for t in temario[:20])
            if len(temario) > 20:
                muestra += f"\n  ... y {len(temario)-20} más"
            return (
                "⚠️ **Imagen no encontrada en la base de datos**\n\n"
                "La imagen no coincide con ninguna estructura documentada en el manual.\n\n"
                f"**Temas disponibles (muestra):**\n{muestra}"
            )

    def _build_system_prompt(self, es_solo_texto: bool, state: AgentState) -> str:
        temario = state.get("temario", [])
        ontologia_str = ""
        if temario:
            temas_muestra = temario[:40]
            ontologia_str = (
                "\nONTOLOGÍA DISPONIBLE (temas del manual):\n"
                + "\n".join(f"  • {t}" for t in temas_muestra)
            )
            if len(temario) > 40:
                ontologia_str += f"\n  ... y {len(temario) - 40} temas más."
            ontologia_str += (
                "\n\nIMPORTANTE: Si la consulta trata sobre un tema que NO está en esta "
                "ontología y NO aparece en las SECCIONES DEL MANUAL, indicá que no está "
                "disponible. NO inventes contenido.\n"
            )

        instruccion_prosa = INSTRUCCION_PROSA
        instruccion_continuidad = (
            "- Adaptá tu respuesta como continuación natural del diálogo, "
            "sin repetir información ya proporcionada.\n"
            if state.get("historial_conversacional") else ""
        )

        instruccion_imagenes = ""
        if state.get("mostrar_imagenes") and state.get("imagenes_para_mostrar"):
            imgs = state["imagenes_para_mostrar"]
            descripciones = [
                f"{i}. **{img.get('etiqueta') or img.get('nombre_archivo')}**: {img.get('caption','')[:300]}"
                for i, img in enumerate(imgs, 1)
            ]
            instruccion_imagenes = (
                "\nIMÁGENES DE REFERENCIA QUE SE MUESTRAN AL USUARIO (debajo de tu respuesta):\n"
                + "\n".join(descripciones)
                + "\n\nINSTRUCCIÓN SOBRE LAS IMÁGENES:\n"
                "- Estas imágenes del manual se muestran automáticamente debajo de tu respuesta.\n"
                "- NO afirmes que 'no hay imágenes' ni cuestiones su existencia o disponibilidad.\n"
                "- Enfocate en explicar el tema consultado; podés referirte a las imágenes con "
                "naturalidad (p. ej. 'en las imágenes de referencia se observa...') y describirlas "
                "brevemente con su caption del manual.\n"
            )

        if es_solo_texto:
            # La instrucción optimizada por GEPA reemplaza la parte estática
            # (reglas + estilo); lo dinámico se sigue agregando igual.
            optimizada = self.prompts.instruccion(CLAVE_RESPUESTA_TEXTO)
            if optimizada:
                return (
                    f"{optimizada}\n\n"
                    f"{ontologia_str}\n"
                    f"{instruccion_continuidad}{instruccion_imagenes}"
                    f"{self.prompts.demos_texto(CLAVE_RESPUESTA_TEXTO)}"
                )
            return (
                f"{INSTRUCCION_BASE_TEXTO}"
                f"{ontologia_str}\n"
                f"{instruccion_prosa}{instruccion_continuidad}{instruccion_imagenes}"
            )
        else:
            tiene_comparativo = bool(_safe(state.get("analisis_comparativo")))
            estructura_str = _safe(state.get("estructura_identificada"))
            instruccion_estructura = ""
            if estructura_str:
                instruccion_estructura = (
                    f"\n⚠️ ESTRUCTURA IDENTIFICADA: {estructura_str}\n"
                    "Esta identificación tiene MÁXIMA PRIORIDAD. Usá la descripción del "
                    "manual para ESTA ESTRUCTURA específicamente.\n"
                )
            regla_validacion = (
                "VALIDACIÓN: Revisa el 'ANÁLISIS COMPARATIVO'. "
                "Si dice 'TEJIDOS DIFERENTES', respondé que no se encontró en el manual. "
                "Si dice 'SÍ son la misma estructura', detallá el tejido según el manual.\n"
            )
            return (
                "Eres un asistente de histología. Responde SOLO con el contenido del manual.\n\n"
                "REGLAS:\n"
                "1. PRIORIDAD: La DESCRIPCIÓN TEXTUAL DEL MANUAL es la fuente de verdad.\n"
                "2. Cita: [Manual: archivo] | [Imagen: archivo]\n"
                "3. NO hagas diagnósticos basados en tu interpretación visual.\n\n"
                f"{ontologia_str}\n"
                f"{instruccion_estructura}"
                f"{instruccion_prosa}{instruccion_continuidad}{instruccion_imagenes}"
                f"{regla_validacion}"
                + ("\n\nIMPORTANTE: El análisis comparativo tiene PRIORIDAD en el diagnóstico." if tiene_comparativo else "")
            )

    def _build_content_parts(self, es_solo_texto: bool, state: AgentState) -> list:
        content_parts = []
        historial_str = _safe(state.get("historial_conversacional"))
        if historial_str:
            content_parts.append({"type": "text", "text": f"**HISTORIAL:**\n{historial_str}\n\n---\n"})

        ctx_docs = state["contexto_documentos"]
        if len(ctx_docs) > 4000:
            ctx_docs = ctx_docs[:4000] + "\n... [contexto truncado]"

        analisis_comp_str = _safe(state.get("analisis_comparativo"))
        estructura_str = _safe(state.get("estructura_identificada"))
        analisis_visual_str = _safe(state.get("analisis_visual"), "No disponible")
        seccion_comp = f"\n\n**ANÁLISIS COMPARATIVO:**\n{analisis_comp_str[:2000]}" if analisis_comp_str else ""
        seccion_est = f"\n\n**ESTRUCTURA IDENTIFICADA:** {estructura_str}" if estructura_str else ""

        content_parts.append({"type": "text", "text": (
            f"**CONSULTA:** {state['consulta_texto']}\n\n"
            f"**TÉRMINOS:** {_safe(state.get('terminos_busqueda'))[:300]}\n\n"
            f"**ENTIDADES:** {json.dumps(state.get('entidades_consulta', {}), ensure_ascii=False)}\n\n"
            f"**TEMA:** {_safe(state.get('tema_encontrado'), 'N/A')}\n\n"
            f"**ANÁLISIS VISUAL:**\n{analisis_visual_str[:800]}\n\n"
            f"**SECCIONES DEL MANUAL:**\n{ctx_docs}"
            f"{seccion_comp}{seccion_est}\n\n"
            "Responde EXCLUSIVAMENTE con el contenido del manual e imágenes de referencia."
        )})

        imagen_path = state.get("imagen_path")
        if state.get("tiene_imagen") and imagen_path and os.path.exists(imagen_path):
            try:
                with open(imagen_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(imagen_path)[1].lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                label = ("NUEVA IMAGEN" if state.get("imagen_es_nueva")
                         else f"IMAGEN ACTIVA (turno {self._get_memoria(state.get('user_id')).imagen_turno_subida})")
                content_parts.append({"type": "text", "text": f"\n**{label}:**"})
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
            except Exception as e:
                print(f"   ⚠️ No se pudo añadir imagen usuario: {e}")

        if not es_solo_texto:
            for img_path in state.get("imagenes_recuperadas", [])[:3]:
                if not os.path.exists(img_path):
                    continue
                try:
                    with open(img_path, "rb") as f:
                        data = base64.b64encode(f.read()).decode("utf-8")
                    ext = os.path.splitext(img_path)[1].lower()
                    mime = "image/png" if ext == ".png" else "image/jpeg"
                    content_parts.append({"type": "text", "text": f"\n**REFERENCIA [Imagen: {os.path.basename(img_path)}]:**"})
                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
                except Exception as e:
                    print(f"   ⚠️ {img_path}: {e}")

        return content_parts

    async def _actualizar_imagenes_desde_respuesta(self, state: AgentState) -> AgentState:
        """
        Parse image references from the LLM response and replace the
        semantic-search image list with exact-match results from Qdrant.
        """
        try:
            refs = re.findall(
                r'\*\*(?:Imagen|imagen)\s+(\d+)\*\*|(?:Imagen|imagen)\s+(\d+)',
                state["respuesta_final"],
            )
            numeros = list(dict.fromkeys(m[0] or m[1] for m in refs if (m[0] or m[1])))
            if not numeros:
                return state

            etiquetas_buscadas = {f"imagen {n}" for n in numeros}
            print(f"   🔍 Imágenes mencionadas en respuesta: {[f'Imagen {n}' for n in numeros]}")

            fuentes_count: Dict[str, int] = {}
            for r in state.get("resultados_validos", []):
                f = r.get("fuente", "")
                if f:
                    fuentes_count[f] = fuentes_count.get(f, 0) + 1
            fuente_dominante = max(fuentes_count, key=fuentes_count.get) if fuentes_count else None

            all_imgs = await self.qdrant_store._scroll_all(COLLECTION_IMAGENES)
            nuevas: List[dict] = []
            vistas: set = set()
            for r in all_imgs:
                etiqueta = (r.payload.get("etiqueta", "") or "").strip().lower()
                nombre = (r.payload.get("nombre_archivo", "") or "").lower()
                if "_full." in nombre or etiqueta not in etiquetas_buscadas:
                    continue
                img_path = r.payload.get("path", "")
                if not img_path or not os.path.exists(img_path) or img_path in vistas:
                    continue
                vistas.add(img_path)
                caption = r.payload.get("caption", "") or ""
                nuevas.append({
                    "id": str(r.id), "path": img_path, "caption": caption[:500],
                    "nombre_archivo": r.payload.get("nombre_archivo", os.path.basename(img_path)),
                    "etiqueta": r.payload.get("etiqueta", ""),
                    "fuente": r.payload.get("fuente", ""),
                    "similitud_semantica": 0.95,
                })

            if fuente_dominante and nuevas:
                de_fuente = [img for img in nuevas if img.get("fuente") == fuente_dominante]
                de_otra = [img for img in nuevas if img not in de_fuente]
                if de_fuente:
                    nuevas = de_fuente + de_otra

            orden = {f"imagen {n}": i for i, n in enumerate(numeros)}
            nuevas.sort(key=lambda x: orden.get(x.get("etiqueta", "").strip().lower(), 999))

            if nuevas:
                state["imagenes_para_mostrar"] = nuevas[:3]
                print(f"   ✅ Reemplazadas {len(state['imagenes_para_mostrar'])} imgs exactas de la respuesta")
        except Exception as e:
            print(f"   ⚠️ Error extrayendo imágenes de la respuesta: {e}")
        return state

    async def _nodo_finalizar(self, state: AgentState) -> AgentState:
        if state.get("respuesta_final"):
            self._get_memoria(state.get("user_id")).add_interaction(
                state["consulta_texto"], state["respuesta_final"])

        total = round(time.time() - state["tiempo_inicio"], 2)
        state["trayectoria"].append({"nodo": "Finalizar", "tiempo_total": total})

        print(f"✅ Flujo v5.0 completado en {total}s")
        if state.get("estructura_identificada"):
            print(f"   → Estructura: {state['estructura_identificada']}")
        return state

    # ── Intent detection helpers ──────────────────────────────────────────────

    def _detectar_retrieval_visual_por_reglas(self, consulta: str) -> Optional[bool]:
        q = re.sub(r"\s+", " ", normalizar(consulta)).strip()
        if not q:
            return False

        visuales_explicitos = ("imagen", "imagenes", "figura", "figuras", "foto",
                               "fotos", "microfotografia", "microfotografias", "laminilla")
        if any(t in q for t in visuales_explicitos):
            return True

        visuales_seguros = ("se observa", "se reconoc", "se identifica", "se ve",
                            "que se observa", "que se ve", "que veo", "que estoy viendo",
                            "que muestra", "que aparece", "identifica esto",
                            "identificar esto", "reconoce esto", "detalle histologico")
        if any(p in q for p in visuales_seguros):
            return True

        conceptuales = ("que es ", "define", "defini", "explica", "explicame",
                        "funcion", "funciones", "caracteristica", "caracteristicas",
                        "componentes", "capas", "diferencia", "diferencias",
                        "describe la funcion", "cual es la funcion", "cuales son")
        if any(p in q for p in conceptuales):
            return False

        return None

    async def _detectar_retrieval_visual(self, consulta: str, state: AgentState) -> bool:
        if state.get("mostrar_imagenes", False):
            return True

        decision = self._detectar_retrieval_visual_por_reglas(consulta)
        if decision is not None:
            return decision

        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=(
                    "Clasificador de intención RAG de histología. "
                    "¿La consulta necesita captions/imágenes del manual o solo texto conceptual?\n\n"
                    "VISUAL: el usuario quiere reconocer, observar, identificar o comparar una estructura.\n"
                    "TEXTO: el usuario pregunta definiciones, funciones o características generales.\n"
                    "Respondé SOLO con 'VISUAL' o 'TEXTO'."
                )),
                HumanMessage(content=f"CONSULTA: {consulta}"),
            ])
            decision_llm = resp.content.strip().upper()
            print(f"   🧭 Intención visual (LLM): {decision_llm}")
            return decision_llm.startswith("VISUAL")
        except Exception as e:
            print(f"   ⚠️ Error clasificador visual, fallback: {e}")
            return bool(state.get("tiene_imagen") or state.get("imagen_path"))

    async def _detectar_solicitud_imagen(self, consulta: str) -> bool:
        consulta_norm = normalizar(consulta)
        patrones = [
            "mostrame una imagen", "muestrame una imagen", "mostrarme una imagen",
            "mostrame imagenes", "muestrame imagenes", "quiero ver imagenes",
            "quiero ver una imagen", "podes mostrarme una imagen",
            "imagen del manual", "imagenes del manual", "foto del manual",
        ]
        if any(p in consulta_norm for p in patrones):
            return True

        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=(
                    "Determiná si la consulta solicita EXPLÍCITAMENTE ver, mostrar o buscar "
                    "imágenes en la base de datos o manual.\n"
                    "Solo respondé 'SI' si la intención es pedir que el sistema muestre una imagen.\n"
                    "Respondé SOLO con 'SI' o 'NO'."
                )),
                HumanMessage(content=f"CONSULTA: {consulta}"),
            ])
            return resp.content.strip().upper().startswith("SI")
        except Exception as e:
            print(f"⚠️ Error detección de solicitud de imagen: {e}")
            return False

    def _resolver_pedido_imagen_ambiguo(self, consulta: str, historial: str) -> str:
        consulta_norm = normalizar(consulta)
        es_generico = any(p in consulta_norm for p in [
            "mostrame una imagen", "muestrame una imagen", "mostrarme una imagen",
            "podes mostrarme una imagen", "quiero ver una imagen", "quiero ver imagenes",
        ])
        if not es_generico:
            return consulta

        historial_norm = normalizar(historial)
        temas_prioridad = [
            ("celulas de Leydig", ["leydig"]),
            ("celulas de Sertoli", ["sertoli"]),
            ("arteria muscular", ["arteria muscular", "tunica media", "lamina elastica"]),
            ("testiculo", ["testiculo", "seminifero", "espermatogenesis"]),
        ]
        for tema, pistas in temas_prioridad:
            if any(p in historial_norm for p in pistas):
                reescrita = f"Mostrame imagenes de {tema} del manual"
                print(f"   🔄 Pedido contextualizado: '{consulta}' → '{reescrita}'")
                return reescrita

        # General fallback: pick the most recently mentioned syllabus topic from
        # the history, so this works for any topic (not just the hardcoded few).
        temas = self.extractor_temario.temas if self.extractor_temario else []
        candidatos = []
        for tema in temas:
            tema_norm = normalizar(tema)
            if len(tema_norm) <= 4:
                continue
            pos = historial_norm.rfind(tema_norm)
            if pos != -1:
                candidatos.append((pos, tema))
        if candidatos:
            _, tema = max(candidatos, key=lambda x: x[0])
            reescrita = f"Mostrame imagenes de {tema} del manual"
            print(f"   🔄 Pedido contextualizado (temario): '{consulta}' → '{reescrita}'")
            return reescrita
        return consulta

    async def _reescribir_consulta_con_contexto(self, consulta: str, historial: str) -> str:
        if not historial:
            return consulta
        try:
            # Single round-trip: rewrite if the query references prior context,
            # otherwise the model returns it unchanged.
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=(
                    "Si la CONSULTA ACTUAL hace referencia a temas de la conversación previa "
                    "(usa pronombres como 'eso', 'esto', o pide 'más sobre' algo previo), "
                    "reescribila para que sea autocontenida resolviendo esas referencias con el "
                    "historial. Si ya es autocontenida, devolvela EXACTAMENTE igual.\n\n"
                    "REGLAS:\n"
                    "- Devolvé SOLO la consulta final, sin explicaciones.\n"
                    "- Máximo 30 palabras.\n"
                    "- NO generes respuestas ni inventes contenido."
                )),
                HumanMessage(content=f"HISTORIAL:\n{historial}\n\nCONSULTA ACTUAL: {consulta}"),
            ])
            reescrita = resp.content.strip()
            if reescrita and reescrita != consulta and len(reescrita) < 200:
                print(f"   🔄 Reescrita: '{consulta}' → '{reescrita}'")
                return reescrita
            if reescrita and len(reescrita) >= 200:
                print(f"   ⚠️ Reescritura descartada (demasiado larga: {len(reescrita)} chars)")
            return consulta
        except Exception as e:
            print(f"   ⚠️ Error reescribiendo consulta: {e}")
            return consulta

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_texto(self, texto: str) -> List[float]:
        return embed_query_con_reintento(self.embeddings, texto)

    # ── Image analysis ────────────────────────────────────────────────────────

    async def _describir_imagen(self, imagen_path: str) -> str:
        try:
            with open(imagen_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(imagen_path)[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            features_lista = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(FEATURES_DISCRIMINATORIAS))
            msg = HumanMessage(content=[
                {"type": "text", "text": (
                    "Describe esta imagen histológica con máximo rigor.\n\n"
                    "PARTE 1 — DESCRIPCIÓN OBJETIVA (sin nombrar el tejido):\n"
                    "Describí SOLO lo que ves: coloración, forma y disposición de células, "
                    "presencia/ausencia de canalículos, vasos, fibras, capas. NO clasifiques.\n\n"
                    f"PARTE 2 — FEATURES DISCRIMINATORIAS:\n{features_lista}\n\n"
                    "PARTE 3 — DIAGNÓSTICO DIFERENCIAL (mínimo 3 opciones):\n"
                    "Para cada opción, listá evidencias a favor y en contra."
                )},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
            ])
            resp = await invoke_con_reintento(self.llm, [msg])
            return resp.content
        except Exception as e:
            print(f"⚠️ Error describiendo imagen: {e}")
            return ""

    # ── Indexing ──────────────────────────────────────────────────────────────

    def procesar_contenido_base(self, directorio: str = DIRECTORIO_PDFS) -> str:
        pdfs = glob.glob(os.path.join(directorio, "*.pdf"))
        if not pdfs:
            print(f"⚠️ Sin PDFs en {directorio}")
            return ""
        self.contenido_base = "\n".join(self._leer_pdf(p) for p in pdfs)
        print(f"📚 {len(pdfs)} PDFs leídos ({len(self.contenido_base)} chars)")
        return self.contenido_base[:500]

    async def extraer_y_preparar_temario(self):
        if not self.contenido_base:
            print("⚠️ Contenido base vacío")
            return
        await self.extractor_temario.extraer_temario(self.contenido_base)
        if self.clasificador_semantico:
            self.clasificador_semantico.temario = self.extractor_temario.temas
            print(f"   🔄 Clasificador actualizado con {len(self.extractor_temario.temas)} temas")

    async def indexar_en_qdrant(
        self, directorio_pdfs: str = DIRECTORIO_PDFS,
        imagen_files_extra: Optional[List[str]] = None,
        forzar: bool = False,
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        marker_path = os.path.join(base_dir, ".qdrant_index_complete")

        if not forzar:
            try:
                n_chunks = self.qdrant_store.client.count(collection_name=COLLECTION_CHUNKS).count
                n_imgs = self.qdrant_store.client.count(collection_name=COLLECTION_IMAGENES).count
                if n_chunks > 0 and n_imgs > 0 and os.path.exists(marker_path):
                    print(f"✅ BD ya poblada ({n_chunks} chunks, {n_imgs} imágenes). Saltando indexación.")
                    return
                if n_chunks > 0 and n_imgs > 0:
                    print("⚠️ BD poblada pero sin marca de completitud — reindexando para garantizar consistencia.")
            except Exception as e:
                print(f"⚠️ No se pudo verificar estado de la BD: {e}")

        print("📄 Extrayendo imágenes para vincular a chunks...")
        imagenes_pdf = self.extractor_imagenes.extraer_de_directorio(directorio_pdfs)
        img_por_pdf_pag: Dict[tuple, list] = {}
        for img_info in imagenes_pdf:
            k = (img_info["fuente_pdf"], img_info["pagina"])
            img_por_pdf_pag.setdefault(k, []).append(img_info["path"])

        fallos = 0
        print("📄 Indexando chunks de texto...")
        for pdf_path in glob.glob(os.path.join(directorio_pdfs, "*.pdf")):
            fuente = os.path.basename(pdf_path)
            paginas_texto = self._leer_pdf_por_paginas(pdf_path)
            for num_pagina, texto_pag in paginas_texto.items():
                chunks = self._chunks(texto_pag)
                img_paths = img_por_pdf_pag.get((fuente, num_pagina), [])
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    try:
                        emb = self._embed_texto(chunk)
                        chunk_id = f"chunk_{fuente}_p{num_pagina}_{i}"
                        entidades = self.extractor_entidades.extraer_de_texto_sync(chunk)
                        await self.qdrant_store.upsert_chunk(
                            chunk_id=chunk_id, texto=chunk, fuente=fuente,
                            chunk_idx=i, embedding=emb, entidades=entidades,
                            pagina=num_pagina, imagenes_pagina=img_paths,
                        )
                    except Exception as e:
                        fallos += 1
                        print(f"  ⚠️ Chunk {fuente} p{num_pagina}-{i}: {e}")

        print("📸 Indexando imágenes...")
        for img_info in imagenes_pdf:
            img_path = img_info["path"]
            if not os.path.exists(img_path):
                continue
            try:
                emb_u = self.uni.embed_image(img_path, preprocess=False)
                emb_p = self.plip.embed_image(img_path, preprocess=False)
                img_id = f"img_{img_info['fuente_pdf']}_{img_info['pagina']}_{os.path.basename(img_path)}"
                caption = img_info.get("caption", "")
                etiqueta = img_info.get("etiqueta", "")
                nombre_archivo = img_info.get("nombre_archivo", os.path.basename(img_path))

                titulo_caption = caption.split("\n")[0].strip() if caption else ""
                partes_emb = []
                if etiqueta and titulo_caption:
                    partes_emb.extend([f"{etiqueta}: {titulo_caption}", titulo_caption])
                elif titulo_caption:
                    partes_emb.extend([titulo_caption, titulo_caption])
                if caption:
                    partes_emb.append(caption[:300])
                texto_relevante = "\n".join(partes_emb) or img_info.get("texto_pagina", "")[:500]

                emb_texto = None
                if texto_relevante:
                    try:
                        emb_texto = self._embed_texto(texto_relevante)
                    except Exception:
                        pass

                await self.qdrant_store.upsert_imagen(
                    imagen_id=img_id, path=img_path,
                    fuente=img_info["fuente_pdf"], pagina=img_info["pagina"],
                    ocr_text=img_info.get("ocr_text", ""),
                    texto_pagina=img_info.get("texto_pagina", ""),
                    emb_uni=emb_u.tolist(), emb_plip=emb_p.tolist(),
                    emb_texto=emb_texto, caption=caption,
                    nombre_archivo=nombre_archivo, etiqueta=etiqueta,
                )
            except Exception as e:
                fallos += 1
                print(f"  ⚠️ Imagen {img_path}: {e}")

        for img_path in (imagen_files_extra or []):
            if not os.path.exists(img_path):
                continue
            try:
                import pytesseract
                from PIL import Image
                ocr = pytesseract.image_to_string(Image.open(img_path)).strip()
            except Exception:
                ocr = ""
            try:
                emb_u = self.uni.embed_image(img_path, preprocess=True)
                emb_p = self.plip.embed_image(img_path, preprocess=True)
                emb_texto = self._embed_texto(ocr[:500]) if ocr else None
                await self.qdrant_store.upsert_imagen(
                    imagen_id=f"img_extra_{os.path.basename(img_path)}",
                    path=img_path, fuente=os.path.basename(img_path),
                    pagina=0, ocr_text=ocr[:300], texto_pagina="",
                    emb_uni=emb_u.tolist(), emb_plip=emb_p.tolist(), emb_texto=emb_texto,
                )
            except Exception as e:
                fallos += 1
                print(f"  ❌ Imagen extra {img_path}: {e}")

        if fallos == 0:
            try:
                with open(marker_path, "w") as f:
                    f.write("ok")
            except Exception as e:
                print(f"⚠️ No se pudo escribir marca de completitud: {e}")
        else:
            print(f"⚠️ {fallos} ítems fallaron — no se escribe marca de completitud "
                  "(se reintentará en el próximo arranque).")
        print("✅ Indexación Qdrant completada")

    # ── PDF utilities ─────────────────────────────────────────────────────────

    def _leer_pdf(self, path: str) -> str:
        try:
            doc = fitz.open(path)
            texto = "".join(page.get_text() for page in doc)
            doc.close()
            return texto
        except Exception as e:
            print(f"⚠️ Error leyendo {path}: {e}")
            return ""

    def _leer_pdf_por_paginas(self, path: str) -> Dict[int, str]:
        paginas: Dict[int, str] = {}
        try:
            doc = fitz.open(path)
            for i, page in enumerate(doc):
                paginas[i + 1] = page.get_text()
            doc.close()
        except Exception as e:
            print(f"⚠️ Error leyendo por páginas {path}: {e}")
        return paginas

    def _chunks(self, texto: str, size: int = 500) -> List[str]:
        return [texto[i:i + size] for i in range(0, len(texto), size)]

    # ── Public entry point ────────────────────────────────────────────────────

    async def consultar(
        self,
        consulta_texto: str,
        imagen_path: Optional[str] = None,
        user_id: str = "default_user",
        origen: str = "api",
    ) -> dict:
        memoria = self._get_memoria(user_id)
        t0 = time.time()
        tiene_imagen_activa = memoria.tiene_imagen_previa() or bool(imagen_path)

        print(f"\n{'='*70}")
        print(f"🔬 RAG Histología Qdrant v5.0 | umbral={self.SIMILARITY_THRESHOLD}")
        print(f"   Texto:          {consulta_texto}")
        print(f"   Imagen turno:   {imagen_path or 'ninguna'}")
        print(f"   Imagen memoria: {memoria.get_imagen_activa() or 'ninguna'}")
        print(f"{'='*70}")

        initial_state = AgentState(
            messages=[], consulta_texto=consulta_texto,
            imagen_path=imagen_path,
            imagen_embedding_uni=None, imagen_embedding_plip=None,
            texto_embedding=None, contexto_memoria="",
            contenido_base=self.contenido_base, terminos_busqueda="",
            entidades_consulta={"tejidos": [], "estructuras": [], "tinciones": []},
            consulta_busqueda_texto="", consulta_busqueda_visual="",
            resultados_busqueda=[], resultados_validos=[], contexto_documentos="",
            respuesta_final="", trayectoria=[], user_id=user_id,
            tiempo_inicio=time.time(),
            analisis_visual=None, tiene_imagen=False, imagen_es_nueva=False,
            contexto_suficiente=False, temario=self.extractor_temario.temas,
            tema_valido=True, tema_encontrado=None,
            imagenes_recuperadas=[], imagenes_texto_map={},
            analisis_comparativo=None, estructura_identificada=None,
            similitud_semantica_dominio=0.0,
            mostrar_imagenes=False, imagenes_para_mostrar=[],
            historial_conversacional="",
        )

        config = {
            "configurable": {"thread_id": user_id},
            "run_name": f"consulta-qdrant-v5.0-{user_id}",
            "tags": ["rag", "histologia", "qdrant", "v5.0"],
            "metadata": {
                "tiene_imagen_nueva": imagen_path is not None,
                "tiene_imagen_activa": tiene_imagen_activa,
                "consulta": consulta_texto[:100],
                "version": "5.0",
            },
        }

        error = None
        try:
            final = await self.compiled_graph.ainvoke(initial_state, config=config)
            respuesta = final["respuesta_final"]
        except Exception as e:
            import traceback; traceback.print_exc()
            respuesta = f"Error: {e}"
            error = respuesta
            final = {}

        traza = construir_traza(
            final=final, consulta_original=consulta_texto, session_id=user_id,
            imagen_subida=imagen_path, duracion_s=time.time() - t0,
            modelo=getattr(self.llm, "model_name", None), origen=origen, error=error,
            # El prompt optimizado solo aplica al modo texto.
            prompt=({"fuente": "multimodal"} if final.get("tiene_imagen")
                    else self.prompts.descripcion(CLAVE_RESPUESTA_TEXTO)),
        )
        self.registro_trazas.registrar(traza)

        print(f"\n{'='*70}\n📖 RESPUESTA:\n{'='*70}")
        print(respuesta)
        print("=" * 70)

        imagen_activa = memoria.get_imagen_activa()
        # Devolvemos el resultado completo (en vez de dejarlo en un atributo
        # compartido `_ultimo_resultado`) para que dos requests concurrentes no
        # se pisen: cada llamador se queda con su propio dict.
        resultado = {
            "respuesta": respuesta,
            "mostrar_imagenes": final.get("mostrar_imagenes", False),
            "imagenes_recuperadas": final.get("imagenes_recuperadas", []),
            "estructura_identificada": final.get("estructura_identificada"),
            "imagenes_para_mostrar": final.get("imagenes_para_mostrar", []),
            "trayectoria": final.get("trayectoria", []),
            "imagen_activa": os.path.basename(imagen_activa) if imagen_activa else None,
            "trace_id": traza["trace_id"],
            # La traza completa, para quien arme datasets (optimizacion/baseline.py)
            # sin depender de que el registro en disco esté habilitado.
            "traza": traza,
        }
        return resultado

    async def cerrar(self):
        if self.qdrant_store:
            await self.qdrant_store.close()
