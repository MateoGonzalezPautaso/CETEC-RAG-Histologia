"""
Evaluación RAGAS del RAG Histología Qdrant v5.0
================================================
Ejecuta el pipeline RAG completo contra un Golden Set de preguntas
con ground truths extraídos del manual, y evalúa la calidad usando
métricas RAGAS: faithfulness, context_precision, context_recall,
answer_relevancy.

Uso:
    python evaluar_ragas.py                  # Evaluación completa
    python evaluar_ragas.py --limit 5        # Solo 5 preguntas (rápido)
    python evaluar_ragas.py --no-ragas       # Solo retrieval, sin métricas LLM
    python evaluar_ragas.py --solo-ragas     # Solo juez RAGAS sobre JSON existente
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from src.assistant import AsistenteHistologiaQdrant
from src.config import DIRECTORIO_PDFS
from src.graph import AgentState

# ═══════════════════════════════════════════════════════════════════════
# GOLDEN SET — 32 preguntas con ground truth del manual
# (12 de arch3/arch4 + 20 de arch2)
# ═══════════════════════════════════════════════════════════════════════

GOLDEN_SET = [
    {
        "question": "Que capas se identifican en una arteria muscular y que estructuras tiene cada una?",
        "ground_truth": "Una arteria muscular presenta tunica intima con endotelio, lamina elastica interna que la separa de la tunica media, tunica media formada por musculo liso dispuesto en capas circunferenciales, y tunica adventicia de tejido conectivo colageno.",
        "fuente_esperada": "arch3.pdf",
        "paginas_esperadas": [1, 2, 6, 7],
    },
    {
        "question": "Como se reconoce la lamina elastica interna en la arteria muscular?",
        "ground_truth": "La lamina elastica interna se reconoce como una linea ondulada de color rosa translucido ubicada por fuera del endotelio y separando la tunica intima de la tunica media.",
        "fuente_esperada": "arch3.pdf",
        "paginas_esperadas": [2],
    },
    {
        "question": "Que caracteristicas histologicas tiene el tejido muscular liso de la tunica media?",
        "ground_truth": "El musculo liso de la tunica media se dispone en capas circunferenciales; sus celulas tienen citoplasma eosinofilo rosado y nucleos centrales alargados o ahusados en corte longitudinal, y mas redondeados en corte transversal.",
        "fuente_esperada": "arch3.pdf",
        "paginas_esperadas": [1, 2, 6],
    },
    {
        "question": "Que diferencia hay entre tunica intima, tunica media y tunica adventicia en la arteria muscular?",
        "ground_truth": "La tunica intima incluye el endotelio y se ubica hacia la luz; la tunica media contiene numerosas capas de musculo liso; la tunica adventicia corresponde a tejido conectivo colageno denso poco organizado.",
        "fuente_esperada": "arch3.pdf",
        "paginas_esperadas": [2, 6, 7],
    },
    {
        "question": "Que se observa en un corte transversal de una arteria muscular pequena?",
        "ground_truth": "En el corte transversal se observa la luz vascular con eritrocitos u otras celulas sanguineas, endotelio, lamina elastica interna, tunica media con musculo liso y tunica adventicia de tejido conectivo.",
        "fuente_esperada": "arch3.pdf",
        "paginas_esperadas": [2, 6],
    },
    {
        "question": "Como se organiza el tejido testicular humano adulto?",
        "ground_truth": "El testiculo esta rodeado por tunica albuginea que emite tabiques y divide el organo en lobulillos. Cada lobulillo contiene tubulos seminiferos y tejido intersticial con vasos y celulas de Leydig.",
        "fuente_esperada": "arch4.pdf",
        "paginas_esperadas": [1],
    },
    {
        "question": "Que celulas forman el epitelio seminifero del tubulo seminifero?",
        "ground_truth": "El epitelio seminifero contiene celulas germinales en distintos estadios de la espermatogenesis y celulas de Sertoli o sustentaculares, que se extienden desde la lamina basal hasta la luz tubular.",
        "fuente_esperada": "arch4.pdf",
        "paginas_esperadas": [1, 2, 13],
    },
    {
        "question": "Donde se ubican las celulas de Leydig y que rasgos histologicos presentan?",
        "ground_truth": "Las celulas de Leydig se ubican en el intersticio testicular, agrupadas entre los tubulos seminiferos, y presentan morfologia poligonal, citoplasma eosinofilo, nucleo central redondeado y organelas vinculadas con esteroidogenesis.",
        "fuente_esperada": "arch4.pdf",
        "paginas_esperadas": [15],
    },
    {
        "question": "Que caracteristicas tiene la celula de Sertoli?",
        "ground_truth": "La celula de Sertoli es una celula somatica sustentacular del epitelio seminifero; se extiende desde la lamina basal hasta la luz tubular y posee un nucleo grande, claro o eucromatico, frecuentemente con nucleolo evidente.",
        "fuente_esperada": "arch4.pdf",
        "paginas_esperadas": [2, 13],
    },
    {
        "question": "Como se describe la espermatogonia A oscura?",
        # Corregida (10/2026): la referencia anterior ("cromatina mas condensada u
        # oscura") contradecia al manual, que describe una rarefaccion central.
        "ground_truth": "Las espermatogonias A oscuras son celulas germinales con forma de cupula, las primeras desde la lamina basal hacia la luz tubular, que se distinguen por una zona de rarefaccion de la cromatina ubicada en el centro del nucleo.",
        "fuente_esperada": "arch4.pdf",
        "paginas_esperadas": [1, 6],
    },
    {
        "question": "Que son las celulas peritubulares o mioides del tubulo seminifero?",
        # Corregida (10/2026): actina, miosina y capacidad contractil no estan en el manual.
        "ground_truth": "Las celulas peritubulares o mioides son celulas alargadas con caracteristicas de miofibroblastos que, junto con la lamina basal del epitelio y las fibras colagenas, constituyen la pared del tubulo seminifero.",
        "fuente_esperada": "arch4.pdf",
        "paginas_esperadas": [2, 14],
    },
    {
        "question": "Que componentes se encuentran en el intersticio testicular?",
        # Corregida (10/2026): los vasos sanguineos no figuran en el manual, y la
        # descripcion del intersticio esta en la pagina 2, no en la 1.
        "ground_truth": "Alrededor de los tubulos seminiferos, el intersticio testicular esta constituido por un estroma de tejido conectivo laxo y por las celulas de Leydig o intersticiales: celulas poliedricas grandes y agrupadas, con nucleo central, cromatina laxa, nucleolo evidente y citoplasma acidofilo con inclusiones lipidicas.",
        "fuente_esperada": "arch4.pdf",
        "paginas_esperadas": [2, 15],
    },
    # ── arch2 (Sprint 1/2: cartílago, hueso, músculo, tejido nervioso) ──
    # Recuperadas del golden set del notebook v4.2. Van al final para que
    # `--limit 12` siga evaluando las mismas preguntas que el baseline previo.
    {
        "question": "Cuales son las dos capas del pericondrio y que celulas contiene cada una?",
        "ground_truth": "El pericondrio tiene una capa fibrosa con fibroblastos y fibras de colagena, y una capa condrogena con celulas condrogenas y condroblastos.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [1],
    },
    {
        "question": "Que estructuras se identifican en el cartilago hialino?",
        "ground_truth": "En el cartilago hialino se identifican nidos o grupos isogenos, condrocitos, condroblastos, y matriz territorial e interterritorial.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [1, 2],
    },
    {
        "question": "Que laminilla corresponde al cartilago hialino de traquea?",
        "ground_truth": "La laminilla numero 44 Traquea 44 ES corresponde al cartilago hialino, donde se senala el pericondrio con su capa fibrosa y condrogena.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [1],
    },
    {
        "question": "Cuales son las estructuras senaladas en la lamina de laringe 43?",
        "ground_truth": "En la lamina de laringe 43 se senalan nidos, condrocitos, matriz territorial e interterritorial del cartilago hialino.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [2],
    },
    {
        "question": "Que caracteristica distintiva tiene el cartilago elastico respecto al hialino?",
        "ground_truth": "El cartilago elastico contiene fibras elasticas en su matriz, ademas de nidos, condrocitos, condroblastos y matriz territorial e interterritorial.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [3],
    },
    {
        "question": "En que laminilla se observa el cartilago elastico y de que organo proviene?",
        "ground_truth": "El cartilago elastico se observa en la laminilla de piel de oreja de raton 85a.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [3],
    },
    {
        "question": "Que se observa en el cartilago hialino del disco de crecimiento?",
        "ground_truth": "En el cartilago hialino del disco de crecimiento se observan hileras de condrocitos. La laminilla corresponde a hueso fracturado 79c.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [4],
    },
    {
        "question": "Como se distingue el cartilago fibroso histologicamente?",
        "ground_truth": "El cartilago fibroso se distingue por hileras de condrocitos y fibras de colagena dispuestas regularmente.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [5],
    },
    {
        "question": "Cuales son los tipos de celulas del hueso compacto?",
        "ground_truth": "Las celulas del hueso compacto son osteoprogenitoras, osteoblastos, osteocitos y osteoclastos.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [7, 8, 9],
    },
    {
        "question": "Que es la osteona y que estructuras la componen?",
        # Corregida (10/2026): el manual no define la osteona; la referencia
        # anterior venia de un libro de texto. Se espera que el asistente lo diga.
        "ground_truth": "El manual no define la osteona ni describe sus componentes: la nombra entre las estructuras que el alumno debe identificar en el microscopio en la practica de tejido oseo (junto con el endostio, el periostio, los osteocitos, los osteoclastos y los conductos de Havers y de Volkmann) y la senala en la laminilla Hueso lijado 81 (Imagen 12.1).",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [9, 10],
    },
    {
        # El manual no explica la diferencia: solo nombra ambos conductos. La
        # referencia anterior traía información de fuera del manual; ahora la
        # pregunta controla que el sistema admita lo que el manual no dice.
        "question": "Que diferencia hay entre el conducto de Havers y el de Volkmann?",
        "ground_truth": "El manual no describe la diferencia entre ambos conductos: solo menciona el conducto de Havers y el conducto de Volckmann como estructuras del hueso compacto que se deben identificar en la practica 12, y los senala en el modelo de hueso compacto (Imagen 12.7).",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [9, 16],
    },
    {
        "question": "Que funcion tienen los osteoclastos y donde se ubican?",
        "ground_truth": "Los osteoclastos son celulas gigantes multinucleadas de 20 a 100 micras cuya funcion es la reabsorcion osea. Se encuentran en las lagunas de Howship.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [9],
    },
    {
        "question": "Cuales son los tres tipos de tejido muscular y sus caracteristicas principales?",
        "ground_truth": "Estriado voluntario con celulas cilindricas y nucleos perifericos, estriado involuntario o cardiaco con discos intercalares, y liso involuntario con celulas fusiformes y nucleo central.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [17],
    },
    {
        "question": "Que estructuras componen la sarcomera?",
        "ground_truth": "La sarcomera se extiende entre dos lineas Z e incluye banda A oscura, banda I clara, banda H dentro de la A, y linea M al centro de la H.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [17, 22],
    },
    {
        "question": "Que laminilla se usa para observar el tejido muscular liso involuntario?",
        "ground_truth": "Se usa la laminilla de estomago HE para observar los miocitos lisos involuntarios.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [21],
    },
    {
        "question": "Como se clasifican las neuronas segun el numero de prolongaciones?",
        "ground_truth": "Las neuronas se clasifican en unipolares, bipolares, pseudounipolares y multipolares.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [24],
    },
    {
        "question": "En que laminilla se observan las neuronas piriformes y en grano?",
        "ground_truth": "Las neuronas piriformes y en grano se observan en la laminilla de corteza cerebelosa 61 HE.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [27],
    },
    {
        "question": "Cual es la diferencia entre astrocito protoplasmático y fibroso?",
        "ground_truth": "El astrocito protoplasmático tiene prolongaciones cortas y gruesas en sustancia gris. El fibroso tiene prolongaciones largas y delgadas en sustancia blanca.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [31],
    },
    {
        "question": "Que funcion tienen los oligodendrocitos?",
        "ground_truth": "Los oligodendrocitos participan en la formacion de la mielina para la proteccion de los axones en el sistema nervioso central.",
        "fuente_esperada": "arch2.pdf",
        "paginas_esperadas": [31, 33],
    },
    {
        "question": "Que son las celulas ependimarias y donde se ubican?",
        "ground_truth": "Las celulas ependimarias revisten el canal ependimario y el conducto central de la medula espinal. Son cubicas o cilindricas.",
        "fuente_esperada": "arch2.pdf",
        # La descripción está en las páginas 30 y 31; la 35 es la imagen.
        "paginas_esperadas": [30, 31, 35],
    },
]


# ═══════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════

async def inicializar_asistente() -> AsistenteHistologiaQdrant:
    """Inicializa el asistente completo (modelos, Qdrant, temario)."""
    print("=" * 60)
    print("🚀 INICIALIZANDO ASISTENTE PARA EVALUACIÓN")
    print("=" * 60)

    asistente = AsistenteHistologiaQdrant()
    await asistente.inicializar_componentes()

    print("\n📚 Leyendo PDFs...")
    asistente.procesar_contenido_base(DIRECTORIO_PDFS)

    print("\n📋 Extrayendo temario...")
    await asistente.extraer_y_preparar_temario()
    n_temas = len(asistente.extractor_temario.temas)
    print(f"   → {n_temas} temas")

    print("\n💾 Verificando e indexando base de datos Qdrant...")
    await asistente.indexar_en_qdrant(DIRECTORIO_PDFS, forzar=False)

    return asistente


async def ejecutar_consulta_con_contexto(asistente, pregunta: str) -> dict:
    """
    Ejecuta el pipeline RAG y captura tanto la respuesta como los
    contextos intermedios necesarios para RAGAS.
    """
    initial_state = AgentState(
        messages=[],
        consulta_texto=pregunta,
        imagen_path=None,
        imagen_embedding_uni=None,
        imagen_embedding_plip=None,
        texto_embedding=None,
        contexto_memoria="",
        contenido_base=asistente.contenido_base,
        terminos_busqueda="",
        entidades_consulta={"tejidos": [], "estructuras": [], "tinciones": []},
        consulta_busqueda_texto="",
        consulta_busqueda_visual="",
        resultados_busqueda=[],
        resultados_validos=[],
        contexto_documentos="",
        respuesta_final="",
        trayectoria=[],
        user_id="eval_ragas",
        tiempo_inicio=time.time(),
        analisis_visual=None,
        tiene_imagen=False,
        imagen_es_nueva=False,
        contexto_suficiente=False,
        temario=asistente.extractor_temario.temas,
        tema_valido=True,
        tema_encontrado=None,
        imagenes_recuperadas=[],
        imagenes_texto_map={},
        analisis_comparativo=None,
        estructura_identificada=None,
        similitud_semantica_dominio=0.0,
        mostrar_imagenes=False,
        imagenes_para_mostrar=[],
        historial_conversacional="",
    )

    config = {
        "configurable": {"thread_id": "eval_ragas"},
        "run_name": "eval-ragas",
        "tags": ["evaluacion", "ragas"],
    }

    final_state = await asistente.compiled_graph.ainvoke(initial_state, config=config)

    # Extraer contextos individuales de los resultados válidos
    contextos = []
    for r in final_state.get("resultados_validos", []):
        texto = r.get("texto", "").strip()
        if texto:
            contextos.append(texto)

    # Si no hay resultados válidos pero hay contexto_documentos, usarlo
    if not contextos and final_state.get("contexto_documentos"):
        contextos = [final_state["contexto_documentos"]]

    # Extraer fuente/pagina recuperadas para Recall@K
    paginas_recuperadas = []
    referencias_recuperadas = []
    for r in final_state.get("resultados_validos", []):
        pag = r.get("pagina")
        if pag is not None:
            paginas_recuperadas.append(pag)
        referencias_recuperadas.append({
            "fuente": r.get("fuente", ""),
            "pagina": pag,
        })

    return {
        "respuesta": final_state.get("respuesta_final", ""),
        "contextos": contextos,
        "paginas_recuperadas": paginas_recuperadas,
        "referencias_recuperadas": referencias_recuperadas,
        "tema_valido": final_state.get("tema_valido", True),
        "contexto_suficiente": final_state.get("contexto_suficiente", False),
        "n_resultados_busqueda": len(final_state.get("resultados_busqueda", [])),
        "n_resultados_validos": len(final_state.get("resultados_validos", [])),
        "similitud_dominio": final_state.get("similitud_semantica_dominio", 0.0),
        "trayectoria": final_state.get("trayectoria", []),
    }


def _normalizar_fuente(fuente: str) -> str:
    return os.path.basename(str(fuente or "")).strip().lower()


def calcular_recall_at_k(resultados: list, paginas_esperadas: list, k: int = 5) -> float:
    """Recall@K: ¿las páginas esperadas están entre los top-K resultados?"""
    if not paginas_esperadas:
        return 0.0
    top_paginas = set()
    for r in resultados[:k]:
        pag = r.get("pagina")
        if pag is not None:
            top_paginas.add(pag)
    encontradas = len(set(paginas_esperadas) & top_paginas)
    return encontradas / len(paginas_esperadas)


def calcular_recall_fuente_pagina_at_k(
    referencias: list,
    fuente_esperada: str,
    paginas_esperadas: list,
    k: int = 5,
) -> float:
    """Recall@K estricto: fuente PDF y página deben coincidir."""
    if not fuente_esperada or not paginas_esperadas:
        return 0.0
    fuente_norm = _normalizar_fuente(fuente_esperada)
    esperadas = {(fuente_norm, pag) for pag in paginas_esperadas}
    recuperadas = set()
    for ref in referencias[:k]:
        pag = ref.get("pagina")
        fuente = _normalizar_fuente(ref.get("fuente", ""))
        if fuente and pag is not None:
            recuperadas.add((fuente, pag))
    return len(esperadas & recuperadas) / len(esperadas)


def medir_ruido_fuente(referencias: list, fuente_esperada: str, k: int = 5) -> dict:
    top_refs = referencias[:k]
    fuentes = [_normalizar_fuente(r.get("fuente", "")) for r in top_refs if r.get("fuente")]
    conteo = {}
    for fuente in fuentes:
        conteo[fuente] = conteo.get(fuente, 0) + 1

    fuente_dominante = max(conteo, key=conteo.get) if conteo else ""
    fuente_esperada_norm = _normalizar_fuente(fuente_esperada)
    fuera_esperada = sum(1 for fuente in fuentes if fuente_esperada_norm and fuente != fuente_esperada_norm)

    return {
        "fuentes_top_k": fuentes,
        "fuente_dominante": fuente_dominante,
        "fuente_dominante_pct": round(conteo.get(fuente_dominante, 0) / len(fuentes), 4) if fuentes else 0.0,
        "n_fuentes_distintas": len(conteo),
        "fuera_fuente_esperada_at_k": fuera_esperada,
    }


def parsear_indices(indices: str) -> list[int]:
    """Parsea indices humanos 1-based: '1,5,9' -> [0,4,8]."""
    if not indices:
        return []
    seleccionados = []
    for parte in indices.split(","):
        parte = parte.strip()
        if not parte:
            continue
        idx = int(parte)
        if idx <= 0:
            raise ValueError("Los indices deben ser 1-based y mayores que cero")
        seleccionados.append(idx - 1)
    return seleccionados


def seleccionar_por_indices(items: list, indices: list[int]) -> list:
    if not indices:
        return items
    seleccionados = []
    for idx in indices:
        if idx >= len(items):
            raise IndexError(f"Indice {idx + 1} fuera de rango; hay {len(items)} entradas")
        seleccionados.append(items[idx])
    return seleccionados


# ═══════════════════════════════════════════════════════════════════════
# EVALUACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

async def evaluar(limit: int = 0, skip_ragas: bool = False, indices: list[int] | None = None):
    asistente = await inicializar_asistente()

    golden = seleccionar_por_indices(GOLDEN_SET, indices or [])
    golden = golden[:limit] if limit > 0 else golden
    print(f"\n{'=' * 60}")
    print(f"📊 EVALUACIÓN RAGAS — {len(golden)} preguntas")
    print(f"{'=' * 60}\n")

    # ── Paso 1: Ejecutar el pipeline RAG para cada pregunta ──────────
    resultados_eval = []
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for i, item in enumerate(golden, 1):
        print(f"\n{'─' * 50}")
        print(f"[{i}/{len(golden)}] {item['question']}")
        print(f"{'─' * 50}")

        try:
            resultado = await ejecutar_consulta_con_contexto(asistente, item["question"])

            paginas_esp = item.get("paginas_esperadas", [])
            fuente_esp = item.get("fuente_esperada", "")
            recall_5_pagina = calcular_recall_at_k(
                [{"pagina": p} for p in resultado["paginas_recuperadas"]],
                paginas_esp, k=5
            ) if paginas_esp else None
            recall_5_fuente_pagina = calcular_recall_fuente_pagina_at_k(
                resultado["referencias_recuperadas"], fuente_esp, paginas_esp, k=5
            ) if paginas_esp and fuente_esp else None
            ruido_fuente = medir_ruido_fuente(
                resultado["referencias_recuperadas"], fuente_esp, k=5
            )

            entry = {
                "pregunta": item["question"],
                "ground_truth": item["ground_truth"],
                "fuente_esperada": fuente_esp,
                "paginas_esperadas": paginas_esp,
                "paginas_recuperadas": resultado["paginas_recuperadas"][:10],
                "referencias_recuperadas": resultado["referencias_recuperadas"][:10],
                "fuentes_recuperadas_top_5": ruido_fuente["fuentes_top_k"],
                "fuente_dominante_top_5": ruido_fuente["fuente_dominante"],
                "fuente_dominante_pct_top_5": ruido_fuente["fuente_dominante_pct"],
                "n_fuentes_distintas_top_5": ruido_fuente["n_fuentes_distintas"],
                "fuera_fuente_esperada_at_5": ruido_fuente["fuera_fuente_esperada_at_k"],
                # recall_at_5 = recall@5 convencional (acertar la página).
                # recall_at_5_fuente_pagina es la variante estricta (fuente + página).
                "recall_at_5": recall_5_pagina,
                "recall_at_5_fuente_pagina": recall_5_fuente_pagina,
                "recall_at_5_pagina": recall_5_pagina,
                "respuesta": resultado["respuesta"],
                "contextos": resultado["contextos"],
                "tema_valido": resultado["tema_valido"],
                "contexto_suficiente": resultado["contexto_suficiente"],
                "n_resultados": resultado["n_resultados_busqueda"],
                "n_validos": resultado["n_resultados_validos"],
                "similitud_dominio": resultado["similitud_dominio"],
            }
            resultados_eval.append(entry)

            # Para RAGAS
            questions.append(item["question"])
            answers.append(resultado["respuesta"])
            contexts_list.append(resultado["contextos"] if resultado["contextos"] else ["Sin contexto"])
            ground_truths.append(item["ground_truth"])

            recall_str = f" | Recall@5 fuente+página: {recall_5_fuente_pagina:.2f}" if recall_5_fuente_pagina is not None else ""
            ruido_str = f" | Fuera fuente@5: {ruido_fuente['fuera_fuente_esperada_at_k']}"
            print(f"   ✅ Respuesta: {len(resultado['respuesta'])} chars | "
                  f"Contextos: {len(resultado['contextos'])} | "
                  f"Válidos: {resultado['n_resultados_validos']}{recall_str}{ruido_str}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            resultados_eval.append({
                "pregunta": item["question"],
                "ground_truth": item["ground_truth"],
                "error": str(e),
            })
            questions.append(item["question"])
            answers.append(f"Error: {e}")
            contexts_list.append(["Error en el pipeline"])
            ground_truths.append(item["ground_truth"])

    # ── Paso 2: Métricas de retrieval (sin LLM) ─────────────────────
    print(f"\n\n{'=' * 60}")
    print("📏 MÉTRICAS DE RETRIEVAL (sin LLM)")
    print(f"{'=' * 60}\n")

    n_eval = len(resultados_eval)
    if n_eval == 0:
        print("⚠️ No hay preguntas evaluadas; se omiten las métricas de retrieval.")
        return

    total_con_contexto = sum(1 for r in resultados_eval if r.get("contexto_suficiente"))
    total_tema_valido = sum(1 for r in resultados_eval if r.get("tema_valido"))
    avg_n_validos = sum(r.get("n_validos", 0) for r in resultados_eval) / n_eval
    avg_sim = sum(r.get("similitud_dominio", 0) for r in resultados_eval) / n_eval

    recall_vals = [r["recall_at_5_fuente_pagina"] for r in resultados_eval if r.get("recall_at_5_fuente_pagina") is not None]
    avg_recall_5 = sum(recall_vals) / len(recall_vals) if recall_vals else None
    recall_pagina_vals = [r["recall_at_5_pagina"] for r in resultados_eval if r.get("recall_at_5_pagina") is not None]
    avg_recall_5_pagina = sum(recall_pagina_vals) / len(recall_pagina_vals) if recall_pagina_vals else None
    ruido_vals = [r.get("fuera_fuente_esperada_at_5", 0) for r in resultados_eval if "fuera_fuente_esperada_at_5" in r]
    avg_fuera_fuente_at_5 = sum(ruido_vals) / len(ruido_vals) if ruido_vals else None
    fuente_dominante_ok = [
        _normalizar_fuente(r.get("fuente_dominante_top_5", "")) == _normalizar_fuente(r.get("fuente_esperada", ""))
        for r in resultados_eval
        if r.get("fuente_esperada") and r.get("fuente_dominante_top_5")
    ]
    fuente_dominante_ok_pct = (
        100 * sum(1 for ok in fuente_dominante_ok if ok) / len(fuente_dominante_ok)
        if fuente_dominante_ok else None
    )

    print(f"  Preguntas evaluadas:        {len(resultados_eval)}")
    print(f"  Tema válido (clasificador):  {total_tema_valido}/{len(resultados_eval)} "
          f"({100*total_tema_valido/len(resultados_eval):.0f}%)")
    print(f"  Con contexto suficiente:     {total_con_contexto}/{len(resultados_eval)} "
          f"({100*total_con_contexto/len(resultados_eval):.0f}%)")
    print(f"  Promedio resultados válidos: {avg_n_validos:.1f}")
    print(f"  Promedio similitud dominio:  {avg_sim:.4f}")
    if avg_recall_5 is not None:
        print(f"  Recall@5 fuente+página:      {avg_recall_5:.4f}  "
              f"({len(recall_vals)}/{len(resultados_eval)} preguntas con fuente/páginas target)")
    if avg_recall_5_pagina is not None:
        print(f"  Recall@5 solo página:        {avg_recall_5_pagina:.4f}  "
              f"({len(recall_pagina_vals)}/{len(resultados_eval)} preguntas con páginas target)")
    if avg_fuera_fuente_at_5 is not None:
        print(f"  Fuera fuente esperada@5:     {avg_fuera_fuente_at_5:.2f} resultados promedio")
    if fuente_dominante_ok_pct is not None:
        print(f"  Fuente dominante correcta:   {fuente_dominante_ok_pct:.1f}%")

    # ── Paso 2.5: Guardar reporte intermedio ─────────────────────────
    reporte = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_preguntas": len(resultados_eval),
        "metricas_retrieval": {
            "tema_valido_pct": round(100 * total_tema_valido / len(resultados_eval), 1) if resultados_eval else 0,
            "contexto_suficiente_pct": round(100 * total_con_contexto / len(resultados_eval), 1) if resultados_eval else 0,
            "avg_resultados_validos": round(avg_n_validos, 2),
            "avg_similitud_dominio": round(avg_sim, 4),
            "recall_at_5": round(avg_recall_5_pagina, 4) if avg_recall_5_pagina is not None else None,
            "recall_at_5_fuente_pagina": round(avg_recall_5, 4) if avg_recall_5 is not None else None,
            "recall_at_5_pagina": round(avg_recall_5_pagina, 4) if avg_recall_5_pagina is not None else None,
            "avg_fuera_fuente_esperada_at_5": round(avg_fuera_fuente_at_5, 2) if avg_fuera_fuente_at_5 is not None else None,
            "fuente_dominante_correcta_pct": round(fuente_dominante_ok_pct, 1) if fuente_dominante_ok_pct is not None else None,
        },
        "metricas_ragas": {},
        "detalle": resultados_eval,
    }

    reporte_path = Path(__file__).parent / "reporte_ragas.json"
    with open(reporte_path, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Reporte intermedio guardado en: {reporte_path} (por si falla RAGAS)")

    # ── Paso 3: Evaluación RAGAS ─────────────────────────────────────
    metricas_ragas = {}
    if not skip_ragas:
        print(f"\n\n{'=' * 60}")
        print("🧪 EVALUACIÓN RAGAS (consume cuota)")
        print(f"{'=' * 60}\n")

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics._faithfulness import Faithfulness
            from ragas.metrics._answer_relevance import AnswerRelevancy
            from ragas.metrics._context_precision import ContextPrecision
            from ragas.metrics._context_recall import ContextRecall
            from ragas.run_config import RunConfig
            from langchain_openai import ChatOpenAI
            from langchain_huggingface import HuggingFaceEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            # Juez LLM. Preferir gpt-4o-mini si hay OPENAI_API_KEY: es mas estable
            # con RAGAS moderno. Fallback a NVIDIA si no se configuro OpenAI.
            if os.getenv("OPENAI_API_KEY"):
                juez_modelo = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini")
                llm_eval = LangchainLLMWrapper(ChatOpenAI(
                    model=juez_modelo,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=0,
                ))
                print(f"   Juez RAGAS: OpenAI {juez_modelo}")
            else:
                juez_modelo = "meta/llama-3.3-70b-instruct"
                llm_eval = LangchainLLMWrapper(ChatOpenAI(
                    model=juez_modelo,
                    api_key=os.getenv("NVIDIA_API_KEY"),
                    base_url="https://integrate.api.nvidia.com/v1",
                    temperature=0,
                ))
                print(f"   Juez RAGAS: NVIDIA {juez_modelo}")

            # Embeddings locales y gratis (MiniLM)
            emb_eval = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            ))

            # Construir dataset RAGAS v0.4 (columnas renombradas)
            ragas_data = {
                "user_input": questions,
                "response": answers,
                "retrieved_contexts": contexts_list,
                "reference": ground_truths,
            }
            dataset = Dataset.from_dict(ragas_data)

            print(f"   Dataset: {len(dataset)} entradas")
            print("   Ejecutando métricas RAGAS (esto puede tardar)...\n")

            metrics = [
                Faithfulness(llm=llm_eval),
                AnswerRelevancy(llm=llm_eval, embeddings=emb_eval),
                ContextPrecision(llm=llm_eval),
                ContextRecall(llm=llm_eval),
            ]

            max_workers_ragas = int(os.getenv("RAGAS_MAX_WORKERS", "2"))
            run_config = RunConfig(
                timeout=180,
                max_retries=15,
                max_workers=max_workers_ragas,
                max_wait=120
            )

            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                run_config=run_config,
            )

            # Extraer métricas del EvaluationResult
            df = result.to_pandas()
            metric_cols = [c for c in df.columns if c not in
                          ("user_input", "response", "retrieved_contexts", "reference")]
            for col in metric_cols:
                vals = df[col].dropna()
                if len(vals) > 0:
                    metricas_ragas[col] = round(vals.mean(), 4)

            print(f"\n{'─' * 40}")
            print("📊 RESULTADOS RAGAS:")
            print(f"{'─' * 40}")
            for metrica, valor in metricas_ragas.items():
                barra = "█" * int(valor * 20) + "░" * (20 - int(valor * 20))
                print(f"  {metrica:25s} {barra} {valor:.4f}")

            # Detalle por pregunta
            print(f"\n📋 Detalle por pregunta:")
            header_metrics = [m[:6] for m in metric_cols[:4]]
            print(f"  {'Pregunta':48s} | {' | '.join(f'{h:>6}' for h in header_metrics)}")
            print("  " + "-" * 85)
            for _, row in df.iterrows():
                q = str(row.get("user_input", ""))[:48]
                vals = [f"{row.get(m, float('nan')):6.3f}" for m in metric_cols[:4]]
                print(f"  {q:48s} | {' | '.join(vals)}")

        except ImportError as e:
            print(f"❌ No se pudieron importar las dependencias RAGAS: {e}")
            print("   Instalá con: pip install ragas datasets")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Error durante evaluación RAGAS: {e}")
    else:
        print("\n⏭️  Evaluación RAGAS omitida (--no-ragas)")

    # ── Paso 4: Guardar reporte final ────────────────────────────────
    reporte["metricas_ragas"] = metricas_ragas
    with open(reporte_path, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Reporte final guardado con RAGAS en: {reporte_path}")

    # ── Resumen final ────────────────────────────────────────────────
    print(f"\n\n{'═' * 60}")
    print("📊 RESUMEN FINAL")
    print(f"{'═' * 60}")
    print(f"  Preguntas evaluadas:       {len(resultados_eval)}")
    print(f"  Clasificación correcta:    {total_tema_valido}/{len(resultados_eval)}")
    print(f"  Con contexto suficiente:   {total_con_contexto}/{len(resultados_eval)}")
    if metricas_ragas:
        print(f"  ── Métricas RAGAS ──")
        for k, v in metricas_ragas.items():
            print(f"    {k:25s}: {v:.4f}")
    print(f"{'═' * 60}\n")

    await asistente.cerrar()


# ═══════════════════════════════════════════════════════════════════════
# EVALUACIÓN IMAGEN → IMAGEN (CLIP vs UNI)
# ═══════════════════════════════════════════════════════════════════════

async def evaluar_imagenes():
    """
    Diagnóstico imagen→imagen: compara CLIP vs UNI.
    Convierte cada página del PDF a imagen, embede con ambos modelos,
    y mide Recall@K para verificar retrieval visual.
    """
    import glob
    import torch
    import numpy as np
    from PIL import Image
    from pdf2image import convert_from_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 60}")
    print(f"🖼️  EVALUACIÓN IMAGEN → IMAGEN (CLIP vs UNI)")
    print(f"   Device: {device}")
    print(f"{'=' * 60}\n")

    # ── Buscar PDF ──────────────────────────────────────────────────
    pdfs = glob.glob(f"{DIRECTORIO_PDFS}/*.pdf")
    if not pdfs:
        print(f"❌ No hay PDFs en {DIRECTORIO_PDFS}")
        return {}
    pdf_path = pdfs[0]
    print(f"📄 PDF: {pdf_path}")

    # ── Convertir páginas a imágenes ─────────────────────────────────
    print("📸 Convirtiendo páginas a imágenes...")
    paginas_img = convert_from_path(pdf_path, dpi=150)
    n_paginas = len(paginas_img)
    print(f"   → {n_paginas} páginas")

    # ── Cargar modelos UNO A LA VEZ (4GB VRAM) ────────────────────────
    modelo_configs = []

    modelo_configs.append({
        "nombre": "CLIP",
        "loader": lambda: _load_clip(device),
    })

    modelo_configs.append({
        "nombre": "UNI",
        "loader": lambda: _load_uni(device),
    })

    all_embeddings = {}

    for cfg in modelo_configs:
        nombre = cfg["nombre"]
        print(f"\n{'─' * 40}")
        print(f"🔄 Cargando {nombre}...")

        try:
            embed_fn, cleanup_refs = cfg["loader"]()
            print(f"✅ {nombre} cargado")

            print(f"🔢 Generando embeddings {nombre} ({n_paginas} páginas)...")
            embeddings = []
            for i, pag_img in enumerate(paginas_img):
                emb = embed_fn(pag_img)
                embeddings.append(emb)
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"   Página {i+1}/{n_paginas}")

            all_embeddings[nombre] = np.array(embeddings)

            print(f"🗑  Liberando {nombre} de VRAM...")
            for ref in cleanup_refs:
                del ref
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc; gc.collect()

        except Exception as e:
            print(f"❌ Error con {nombre}: {e}")

    if not all_embeddings:
        print("❌ No se pudo cargar ningún modelo. Abortando.")
        return {}

    resultados_modelos = {}

    for nombre, embeddings in all_embeddings.items():
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_norm = embeddings / norms

        sim_matrix = embeddings_norm @ embeddings_norm.T

        top_scores = {1: [], 3: [], 5: []}
        for i in range(n_paginas):
            sims = sim_matrix[i].copy()
            sims[i] = -1
            sorted_sims = np.sort(sims)[::-1]
            top_scores[1].append(sorted_sims[0])
            top_scores[3].append(np.mean(sorted_sims[:3]))
            top_scores[5].append(np.mean(sorted_sims[:5]))

        avg_top1 = np.mean(top_scores[1])
        avg_top3 = np.mean(top_scores[3])
        avg_top5 = np.mean(top_scores[5])

        vecindad_hits = 0
        for i in range(n_paginas):
            sims = sim_matrix[i].copy()
            sims[i] = -1
            best_match = np.argmax(sims)
            if abs(best_match - i) <= 2:
                vecindad_hits += 1
        vecindad_pct = vecindad_hits / n_paginas

        resultados_modelos[nombre] = {
            "avg_sim_top1": round(float(avg_top1), 4),
            "avg_sim_top3": round(float(avg_top3), 4),
            "avg_sim_top5": round(float(avg_top5), 4),
            "vecindad_pct": round(float(vecindad_pct) * 100, 1),
            "n_paginas": n_paginas,
        }

        print(f"\n   📊 Resultados {nombre}:")
        print(f"      Sim promedio Top-1: {avg_top1:.4f}")
        print(f"      Sim promedio Top-3: {avg_top3:.4f}")
        print(f"      Sim promedio Top-5: {avg_top5:.4f}")
        print(f"      Vecindad (±2 págs): {vecindad_pct*100:.1f}%")

    if len(resultados_modelos) > 1:
        print(f"\n\n{'═' * 60}")
        print("📊 COMPARACIÓN IMAGEN→IMAGEN")
        print(f"{'═' * 60}")
        print(f"  {'Métrica':25s}", end="")
        for nombre in resultados_modelos:
            print(f" | {nombre:>10s}", end="")
        print()
        print("  " + "─" * (27 + 13 * len(resultados_modelos)))

        for metrica in ["avg_sim_top1", "avg_sim_top3", "avg_sim_top5", "vecindad_pct"]:
            label = metrica.replace("avg_sim_", "Sim Top-").replace("vecindad_pct", "Vecindad ±2")
            print(f"  {label:25s}", end="")
            for nombre in resultados_modelos:
                val = resultados_modelos[nombre][metrica]
                fmt = f"{val:>9.1f}%" if "pct" in metrica else f"{val:>10.4f}"
                print(f" | {fmt}", end="")
            print()
        print(f"{'═' * 60}\n")

    return resultados_modelos


def _load_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return (lambda img, m=model, p=processor: _embed_clip(img, m, p, device), [model, processor])

def _load_uni(device):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    model = timm.create_model("hf_hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    model.to(device).eval()
    config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**config)
    return (lambda img, m=model, t=transform: _embed_uni(img, m, t, device), [model, transform])

def _embed_uni(pil_image, model, transform, device):
    import torch
    import numpy as np
    try:
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.inference_mode():
            emb = model(image_tensor)
        return emb.cpu().numpy().flatten()
    except Exception as e:
        print(f"⚠️ Error embed UNI: {e}")
        return np.zeros(1024)

def _embed_clip(pil_image, model, processor, device):
    import torch
    import numpy as np
    try:
        inputs = processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            vision_out = model.vision_model(pixel_values=pixel_values)
            pooled = vision_out.pooler_output
            image_features = model.visual_projection(pooled)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"⚠️ Error embed CLIP: {e}")
        return np.zeros(512)


# ═══════════════════════════════════════════════════════════════════════
# MODO --solo-ragas: carga JSON existente y solo corre el juez
# ═══════════════════════════════════════════════════════════════════════

async def evaluar_solo_ragas(limit: int = 0, indices: list[int] | None = None):
    """Carga reporte_ragas.json existente y corre solo el juez RAGAS."""
    reporte_path = Path(__file__).parent / "reporte_ragas.json"
    if not reporte_path.exists():
        print("❌ No existe reporte_ragas.json. Corre primero con --no-ragas.")
        return

    with open(reporte_path, "r", encoding="utf-8") as f:
        reporte = json.load(f)

    detalle = reporte.get("detalle", [])
    if indices:
        if max(indices) < len(detalle):
            detalle = seleccionar_por_indices(detalle, indices)
        else:
            # Algún índice excede el reporte: lo más probable es que el reporte
            # ya se haya generado filtrado por estos mismos --indices, así que lo
            # usamos tal cual en lugar de reventar con IndexError.
            print(f"ℹ️  Algunos índices exceden el reporte ({len(detalle)} entradas); "
                  "se asume que ya está filtrado y se usa tal cual.")
    if limit > 0:
        detalle = detalle[:limit]
    if not detalle:
        print("❌ El JSON no tiene entradas en 'detalle'. Corre primero con --no-ragas.")
        return

    print(f"\n✅ Cargando {len(detalle)} preguntas desde {reporte_path}")
    print(f"   Timestamp del reporte: {reporte.get('timestamp', 'desconocido')}")

    # Reconstruir listas para RAGAS
    questions, answers, contexts_list, ground_truths = [], [], [], []
    for entry in detalle:
        if "error" in entry:
            continue
        questions.append(entry.get("pregunta", ""))
        answers.append(entry.get("respuesta", ""))
        ctx = entry.get("contextos", [])
        contexts_list.append(ctx if ctx else ["Sin contexto"])
        ground_truths.append(entry.get("ground_truth", ""))

    print(f"   Preguntas válidas para RAGAS: {len(questions)}")

    print(f"\n\n{'=' * 60}")
    print("🧪 EVALUACIÓN RAGAS")
    print(f"{'=' * 60}\n")

    metricas_ragas = {}
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._context_recall import ContextRecall
        from ragas.run_config import RunConfig
        from langchain_openai import ChatOpenAI
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        if os.getenv("OPENAI_API_KEY"):
            juez_modelo = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini")
            llm_eval = LangchainLLMWrapper(ChatOpenAI(
                model=juez_modelo,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0,
            ))
            print(f"   Juez RAGAS: OpenAI {juez_modelo}")
        else:
            juez_modelo = "meta/llama-3.3-70b-instruct"
            llm_eval = LangchainLLMWrapper(ChatOpenAI(
                model=juez_modelo,
                api_key=os.getenv("NVIDIA_API_KEY"),
                base_url="https://integrate.api.nvidia.com/v1",
                temperature=0,
            ))
            print(f"   Juez RAGAS: NVIDIA {juez_modelo}")

        emb_eval = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ))

        dataset = Dataset.from_dict({
            "user_input": questions, "response": answers,
            "retrieved_contexts": contexts_list, "reference": ground_truths,
        })
        print(f"   Dataset: {len(dataset)} entradas")
        print("   Ejecutando métricas RAGAS...\n")

        max_workers_ragas = int(os.getenv("RAGAS_MAX_WORKERS", "2"))
        run_config = RunConfig(
            timeout=600,
            max_retries=10,
            max_workers=max_workers_ragas,
            max_wait=120,
        )
        result = evaluate(dataset=dataset, metrics=[
            Faithfulness(llm=llm_eval),
            AnswerRelevancy(llm=llm_eval, embeddings=emb_eval),
            ContextPrecision(llm=llm_eval),
            ContextRecall(llm=llm_eval),
        ], run_config=run_config)

        df = result.to_pandas()
        metric_cols = [c for c in df.columns if c not in
                       ("user_input", "response", "retrieved_contexts", "reference")]
        for col in metric_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                metricas_ragas[col] = round(float(vals.mean()), 4)

        print(f"\n{'─' * 40}")
        print("📊 RESULTADOS RAGAS:")
        print(f"{'─' * 40}")
        for metrica, valor in metricas_ragas.items():
            barra = "█" * int(valor * 20) + "░" * (20 - int(valor * 20))
            print(f"  {metrica:25s} {barra} {valor:.4f}")

        print(f"\n📋 Detalle por pregunta:")
        header_metrics = [m[:6] for m in metric_cols[:4]]
        print(f"  {'Pregunta':48s} | {' | '.join(f'{h:>6}' for h in header_metrics)}")
        print("  " + "-" * 85)
        for _, row in df.iterrows():
            q = str(row.get("user_input", ""))[:48]
            vals = [f"{row.get(m, float('nan')):6.3f}" for m in metric_cols[:4]]
            print(f"  {q:48s} | {' | '.join(vals)}")

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"❌ Error durante evaluación RAGAS: {e}")

    reporte["metricas_ragas"] = metricas_ragas
    reporte["timestamp_ragas"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(reporte_path, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Métricas RAGAS guardadas en: {reporte_path}")


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluación RAGAS del RAG Histología v5.0")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limitar a N preguntas (0 = todas)")
    parser.add_argument("--no-ragas", action="store_true",
                        help="Solo retrieval, sin métricas RAGAS (no consume cuota LLM)")
    parser.add_argument("--solo-ragas", action="store_true",
                        help="Solo juez RAGAS sobre JSON existente (sin retrieval)")
    parser.add_argument("--indices", type=str, default="",
                        help="Indices 1-based separados por coma (ej: 1,5,9)")
    parser.add_argument("--imagenes", action="store_true",
                        help="Ejecutar diagnóstico imagen→imagen (CLIP vs UNI)")
    args = parser.parse_args()

    async def main():
        reporte_imagenes = {}
        indices = parsear_indices(args.indices)

        if args.solo_ragas:
            await evaluar_solo_ragas(limit=args.limit, indices=indices)
            return

        if args.imagenes:
            reporte_imagenes = await evaluar_imagenes()

        # Evaluación de texto/RAGAS: siempre, salvo cuando se pidió SOLO
        # --imagenes (sin --limit), en cuyo caso corre únicamente el
        # diagnóstico imagen→imagen.
        if not args.imagenes or args.limit > 0:
            await evaluar(limit=args.limit, skip_ragas=args.no_ragas, indices=indices)

        if reporte_imagenes:
            reporte_path = Path(__file__).parent / "reporte_ragas.json"
            try:
                with open(reporte_path, "r", encoding="utf-8") as f:
                    reporte = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                reporte = {}
            reporte["evaluacion_imagenes"] = reporte_imagenes
            with open(reporte_path, "w", encoding="utf-8") as f:
                json.dump(reporte, f, indent=2, ensure_ascii=False)
            print(f"💾 Resultados imagen→imagen guardados en: {reporte_path}")

    asyncio.run(main())
