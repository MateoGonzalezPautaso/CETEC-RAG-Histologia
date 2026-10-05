"""
Recall de la búsqueda de texto con distintos pesos coseno/BM25, sin llamar a ningún LLM.

Para cada pregunta del golden set calcula el embedding de la pregunta (MiniLM,
como el pipeline) y ordena los chunks con `busqueda_densa_lexica` para cada
peso del coseno: 1.0 es solo embeddings, 0.0 es solo BM25. Mide recall
fuente+página@5 sobre ese orden.

Es la búsqueda sola: no aplica el filtro de fuente dominante ni suma el texto
de las imágenes, así que no coincide exactamente con el recall del baseline.
Sirve para elegir HIBRIDA_PESO_VECTOR antes de gastar cuota en una corrida.

Con --baseline usa, además de la pregunta, la consulta reescrita que guardó esa
corrida (como hace el pipeline); sin él, BM25 busca solo con la pregunta.

Uso (desde app/, con el servidor DETENIDO y los chunks ya indexados):
    uv run python -m optimizacion.recuperacion --baseline optimizacion/resultados/baseline-v4.jsonl
    uv run python -m optimizacion.recuperacion --pesos 0.3 0.5 0.7
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from evaluar_ragas import GOLDEN_SET, calcular_recall_fuente_pagina_at_k  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from src.config import HIBRIDA_PESO_VECTOR, QDRANT_PATH  # noqa: E402
from src.qdrant_store import QdrantVectorStore  # noqa: E402

PESOS = [1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0]


def _consultas_reescritas(ruta: Path) -> dict:
    salida = {}
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            try:
                r = json.loads(linea)
                salida[r["indice"]] = r["traza"]["consulta"].get("busqueda_texto") or ""
            except (json.JSONDecodeError, KeyError):
                continue
    return salida


async def main_async(args) -> int:
    reescritas = _consultas_reescritas(args.baseline) if args.baseline else {}
    store = QdrantVectorStore(path=QDRANT_PATH)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pesos = args.pesos or PESOS

    recall = {p: [] for p in pesos}
    for i, item in enumerate(GOLDEN_SET):
        emb = embeddings.embed_query(item["question"])
        consulta = f"{item['question']} {reescritas.get(i, '')}"
        for p in pesos:
            res = await store.busqueda_densa_lexica(emb, consulta, top_k=5, peso_vector=p)
            recall[p].append(calcular_recall_fuente_pagina_at_k(
                res, item.get("fuente_esperada", ""), item.get("paginas_esperadas", []), k=5,
            ))
    store.client.close()

    print(f"\nRecall fuente+página@5 de la búsqueda de texto ({len(GOLDEN_SET)} preguntas; "
          f"peso actual = {HIBRIDA_PESO_VECTOR})\n")
    fuentes = sorted({it.get("fuente_esperada", "") for it in GOLDEN_SET})
    print(f"{'peso coseno':<12}{'total':>7}" + "".join(f"{f:>11}" for f in fuentes))
    for p in pesos:
        fila = f"{p:<12.2f}{sum(recall[p]) / len(recall[p]):>7.3f}"
        for f in fuentes:
            xs = [r for r, it in zip(recall[p], GOLDEN_SET) if it.get("fuente_esperada") == f]
            fila += f"{sum(xs) / len(xs):>11.3f}"
        print(fila)

    print("\nPreguntas en las que cambia el recall según el peso:")
    print("  #  " + "".join(f"{p:>6.1f}" for p in pesos) + "  pregunta")
    for i, item in enumerate(GOLDEN_SET):
        valores = [recall[p][i] for p in pesos]
        if max(valores) != min(valores):
            print(f"{i:3d}  " + "".join(f"{v:>6.2f}" for v in valores) + f"  {item['question'][:50]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Recall de la búsqueda de texto según el peso coseno/BM25")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="JSONL de un baseline: suma la consulta reescrita de cada pregunta a BM25")
    parser.add_argument("--pesos", type=float, nargs="+", default=None,
                        help=f"Pesos del coseno a probar (por defecto {PESOS})")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
