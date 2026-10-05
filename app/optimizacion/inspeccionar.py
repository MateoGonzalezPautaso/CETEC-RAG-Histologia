"""
Muestra qué le llegó al LLM en preguntas puntuales de una corrida del baseline.

Sirve para separar fallas de recuperación (el dato no estaba en las secciones
del manual) de fallas de generación (estaba y el modelo no lo usó). No llama a
ningún modelo: solo lee el JSONL del baseline y, si existe, el del juez.

Por pregunta imprime los resultados válidos (fuente, página, similitud y el
principio del texto, con ✔ en las páginas esperadas), qué fracción de las
palabras de la referencia aparece en el contexto, la respuesta y la
justificación del juez en corrección.

Uso (desde app/):
    uv run python -m optimizacion.inspeccionar optimizacion/resultados/baseline-v4.jsonl 1 5 15
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from src.config import normalizar  # noqa: E402

_STOP = {
    "como", "cual", "cuales", "tiene", "tienen", "para", "sobre", "entre", "donde",
    "esta", "este", "estas", "estos", "cada", "una", "unos", "unas", "los", "las",
    "del", "con", "por", "sus", "son", "mas", "que", "se", "el", "la", "de", "en", "y",
}


def _palabras(texto: str) -> set:
    return {
        p for p in (w.strip(".,;:()[]\"'¿?¡!") for w in normalizar(texto or "").split())
        if len(p) > 3 and p not in _STOP
    }


def _cargar(ruta: Path) -> dict:
    registros = {}
    if not ruta.exists():
        return registros
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            try:
                r = json.loads(linea)
                registros[r["indice"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return registros


def mostrar(reg: dict, juicio: Optional[dict]) -> None:
    rec = reg["traza"]["recuperacion"]
    esperadas = {(reg.get("fuente_esperada"), p) for p in reg.get("paginas_esperadas", [])}
    print("=" * 78)
    print(f"#{reg['indice']} {reg['question']}")
    print(f"   búsqueda: {reg['traza']['consulta'].get('busqueda_texto')!r}")
    print(f"   esperado: {reg.get('fuente_esperada')} págs {reg.get('paginas_esperadas')} | "
          f"recall@5 = {reg.get('recall_fuente_pagina_at_5')}")
    for i, r in enumerate(rec.get("resultados_validos") or [], 1):
        marca = "✔" if (r.get("fuente"), r.get("pagina")) in esperadas else " "
        texto = " ".join((r.get("texto") or "").split())
        partes = "" if r.get("bm25") is None else f" (coseno={r['sim_vector']:.2f} bm25={r['bm25']:.2f})"
        print(f"   {marca} {i}. {r.get('fuente')} p{r.get('pagina')} {r.get('tipo')} "
              f"sim={r.get('similitud', 0):.2f}{partes} | {texto[:110]}")
    ref = _palabras(reg.get("ground_truth", ""))
    ctx = _palabras(rec.get("contexto_documentos") or "")
    if ref:
        faltan = sorted(ref - ctx)
        print(f"   referencia en el contexto: {len(ref & ctx)}/{len(ref)} palabras"
              + (f" | faltan: {', '.join(faltan[:15])}" if faltan else ""))
    respuesta = " ".join((reg["traza"]["respuesta"].get("texto") or "").split())
    print(f"   respuesta: {respuesta[:300]}{'…' if len(respuesta) > 300 else ''}")
    if juicio:
        v = juicio["veredicto"]["correccion"]
        print(f"   juez: global={juicio['puntaje_global']:.2f} | corrección {v['puntaje']}: {v['justificacion']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Muestra el contexto recuperado de preguntas del baseline")
    parser.add_argument("baseline", type=Path, help="JSONL del baseline (p. ej. baseline-v4.jsonl)")
    parser.add_argument("indices", type=int, nargs="+", help="Índices del golden set")
    args = parser.parse_args()

    registros = _cargar(args.baseline)
    juicios = _cargar(args.baseline.with_name(args.baseline.stem + "-juez.jsonl"))
    for i in args.indices:
        if i not in registros:
            print(f"⚠️ #{i} no está en {args.baseline.name}")
            continue
        mostrar(registros[i], juicios.get(i))
    return 0


if __name__ == "__main__":
    sys.exit(main())
