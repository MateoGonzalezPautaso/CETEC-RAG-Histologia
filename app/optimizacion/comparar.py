"""
Compara dos corridas del juez (p. ej. prompt actual vs. prompt optimizado por GEPA).

Empareja los juicios por índice del golden set y muestra, por dimensión y en
el puntaje global, el promedio de cada corrida y la diferencia.

Para no medir sobre las mismas preguntas con las que se optimizó, --solo-val
toma los índices de validación del optimized_prompts.json de GEPA.

Uso (desde app/):
    uv run python -m optimizacion.comparar A-juez.jsonl B-juez.jsonl
    uv run python -m optimizacion.comparar A-juez.jsonl B-juez.jsonl \
        --solo-val optimizacion/resultados/gepa-X/optimized_prompts.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

DIMENSIONES = ["correccion", "fidelidad", "notacion", "estilo_docente"]


def _cargar(ruta: Path) -> Dict[int, dict]:
    juicios = {}
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            try:
                j = json.loads(linea)
                juicios[j["indice"]] = j
            except (json.JSONDecodeError, KeyError):
                continue
    return juicios


def _prom(xs: List[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 4) if xs else None


def comparar(a: Dict[int, dict], b: Dict[int, dict], indices: Optional[Set[int]] = None) -> dict:
    comunes = sorted(set(a) & set(b) & (indices if indices is not None else set(a) | set(b)))
    filas = {"puntaje_global": ([a[i]["puntaje_global"] for i in comunes], [b[i]["puntaje_global"] for i in comunes])}
    for d in DIMENSIONES:
        filas[d] = ([a[i]["veredicto"][d]["puntaje"] for i in comunes], [b[i]["veredicto"][d]["puntaje"] for i in comunes])

    resumen = {"n": len(comunes), "indices": comunes, "metricas": {}}
    for nombre, (xa, xb) in filas.items():
        pa, pb = _prom(xa), _prom(xb)
        resumen["metricas"][nombre] = {"a": pa, "b": pb, "delta": None if pa is None else round(pb - pa, 4)}
    ga, gb = filas["puntaje_global"]
    resumen["b_gana"] = sum(1 for x, y in zip(ga, gb) if y > x)
    resumen["b_pierde"] = sum(1 for x, y in zip(ga, gb) if y < x)
    resumen["empates"] = len(comunes) - resumen["b_gana"] - resumen["b_pierde"]
    return resumen


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara dos corridas del juez")
    parser.add_argument("a", type=Path, help="Juicios de referencia (p. ej. prompt actual)")
    parser.add_argument("b", type=Path, help="Juicios a comparar (p. ej. prompt optimizado)")
    parser.add_argument("--solo-val", type=Path, default=None,
                        help="optimized_prompts.json de GEPA: restringe a sus índices de validación")
    args = parser.parse_args()

    indices = None
    if args.solo_val:
        meta = json.loads(args.solo_val.read_text(encoding="utf-8")).get("metadata", {})
        indices = set(meta.get("indices_val", []))
        print(f"🎯 Solo índices de validación: {sorted(indices)}")

    resumen = comparar(_cargar(args.a), _cargar(args.b), indices)
    if not resumen["n"]:
        print("❌ No hay índices en común entre las dos corridas.")
        return 1

    print(f"\nA = {args.a.name}\nB = {args.b.name}\nPreguntas comparadas: {resumen['n']}\n")
    print(f"{'métrica':<16}{'A':>8}{'B':>8}{'Δ (B−A)':>10}")
    for nombre, m in resumen["metricas"].items():
        print(f"{nombre:<16}{m['a']:>8.3f}{m['b']:>8.3f}{m['delta']:>+10.3f}")
    print(f"\nB mejor en {resumen['b_gana']} | peor en {resumen['b_pierde']} | igual en {resumen['empates']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
