"""
Baseline del pipeline sobre el golden set (Etapa 2, semana 12).

Corre cada pregunta de GOLDEN_SET por `AsistenteHistologiaQdrant.consultar()`
—el mismo camino que usa /api/chat— y guarda, por pregunta, la traza completa
junto con la respuesta de referencia y el recall de fuente+página.

Cada pregunta usa una sesión propia: sin historial de las anteriores, la
reescritura de la consulta no se contamina con otras preguntas del set.

Si el modelo se queda sin cuota, la corrida se corta sin registrar la pregunta
fallida; volver a correr con el mismo --salida retoma donde quedó.

Uso (desde app/, con el servidor DETENIDO: Qdrant local no admite dos procesos):
    uv run python -m optimizacion.baseline                  # golden set completo
    uv run python -m optimizacion.baseline --limit 3        # prueba rápida
    uv run python -m optimizacion.baseline --salida optimizacion/resultados/baseline-X.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from evaluar_ragas import (  # noqa: E402  (carga .env e importa el pipeline)
    GOLDEN_SET, calcular_recall_fuente_pagina_at_k, inicializar_asistente, medir_ruido_fuente,
)
from src.llm import _quota_blocked  # noqa: E402

RESULTADOS_DIR = APP_DIR / "optimizacion" / "resultados"


def _cargar_hechos(ruta: Path) -> Dict[int, dict]:
    hechos: Dict[int, dict] = {}
    if not ruta.exists():
        return hechos
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            try:
                reg = json.loads(linea)
                hechos[reg["indice"]] = reg
            except (json.JSONDecodeError, KeyError):
                continue
    return hechos


def construir_registro(indice: int, item: dict, traza: dict, run_id: str) -> dict:
    referencias = [
        {"fuente": r.get("fuente", ""), "pagina": r.get("pagina")}
        for r in traza["recuperacion"]["resultados_validos"]
    ]
    ruido = medir_ruido_fuente(referencias, item.get("fuente_esperada", ""), k=5)
    return {
        "run_id": run_id,
        "indice": indice,
        "question": item["question"],
        "ground_truth": item["ground_truth"],
        "fuente_esperada": item.get("fuente_esperada"),
        "paginas_esperadas": item.get("paginas_esperadas", []),
        "recall_fuente_pagina_at_5": calcular_recall_fuente_pagina_at_k(
            referencias, item.get("fuente_esperada", ""), item.get("paginas_esperadas", []), k=5,
        ),
        "fuente_dominante_top_5": ruido["fuente_dominante"],
        "fuera_fuente_esperada_at_5": ruido["fuera_fuente_esperada_at_k"],
        "traza": traza,
    }


def resumir(registros: List[dict]) -> dict:
    validos = [r for r in registros if not r["traza"].get("error")]
    por_fuente: Dict[str, List[float]] = {}
    for r in validos:
        por_fuente.setdefault(r["fuente_esperada"] or "?", []).append(r["recall_fuente_pagina_at_5"])

    def _prom(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "n_preguntas": len(registros),
        "n_errores": len(registros) - len(validos),
        "n_fuera_de_dominio": sum(1 for r in validos if r["traza"]["clasificacion"].get("tema_valido") is False),
        "n_sin_contexto": sum(1 for r in validos if not r["traza"]["recuperacion"].get("contexto_suficiente")),
        "recall_fuente_pagina_at_5": _prom([r["recall_fuente_pagina_at_5"] for r in validos]),
        "recall_fuente_pagina_at_5_por_fuente": {f: _prom(v) for f, v in sorted(por_fuente.items())},
        "fuente_dominante_correcta_pct": _prom([
            1.0 if r["fuente_dominante_top_5"] == (r["fuente_esperada"] or "").lower() else 0.0 for r in validos
        ]),
        "duracion_promedio_s": _prom([r["traza"]["duracion_s"] for r in validos]),
    }


async def correr(limit: int, salida: Path) -> int:
    golden = GOLDEN_SET[:limit] if limit > 0 else GOLDEN_SET
    hechos = _cargar_hechos(salida)
    pendientes = [(i, item) for i, item in enumerate(golden) if i not in hechos]
    print(f"📋 Golden set: {len(golden)} preguntas | ya hechas: {len(golden) - len(pendientes)} | "
          f"pendientes: {len(pendientes)}")
    print(f"💾 Salida: {salida}")
    if not pendientes:
        print("✅ Nada pendiente.")
    else:
        asistente = await inicializar_asistente()
        run_id = uuid.uuid4().hex[:8]
        salida.parent.mkdir(parents=True, exist_ok=True)
        try:
            for n, (i, item) in enumerate(pendientes, 1):
                print(f"\n── [{n}/{len(pendientes)}] #{i} {item['question'][:70]}")
                resultado = await asistente.consultar(
                    item["question"], imagen_path=None,
                    user_id=f"baseline-{run_id}-{i}", origen="baseline",
                )
                traza = resultado["traza"]
                if traza.get("error") and _quota_blocked():
                    print("\n⛔ Sin cuota del modelo. La pregunta no se registró: volvé a correr el "
                          "mismo comando más tarde para retomar desde acá.")
                    break
                registro = construir_registro(i, item, traza, run_id)
                with open(salida, "a", encoding="utf-8") as f:
                    f.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")
                hechos[i] = registro
                print(f"   recall fuente+página@5 = {registro['recall_fuente_pagina_at_5']:.2f} | "
                      f"{traza['duracion_s']:.1f}s" + (f" | ERROR: {traza['error']}" if traza.get("error") else ""))
        finally:
            await asistente.cerrar()

    registros = [hechos[i] for i in sorted(hechos) if i < len(golden)]
    resumen = resumir(registros)
    resumen["generado"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ruta_resumen = salida.with_name(salida.stem + "-resumen.json")
    ruta_resumen.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 60 + "\n📊 RESUMEN BASELINE\n" + "=" * 60)
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    print(f"\n💾 {ruta_resumen}")
    return 0 if len(registros) == len(golden) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline del pipeline sobre el golden set")
    parser.add_argument("--limit", type=int, default=0, help="Solo las primeras N preguntas (0 = todas)")
    parser.add_argument("--salida", type=Path, default=None,
                        help="Archivo JSONL de salida. Si existe, se retoma (default: uno nuevo con fecha)")
    args = parser.parse_args()
    salida = args.salida or RESULTADOS_DIR / f"baseline-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    if not salida.is_absolute():
        salida = Path(os.getcwd()) / salida
    return asyncio.run(correr(args.limit, salida))


if __name__ == "__main__":
    sys.exit(main())
