"""
LLM as a Judge sobre los resultados del baseline (Etapa 2, semana 12).

Para cada registro de `optimizacion/baseline.py` evalúa la respuesta del
sistema en cuatro dimensiones (1 a 5, con justificación):

- correccion:      coincide con la respuesta de referencia del manual.
- fidelidad:       todo lo afirmado está sustentado en el contexto recuperado.
- notacion:        terminología y notación histológica correctas y precisas.
- estilo_docente:  explica como un docente (ver RUBRICA). Es un proxy hasta
                   contar con material de los docentes de la cátedra.

Además calcula chequeos deterministas de citas ([Manual: archivo]), sin LLM.
Las justificaciones se guardan como `feedback`: es el texto que GEPA usa para
proponer mejoras de prompt.

El modelo juez se elige con JUEZ_PROVEEDOR (groq | openai) y JUEZ_MODELO. Por
defecto, Groq openai/gpt-oss-120b (llama-3.3-70b-versatile salió del plan
gratuito el 16/08/2026): modelos chicos devuelven JSON inválido (ver Informe
Sprint 1). Si se agota la cuota, se corta y se retoma volviendo a
correr el mismo comando.

Uso (desde app/; no necesita el servidor ni los modelos de visión):
    uv run python -m optimizacion.juez optimizacion/resultados/baseline-X.jsonl
    uv run python -m optimizacion.juez optimizacion/resultados/baseline-X.jsonl --limit 3

Si se corrigen referencias del golden set (evaluar_ragas.GOLDEN_SET) después de
una corrida, --referencias-actuales vuelve a juzgar solo las preguntas cuya
referencia cambió y copia los demás juicios de baseline-X-juez.jsonl; escribe
baseline-X-juez-ref.jsonl, comparable con las corridas nuevas:
    uv run python -m optimizacion.juez optimizacion/resultados/baseline-X.jsonl --referencias-actuales
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(APP_DIR / ".env")

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from src.llm import _quota_blocked, invoke_con_reintento  # noqa: E402

DIMENSIONES = ["correccion", "fidelidad", "notacion", "estilo_docente"]

# Peso de cada dimensión en el puntaje global (0 a 1). El plan prevé ajustarlos
# (semana 7 del plan: precisión vs. clonación de estilo).
PESOS = {"correccion": 0.35, "fidelidad": 0.30, "notacion": 0.20, "estilo_docente": 0.15}

MODELOS_POR_DEFECTO = {"groq": "openai/gpt-oss-120b", "openai": "gpt-4o-mini"}

RUBRICA = """Sos un docente de Histología de la Facultad de Ingeniería de la UBA evaluando
las respuestas de un asistente para estudiantes. Evaluá la RESPUESTA DEL SISTEMA en
cuatro dimensiones, cada una con un puntaje entero de 1 a 5:

1. correccion — ¿Coincide con la RESPUESTA DE REFERENCIA (extraída del manual)?
   5: cubre todos los puntos clave sin errores. 3: cubre parte o tiene imprecisiones
   menores. 1: incorrecta, contradice la referencia o no responde.
2. fidelidad — ¿Todo lo que afirma está sustentado en el CONTEXTO RECUPERADO?
   5: nada inventado. 3: agrega detalles no presentes en el contexto pero plausibles.
   1: afirma cosas ausentes del contexto o contradictorias con él.
3. notacion — ¿Usa terminología y notación histológica correctas y precisas
   (nombres de estructuras, células, tinciones como H&E, capas, orientación del corte)?
   5: precisa y consistente. 3: correcta pero vaga o con algún término impropio.
   1: terminología errónea o coloquial.
4. estilo_docente — ¿Explica como lo haría un docente de la cátedra? Responde
   primero la pregunta en 1 a 3 frases, luego explica en prosa didáctica y natural,
   relaciona estructura con función o con lo que se observa al microscopio cuando
   corresponde, y evita listas innecesarias y relleno.
   5: así respondería un docente. 3: correcta pero enciclopédica, rígida o desordenada.
   1: confusa, telegráfica o impropia para un estudiante.

Para cada dimensión escribí una justificación breve y concreta que diga QUÉ mejorar.

Respondé ÚNICAMENTE con JSON válido, sin backticks ni texto adicional:
{"correccion": {"puntaje": 1-5, "justificacion": "..."},
 "fidelidad": {"puntaje": 1-5, "justificacion": "..."},
 "notacion": {"puntaje": 1-5, "justificacion": "..."},
 "estilo_docente": {"puntaje": 1-5, "justificacion": "..."}}"""


# ── Modelo juez ───────────────────────────────────────────────────────────────

def crear_llm_juez(proveedor: Optional[str] = None, modelo: Optional[str] = None):
    proveedor = (proveedor or os.getenv("JUEZ_PROVEEDOR") or "groq").strip().lower()
    if proveedor not in MODELOS_POR_DEFECTO:
        raise ValueError(f"JUEZ_PROVEEDOR inválido: {proveedor!r} (usar groq u openai)")
    modelo = modelo or os.getenv("JUEZ_MODELO") or MODELOS_POR_DEFECTO[proveedor]
    if proveedor == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=modelo, api_key=os.getenv("OPENAI_API_KEY"), temperature=0, max_retries=1)
    else:
        from src.claves import crear_chat_groq
        llm = crear_chat_groq(modelo, temperature=0, max_retries=1)
    return llm, f"{proveedor}/{modelo}"


# ── Evaluación ────────────────────────────────────────────────────────────────

def chequear_citas(respuesta: str, fuentes_recuperadas: List[str], fuente_esperada: Optional[str]) -> dict:
    """Chequeos deterministas de citas [Manual: a.pdf, b.pdf] en la respuesta."""
    citadas = set()
    for grupo in re.findall(r"\[Manual:\s*([^\]]+)\]", respuesta or "", flags=re.IGNORECASE):
        for nombre in grupo.split(","):
            nombre = os.path.basename(nombre.strip()).lower()
            if nombre:
                citadas.add(nombre)
    recuperadas = {os.path.basename(f).lower() for f in fuentes_recuperadas if f}
    esperada = os.path.basename(fuente_esperada or "").lower()
    return {
        "fuentes_citadas": sorted(citadas),
        "tiene_cita": bool(citadas),
        # Toda fuente citada debe estar entre las recuperadas (si no, es una cita inventada).
        "citas_en_contexto": bool(citadas) and citadas <= recuperadas,
        "cita_fuente_esperada": bool(esperada) and esperada in citadas,
    }


def mensaje_juez(pregunta: str, referencia: str, contexto: str, respuesta: str,
                 origen_referencia: str = "manual") -> str:
    """Mensaje para el juez. Lo comparten la evaluación y la métrica de GEPA (optimizacion/gepa.py)."""
    return (
        f"PREGUNTA DEL ESTUDIANTE:\n{pregunta}\n\n"
        f"RESPUESTA DE REFERENCIA ({origen_referencia}):\n{referencia}\n\n"
        f"CONTEXTO RECUPERADO (lo único que el sistema tenía disponible):\n"
        f"{(contexto or '(sin contexto recuperado)')[:6000]}\n\n"
        f"RESPUESTA DEL SISTEMA:\n{respuesta}"
    )


def construir_mensaje(registro: dict) -> str:
    traza = registro["traza"]
    return mensaje_juez(
        registro["question"], registro["ground_truth"],
        traza["recuperacion"].get("contexto_documentos"), traza["respuesta"]["texto"],
    )


def parsear_veredicto(texto: str) -> Dict[str, dict]:
    """Extrae y valida el JSON del juez. Lanza ValueError si es inválido."""
    texto = re.sub(r"```(?:json)?\s*|\s*```", "", (texto or "").strip())
    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if not match:
        raise ValueError("el juez no devolvió JSON")
    data = json.loads(match.group(0))
    veredicto = {}
    for dim in DIMENSIONES:
        d = data.get(dim)
        if not isinstance(d, dict):
            raise ValueError(f"falta la dimensión {dim!r}")
        puntaje = d.get("puntaje")
        if isinstance(puntaje, str) and puntaje.strip().isdigit():
            puntaje = int(puntaje.strip())
        if not isinstance(puntaje, (int, float)) or not 1 <= puntaje <= 5:
            raise ValueError(f"puntaje inválido en {dim!r}: {puntaje!r}")
        veredicto[dim] = {"puntaje": int(round(puntaje)), "justificacion": str(d.get("justificacion", "")).strip()}
    return veredicto


def puntaje_global(veredicto: Dict[str, dict]) -> float:
    """Promedio ponderado normalizado a 0–1 (1 → 0.0, 5 → 1.0)."""
    return round(sum(PESOS[d] * (veredicto[d]["puntaje"] - 1) / 4 for d in DIMENSIONES) / sum(PESOS.values()), 4)


async def juzgar(llm, registro: dict) -> dict:
    resp = await invoke_con_reintento(llm, [
        SystemMessage(content=RUBRICA),
        HumanMessage(content=construir_mensaje(registro)),
    ])
    veredicto = parsear_veredicto(resp.content)
    fuentes = [r.get("fuente") for r in registro["traza"]["recuperacion"]["resultados_validos"]]
    return {
        "veredicto": veredicto,
        "puntaje_global": puntaje_global(veredicto),
        "citas": chequear_citas(registro["traza"]["respuesta"]["texto"], fuentes, registro.get("fuente_esperada")),
        "feedback": "\n".join(f"[{d}] {veredicto[d]['justificacion']}" for d in DIMENSIONES),
    }


# ── Resumen ───────────────────────────────────────────────────────────────────

def resumir(juicios: List[dict]) -> dict:
    def _prom(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    por_fuente: Dict[str, List[float]] = {}
    for j in juicios:
        por_fuente.setdefault(j.get("fuente_esperada") or "?", []).append(j["puntaje_global"])
    return {
        "n_juzgados": len(juicios),
        "puntaje_global": _prom([j["puntaje_global"] for j in juicios]),
        "por_dimension": {d: _prom([j["veredicto"][d]["puntaje"] for j in juicios]) for d in DIMENSIONES},
        "puntaje_global_por_fuente": {f: _prom(v) for f, v in sorted(por_fuente.items())},
        "tiene_cita_pct": _prom([1.0 if j["citas"]["tiene_cita"] else 0.0 for j in juicios]),
        "citas_en_contexto_pct": _prom([1.0 if j["citas"]["citas_en_contexto"] else 0.0 for j in juicios]),
        "cita_fuente_esperada_pct": _prom([1.0 if j["citas"]["cita_fuente_esperada"] else 0.0 for j in juicios]),
        "pesos": PESOS,
    }


def _ultimo_por_indice(registros: List[dict]) -> List[dict]:
    """Un registro por pregunta: el baseline agrega al final los reintentos de
    preguntas que habían fallado, así que gana el último."""
    return sorted({r["indice"]: r for r in registros}.values(), key=lambda r: r["indice"])


def _leer_jsonl(ruta: Path) -> List[dict]:
    if not ruta.exists():
        return []
    salida = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            try:
                salida.append(json.loads(linea))
            except json.JSONDecodeError:
                continue
    return salida


def _actualizar_referencias(registros: List[dict], previos: Dict[int, dict], hechos: Dict[int, dict],
                            salida: Path) -> None:
    """Pone en cada registro la referencia actual del golden set. Las preguntas
    cuya referencia no cambió reutilizan el juicio anterior (se copia a `salida`)."""
    from evaluar_ragas import GOLDEN_SET  # importa el pipeline: solo cuando hace falta

    cambiadas = []
    for r in registros:
        item = GOLDEN_SET[r["indice"]] if r["indice"] < len(GOLDEN_SET) else None
        if not item or item["question"] != r["question"]:
            continue
        if item["ground_truth"] != r["ground_truth"]:
            r["ground_truth"] = item["ground_truth"]
            cambiadas.append(r["indice"])
        elif r["indice"] in previos and r["indice"] not in hechos:
            hechos[r["indice"]] = previos[r["indice"]]
            with open(salida, "a", encoding="utf-8") as f:
                f.write(json.dumps(previos[r["indice"]], ensure_ascii=False) + "\n")
    print(f"📝 Referencias cambiadas desde la corrida: {cambiadas or 'ninguna'} (se juzgan de nuevo); "
          f"el resto se copia de {salida.name.replace('-juez-ref', '-juez')}")


async def correr(entrada: Path, limit: int, proveedor: Optional[str], modelo: Optional[str],
                 referencias_actuales: bool = False) -> int:
    registros = _ultimo_por_indice(_leer_jsonl(entrada))
    if limit > 0:
        registros = registros[:limit]
    sufijo = "-juez-ref" if referencias_actuales else "-juez"
    salida = entrada.with_name(entrada.stem + sufijo + ".jsonl")
    hechos = {j["indice"]: j for j in _leer_jsonl(salida)}
    if referencias_actuales:
        previos = {j["indice"]: j for j in _leer_jsonl(entrada.with_name(entrada.stem + "-juez.jsonl"))}
        _actualizar_referencias(registros, previos, hechos, salida)

    llm, nombre_juez = crear_llm_juez(proveedor, modelo)
    omitidos = [r["indice"] for r in registros if r["traza"].get("error")]
    pendientes = [r for r in registros if r["indice"] not in hechos and not r["traza"].get("error")]
    print(f"⚖️  Juez: {nombre_juez} | registros: {len(registros)} | ya juzgados: {len(hechos)} | "
          f"pendientes: {len(pendientes)} | con error en el pipeline (se omiten): {len(omitidos)}")

    fallidos = 0
    for n, registro in enumerate(pendientes, 1):
        print(f"── [{n}/{len(pendientes)}] #{registro['indice']} {registro['question'][:70]}")
        try:
            juicio = await juzgar(llm, registro)
        except Exception as e:
            if _quota_blocked():
                print("⛔ Sin cuota del juez. Volvé a correr el mismo comando más tarde para retomar.")
                break
            # Un JSON inválido no se registra: la próxima corrida lo reintenta.
            fallidos += 1
            print(f"   ⚠️ Juicio inválido, se reintentará en la próxima corrida: {e}")
            continue
        juicio.update({
            "indice": registro["indice"], "trace_id": registro["traza"]["trace_id"],
            "fuente_esperada": registro.get("fuente_esperada"), "juez": nombre_juez,
            "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        with open(salida, "a", encoding="utf-8") as f:
            f.write(json.dumps(juicio, ensure_ascii=False) + "\n")
        hechos[registro["indice"]] = juicio
        v = juicio["veredicto"]
        print("   " + " | ".join(f"{d}={v[d]['puntaje']}" for d in DIMENSIONES)
              + f" | global={juicio['puntaje_global']:.2f}")

    indices = {r["indice"] for r in registros}
    resumen = resumir([hechos[i] for i in sorted(hechos) if i in indices])
    resumen.update({"juez": nombre_juez, "omitidos_por_error": omitidos, "fallidos_esta_corrida": fallidos})
    ruta_resumen = entrada.with_name(entrada.stem + sufijo + "-resumen.json")
    ruta_resumen.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 60 + "\n📊 RESUMEN JUEZ\n" + "=" * 60)
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    print(f"\n💾 {salida}\n💾 {ruta_resumen}")
    return 0 if resumen["n_juzgados"] == len(registros) - len(omitidos) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM as a Judge sobre un baseline")
    parser.add_argument("entrada", type=Path, help="JSONL generado por optimizacion.baseline")
    parser.add_argument("--limit", type=int, default=0, help="Solo los primeros N registros (0 = todos)")
    parser.add_argument("--proveedor", choices=sorted(MODELOS_POR_DEFECTO), default=None,
                        help="Proveedor del juez (default: JUEZ_PROVEEDOR o groq)")
    parser.add_argument("--modelo", default=None, help="Modelo del juez (default: JUEZ_MODELO o el del proveedor)")
    parser.add_argument("--referencias-actuales", action="store_true",
                        help="Juzga con las referencias actuales del golden set (solo rejuzga las que cambiaron)")
    args = parser.parse_args()
    return asyncio.run(correr(args.entrada.resolve(), args.limit, args.proveedor, args.modelo,
                              args.referencias_actuales))


if __name__ == "__main__":
    sys.exit(main())
