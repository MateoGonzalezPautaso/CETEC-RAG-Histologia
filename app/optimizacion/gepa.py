"""
Optimización del prompt de respuesta con DSPy GEPA (Etapa 2, semanas 13–14).

GEPA (reflective prompt evolution) ejecuta el programa sobre ejemplos, pide a un
LLM "juez" puntaje y críticas textuales, y un LLM de reflexión propone versiones
mejores de la instrucción a partir de esas críticas.

- Programa: una Signature (pregunta + SECCIONES DEL MANUAL → respuesta) cuya
  instrucción inicial es la parte estática del prompt actual (src/prompts.py).
- Datos: un baseline de optimizacion/baseline.py. De cada registro se toma la
  pregunta, el contexto que el LLM vio en producción y la respuesta de
  referencia actual del golden set (si se corrigió después de la corrida, la
  corregida). Solo consultas de texto con contexto suficiente (las demás no
  pasan por el LLM de respuesta).
- Referencias del docente (opcional): --ejemplos-docente con
  [{"question": "...", "professor_response": "..."}] reemplaza la respuesta de
  referencia del manual para esas preguntas. Es lo que permite clonar el estilo
  docente; sin ese material, se optimiza contra la referencia del manual.
- Métrica: el mismo juez de optimizacion/juez.py (rúbrica y parseo), devuelto
  como puntaje 0–1 + feedback por dimensión.
- Salida: optimized_prompts.json con el formato del agente de Física de
  a2a-test-alone ({"metadata", "prompts": {"direct_response": ...}}), el módulo
  DSPy compilado y los logs de GEPA, en optimizacion/resultados/gepa-<fecha>/.

Para usar el resultado en el pipeline hay que copiarlo explícitamente:
    cp optimizacion/resultados/gepa-<fecha>/optimized_prompts.json optimizacion/optimized_prompts.json
    PROMPTS_OPTIMIZADOS=true  (en app/.env)

Uso (desde app/; necesita el grupo de dependencias `optim`):
    uv sync --group optim
    uv run python -m optimizacion.gepa optimizacion/resultados/baseline-completo.jsonl --dry-run
    uv run python -m optimizacion.gepa optimizacion/resultados/baseline-completo.jsonl --budget light
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from optimizacion.juez import (  # noqa: E402  (carga app/.env)
    DIMENSIONES, MODELOS_POR_DEFECTO, PESOS, RUBRICA, _leer_jsonl, _ultimo_por_indice,
    mensaje_juez, parsear_veredicto, puntaje_global,
)
from src.claves import RotadorClaves, _es_limite, opciones_razonamiento  # noqa: E402
from src.config import LLM_MODELO, MAX_CONTEXTO_PROMPT, normalizar  # noqa: E402
from src.prompts import CLAVE_RESPUESTA_TEXTO, instruccion_texto_default  # noqa: E402

try:
    import dspy
except ImportError:  # pragma: no cover - mensaje para quien no instaló el grupo optim
    print("❌ Falta DSPy. Instalalo con: uv sync --group optim")
    raise

# El generador de producción (src/assistant.py, LLM_MODELO) y un modelo más
# grande para proponer instrucciones.
MODELO_TAREA = f"groq/{LLM_MODELO}"
MODELO_REFLEXION = "groq/openai/gpt-oss-120b"
# Mismo truncado que _build_content_parts en producción.
MAX_CONTEXTO = MAX_CONTEXTO_PROMPT
# Tope de salida del generador. Groq rechaza (sin reintento posible) los pedidos
# cuyo max_tokens supera el límite de tokens de salida por minuto del modelo
# (OTPM: 1000 para Qwen en el plan gratuito). Una respuesta ronda 300 tokens.
MAX_TOKENS_TAREA = int(os.getenv("GEPA_MAX_TOKENS_TAREA", "900"))


# ── Modelos ───────────────────────────────────────────────────────────────────

class CuotaAgotada(RuntimeError):
    """Una llamada al LM falló por límite de uso aun después de rotar keys y esperar."""


# GEPA y dspy.Evaluate convierten cualquier excepción del programa o de la
# métrica en puntaje 0 y siguen. Si se agota la cuota a mitad de camino, la
# optimización seguiría con ceros y el prompt "optimizado" no valdría nada.
# Por eso el primer límite de uso que no se pudo resolver marca la corrida, las
# llamadas siguientes fallan sin ir a la API y main() descarta el resultado.
_ESTADO_CUOTA = {"agotada": None, "pedido_grande": False}


def cuota_agotada() -> Optional[str]:
    return _ESTADO_CUOTA["agotada"]


def _es_pedido_grande(error: Exception) -> bool:
    """413 / "Request too large": el pedido excede un límite del modelo; fallaría siempre igual."""
    if 413 in (getattr(error, "status_code", None), getattr(error, "status", None)):
        return True
    return "request too large" in str(error).lower()


def _es_transitorio(error: Exception) -> bool:
    """Errores de red o del servidor que conviene reintentar (no son límites de uso)."""
    if getattr(error, "status", None) in (408, 409, 500, 502, 503, 504, 529):
        return True
    nombre = type(error).__name__.lower()
    return any(t in nombre for t in ("timeout", "servererror", "transport", "connection", "unavailable"))


def _normalizar_salidas(salidas):
    """
    Una respuesta que sale del caché de DSPy viene como AttributeDict (subclase
    de dict) cuando el modelo razona (gpt-oss). GEPA valida la reflexión con
    type(x) == dict y la rechaza ("Unexpected output type from the base LM"):
    al retomar una corrida desde el caché ninguna reflexión funcionaba.
    """
    if not isinstance(salidas, list):
        return salidas
    return [dict(o) if isinstance(o, dict) else str(o) if isinstance(o, str) else o for o in salidas]


class LMRotativo(dspy.LM):
    """
    dspy.LM de Groq que rota las keys (GROQ_API_KEYS), espera ante límites por
    minuto y marca la corrida si se agota la cuota.

    Los 429 los resuelve el rotador: DSPy no reintenta por su cuenta
    (num_retries=0), porque dormiría el "try again in" de Groq, que con la cuota
    diaria agotada son minutos por llamada y la corrida parecería colgada.
    """

    INTENTOS_TRANSITORIOS = 3

    def __init__(self, modelo: str, rotador: RotadorClaves, **kwargs):
        super().__init__(modelo, **kwargs)
        self.rotador = rotador

    def _guardia(self, error: Exception):
        grande = _es_pedido_grande(error)
        if grande or _es_limite(error):
            _ESTADO_CUOTA["pedido_grande"] = _ESTADO_CUOTA["pedido_grande"] or grande
            if not _ESTADO_CUOTA["agotada"]:
                causa = "Pedido rechazado por tamaño" if grande else "Límite de uso sin resolver"
                print(f"\n⛔ {causa} en {self.model}: {str(error)[:200]}")
            _ESTADO_CUOTA["agotada"] = _ESTADO_CUOTA["agotada"] or f"{self.model}: {str(error)[:300]}"
            raise CuotaAgotada(str(error)) from error
        raise error

    def __call__(self, prompt=None, *, messages=None, **kwargs):
        for intento in range(self.INTENTOS_TRANSITORIOS):
            if _ESTADO_CUOTA["agotada"]:
                raise CuotaAgotada(_ESTADO_CUOTA["agotada"])
            try:
                return _normalizar_salidas(self.rotador.ejecutar(
                    lambda clave: super(LMRotativo, self).__call__(prompt, messages=messages, api_key=clave, **kwargs)))
            except Exception as e:
                if not _es_limite(e) and _es_transitorio(e) and intento < self.INTENTOS_TRANSITORIOS - 1:
                    time.sleep(3 * 2 ** intento)
                    continue
                self._guardia(e)

    async def acall(self, prompt=None, *, messages=None, **kwargs):
        import asyncio

        for intento in range(self.INTENTOS_TRANSITORIOS):
            if _ESTADO_CUOTA["agotada"]:
                raise CuotaAgotada(_ESTADO_CUOTA["agotada"])
            try:
                return _normalizar_salidas(await self.rotador.aejecutar(
                    lambda clave: super(LMRotativo, self).acall(prompt, messages=messages, api_key=clave, **kwargs)))
            except Exception as e:
                if not _es_limite(e) and _es_transitorio(e) and intento < self.INTENTOS_TRANSITORIOS - 1:
                    await asyncio.sleep(3 * 2 ** intento)
                    continue
                self._guardia(e)


def crear_lm(modelo: str, rotador: Optional[RotadorClaves] = None, **kwargs) -> "dspy.LM":
    """LM de Groq con rotación de keys, espera ante límites por minuto y guardia de cuota."""
    if modelo.startswith("groq/") and rotador is not None and len(rotador):
        kwargs = {**opciones_razonamiento(modelo), **kwargs, "num_retries": 0}
        return LMRotativo(modelo, rotador, **kwargs)
    return dspy.LM(modelo, **kwargs)


# ── Programa ──────────────────────────────────────────────────────────────────

class RespuestaHistologia(dspy.Signature):
    """(La instrucción real se inyecta con with_instructions: ver crear_programa)."""

    question: str = dspy.InputField(desc="Consulta del estudiante")
    context: str = dspy.InputField(desc="SECCIONES DEL MANUAL recuperadas: única fuente de verdad")
    response: str = dspy.OutputField(desc="Respuesta al estudiante, citando [Manual: archivo]")


class Respondedor(dspy.Module):
    def __init__(self, instruccion: str):
        super().__init__()
        # Predict y no ChainOfThought: en producción la respuesta se genera sin
        # razonamiento previo, y la instrucción optimizada se usa tal cual.
        self.responder = dspy.Predict(RespuestaHistologia.with_instructions(instruccion))

    def forward(self, question: str, context: str):
        return self.responder(question=question, context=context)


def crear_programa(instruccion: Optional[str] = None) -> Respondedor:
    return Respondedor(instruccion or instruccion_texto_default())


# ── Datos ─────────────────────────────────────────────────────────────────────

def _truncar_contexto(contexto: str) -> str:
    if len(contexto) > MAX_CONTEXTO:
        return contexto[:MAX_CONTEXTO] + "\n... [contexto truncado]"
    return contexto


def usar_referencias_actuales(registros: List[dict]) -> List[int]:
    """Reemplaza la referencia guardada en cada registro por la actual del golden
    set. Devuelve los índices cuya referencia cambió desde la corrida."""
    from evaluar_ragas import GOLDEN_SET  # importa el pipeline: solo cuando hace falta

    cambiadas = []
    for r in registros:
        item = GOLDEN_SET[r["indice"]] if r["indice"] < len(GOLDEN_SET) else None
        if item and item["question"] == r["question"] and item["ground_truth"] != r["ground_truth"]:
            r["ground_truth"] = item["ground_truth"]
            cambiadas.append(r["indice"])
    return cambiadas


def cargar_ejemplos_docente(ruta: Optional[Path]) -> Dict[str, str]:
    if not ruta:
        return {}
    with open(ruta, encoding="utf-8") as f:
        items = json.load(f)
    docentes = {}
    for item in items:
        pregunta = (item.get("question") or "").strip()
        respuesta = (item.get("professor_response") or "").strip()
        if pregunta and respuesta:
            docentes[normalizar(pregunta)] = respuesta
    return docentes


def construir_ejemplos(registros: List[dict], docentes: Dict[str, str]) -> Tuple[List["dspy.Example"], dict]:
    """Convierte registros del baseline en ejemplos DSPy. Devuelve (ejemplos, conteo de descartes)."""
    ejemplos = []
    descartes = {"error": 0, "sin_contexto": 0, "con_imagen": 0}
    for r in sorted(registros, key=lambda x: x["indice"]):
        traza = r["traza"]
        contexto = traza["recuperacion"].get("contexto_documentos") or ""
        if traza.get("error"):
            descartes["error"] += 1
            continue
        if traza["imagen"].get("tiene_imagen"):
            descartes["con_imagen"] += 1
            continue
        if not traza["recuperacion"].get("contexto_suficiente") or not contexto.strip():
            # Sin contexto el pipeline responde un texto fijo, sin pasar por el LLM.
            descartes["sin_contexto"] += 1
            continue
        referencia_docente = docentes.get(normalizar(r["question"]))
        ejemplos.append(dspy.Example(
            indice=r["indice"],
            fuente=r.get("fuente_esperada") or "?",
            question=traza["consulta"].get("reescrita") or r["question"],
            context=_truncar_contexto(contexto),
            reference=referencia_docente or r["ground_truth"],
            origen_referencia="docente" if referencia_docente else "manual",
        ).with_inputs("question", "context"))
    return ejemplos, descartes


def dividir(ejemplos: List["dspy.Example"], frac_val: float, semilla: int):
    """Split estratificado por fuente del manual, para que validación cubra todos los PDFs."""
    if len(ejemplos) < 4 or frac_val <= 0:
        return list(ejemplos), list(ejemplos)
    rng = random.Random(semilla)
    por_fuente: Dict[str, list] = {}
    for ex in ejemplos:
        por_fuente.setdefault(ex.fuente, []).append(ex)
    train, val = [], []
    for fuente in sorted(por_fuente):
        grupo = por_fuente[fuente][:]
        rng.shuffle(grupo)
        n_val = round(len(grupo) * frac_val) if len(grupo) > 1 else 0
        val += grupo[:n_val]
        train += grupo[n_val:]
    if not val:
        val = train[-1:]
    return sorted(train, key=lambda e: e.indice), sorted(val, key=lambda e: e.indice)


# ── Métrica (juez) ────────────────────────────────────────────────────────────

def _texto_salida(salida) -> str:
    item = salida[0] if isinstance(salida, list) and salida else salida
    if isinstance(item, dict):
        return item.get("text") or item.get("content") or ""
    return str(item or "")


def crear_metrica(juez_lm: "dspy.LM", intentos: int = 3):
    """Métrica compatible con GEPA: puntaje del juez (0–1) + feedback textual por dimensión."""

    def metrica(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
        respuesta = (getattr(pred, "response", "") or "").strip()
        if not respuesta:
            return dspy.Prediction(score=0.0, feedback="No se generó ninguna respuesta.")
        mensajes = [
            {"role": "system", "content": RUBRICA},
            {"role": "user", "content": mensaje_juez(
                gold.question, gold.reference, gold.context, respuesta, gold.origen_referencia)},
        ]
        error = None
        for intento in range(intentos):
            try:
                # rollout_id distinto en cada reintento: si no, el caché de DSPy
                # devolvería el mismo JSON inválido.
                extra = {"rollout_id": intento, "temperature": 0.2} if intento else {}
                veredicto = parsear_veredicto(_texto_salida(juez_lm(messages=mensajes, **extra)))
                break
            except ValueError as e:
                error = e
        else:
            print(f"   ⚠️ Juez inválido tras {intentos} intentos ({error}); puntaje 0 para #{gold.indice}")
            return dspy.Prediction(score=0.0, feedback=f"El juez no devolvió un veredicto válido: {error}")

        feedback = "\n".join(
            f"[{d}] {veredicto[d]['puntaje']}/5 — {veredicto[d]['justificacion']}" for d in DIMENSIONES
        )
        return dspy.Prediction(score=puntaje_global(veredicto), feedback=feedback)

    return metrica


# ── Salida ────────────────────────────────────────────────────────────────────

def instruccion_de(programa: Respondedor) -> str:
    return programa.responder.signature.instructions


def demos_de(programa: Respondedor) -> List[dict]:
    demos = []
    for demo in getattr(programa.responder, "demos", []) or []:
        demos.append({k: str(v) for k, v in demo.items() if k in ("question", "context", "response")})
    return demos


def evaluar(programa, ejemplos, metrica) -> Tuple[float, Dict[int, float]]:
    puntajes = {}
    for ex in ejemplos:
        pred = programa(question=ex.question, context=ex.context)
        puntajes[ex.indice] = float(metrica(ex, pred).score)
    promedio = sum(puntajes.values()) / len(puntajes) if puntajes else 0.0
    return round(promedio, 4), puntajes


def _modelo_juez_por_defecto() -> str:
    proveedor = (os.getenv("JUEZ_PROVEEDOR") or "groq").strip().lower()
    modelo = os.getenv("JUEZ_MODELO") or MODELOS_POR_DEFECTO.get(proveedor, MODELOS_POR_DEFECTO["groq"])
    return f"{proveedor}/{modelo}"


def _abortar_por_cuota(salida: Path) -> int:
    if _ESTADO_CUOTA["pedido_grande"]:
        print(
            "\n⛔ Groq rechazó un pedido por ser demasiado grande para el modelo. El resultado NO se "
            "guardó.\n"
            f"   Motivo: {cuota_agotada()}\n"
            "   Si habla de output tokens (OTPM), bajá GEPA_MAX_TOKENS_TAREA en app/.env (o "
            "--max-tokens-tarea). Si habla de tokens por minuto (TPM), el contexto es muy largo.\n"
            f"   Logs parciales: {salida}"
        )
        return 2
    print(
        "\n⛔ Se agotó la cuota de Groq durante la optimización. El resultado NO se guardó: "
        "las evaluaciones posteriores al corte valen 0 y el prompt no sería válido.\n"
        f"   Motivo: {cuota_agotada()}\n"
        "   Volvé a correr el mismo comando (sin --salida, o con una carpeta nueva) cuando se "
        "renueve la cuota o con más keys en GROQ_API_KEYS: DSPy reutiliza desde su caché las "
        "llamadas que ya se hicieron, así que retoma casi donde quedó.\n"
        f"   Logs parciales: {salida}"
    )
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimiza el prompt de respuesta con DSPy GEPA")
    parser.add_argument("baseline", type=Path, help="JSONL generado por optimizacion.baseline")
    parser.add_argument("--ejemplos-docente", type=Path, default=None,
                        help='JSON [{"question", "professor_response"}] con respuestas de referencia del docente')
    parser.add_argument("--budget", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--max-metric-calls", type=int, default=None, help="Reemplaza --budget por un tope exacto")
    parser.add_argument("--frac-val", type=float, default=0.3, help="Fracción de ejemplos para validación")
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument("--modelo", default=MODELO_TAREA, help="LM que genera las respuestas (LiteLLM)")
    parser.add_argument("--modelo-reflexion", default=MODELO_REFLEXION, help="LM que propone instrucciones")
    parser.add_argument("--modelo-juez", default=None, help="LM juez (default: JUEZ_PROVEEDOR/JUEZ_MODELO)")
    parser.add_argument("--max-tokens-tarea", type=int, default=MAX_TOKENS_TAREA,
                        help="Tope de tokens de la respuesta (debe quedar bajo el OTPM del modelo)")
    parser.add_argument("--reflexion-temperatura", type=float, default=1.0)
    parser.add_argument("--hilos", type=int, default=1, help="Evaluaciones en paralelo (1 = amigable con la cuota)")
    parser.add_argument("--salida", type=Path, default=None, help="Carpeta de salida (default: resultados/gepa-<fecha>)")
    parser.add_argument("--dry-run", action="store_true", help="Solo prueba programa + juez sobre un ejemplo")
    args = parser.parse_args()

    registros = _ultimo_por_indice(_leer_jsonl(args.baseline.resolve()))
    if not registros:
        print(f"❌ No hay registros en {args.baseline}")
        return 1
    cambiadas = usar_referencias_actuales(registros)
    if cambiadas:
        print(f"📝 Referencias corregidas en el golden set desde la corrida (se usa la actual): {cambiadas}")
    docentes = cargar_ejemplos_docente(args.ejemplos_docente)
    ejemplos, descartes = construir_ejemplos(registros, docentes)
    n_docente = sum(1 for e in ejemplos if e.origen_referencia == "docente")
    print(f"📚 {len(ejemplos)} ejemplos de {len(registros)} registros | descartados: {descartes} | "
          f"referencias del docente: {n_docente}")
    if not ejemplos:
        print("❌ Ningún registro sirve como ejemplo (¿el baseline tiene errores o no recuperó contexto?)")
        return 1
    if n_docente == 0:
        print("ℹ️ Sin --ejemplos-docente: se optimiza contra la respuesta de referencia del manual.")

    modelo_juez = args.modelo_juez or _modelo_juez_por_defecto()
    rotador = RotadorClaves.desde_entorno()
    if len(rotador) > 1:
        print(f"🔑 Rotación de keys de Groq: {len(rotador)} keys")
    lm_tarea = crear_lm(args.modelo, rotador, temperature=0.0, max_tokens=args.max_tokens_tarea,
                        num_retries=8)
    lm_juez = crear_lm(modelo_juez, rotador, temperature=0.0, max_tokens=2048, num_retries=8)
    lm_reflexion = crear_lm(args.modelo_reflexion, rotador, temperature=args.reflexion_temperatura,
                            max_tokens=8192, num_retries=8)
    dspy.configure(lm=lm_tarea)
    metrica = crear_metrica(lm_juez)
    print(f"🤖 Tarea: {args.modelo} | Juez: {modelo_juez} | Reflexión: {args.modelo_reflexion}")

    semilla = crear_programa()

    if args.dry_run:
        ex = ejemplos[0]
        print(f"\n🧪 DRY RUN con #{ex.indice}: {ex.question}")
        try:
            pred = semilla(question=ex.question, context=ex.context)
            print(f"\n📝 Respuesta:\n{pred.response[:800]}")
            resultado = metrica(ex, pred)
        except CuotaAgotada as e:
            print(f"\n⛔ Sin cuota de Groq: {e}")
            return 2
        print(f"\n⚖️  Puntaje: {resultado.score:.3f}\n{resultado.feedback}")
        print("\n✅ Dry run completo. Sacá --dry-run para optimizar.")
        return 0

    train, val = dividir(ejemplos, args.frac_val, args.semilla)
    print(f"✂️  Train: {len(train)} {[e.indice for e in train]} | Val: {len(val)} {[e.indice for e in val]}")

    salida = args.salida or APP_DIR / "optimizacion" / "resultados" / f"gepa-{time.strftime('%Y%m%d-%H%M%S')}"
    salida.mkdir(parents=True, exist_ok=True)
    presupuesto = {"max_metric_calls": args.max_metric_calls} if args.max_metric_calls else {"auto": args.budget}

    t0 = time.time()
    optimizador = dspy.GEPA(
        metric=metrica,
        reflection_lm=lm_reflexion,
        candidate_selection_strategy="pareto",
        use_merge=True,
        num_threads=args.hilos,
        track_stats=True,
        log_dir=str(salida / "logs"),
        seed=args.semilla,
        **presupuesto,
    )
    optimizado = optimizador.compile(semilla, trainset=train, valset=val)
    duracion = time.time() - t0
    if cuota_agotada():
        return _abortar_por_cuota(salida)
    print(f"\n⏱️ GEPA terminó en {duracion / 60:.1f} min")

    # Puntajes en validación, con la misma métrica, de la semilla y del optimizado.
    try:
        val_semilla, por_indice_semilla = evaluar(semilla, val, metrica)
        val_optimizado, por_indice_opt = evaluar(optimizado, val, metrica)
    except CuotaAgotada:
        return _abortar_por_cuota(salida)

    resultado = {
        "metadata": {
            "optimizer": "dspy.GEPA",
            "dspy_version": getattr(dspy, "__version__", None),
            "model": args.modelo,
            "reflection_model": args.modelo_reflexion,
            "judge_model": modelo_juez,
            "judge_weights": PESOS,
            "budget": None if args.max_metric_calls else args.budget,
            "max_metric_calls": args.max_metric_calls,
            "baseline": str(args.baseline),
            "num_train": len(train),
            "num_val": len(val),
            "indices_train": [e.indice for e in train],
            "indices_val": [e.indice for e in val],
            "referencias_docente": n_docente,
            "val_score_seed": val_semilla,
            "val_score_optimized": val_optimizado,
            "val_scores_seed_por_indice": por_indice_semilla,
            "val_scores_optimized_por_indice": por_indice_opt,
            "optimization_time_seconds": round(duracion, 1),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "prompts": {
            CLAVE_RESPUESTA_TEXTO: {"instruction": instruccion_de(optimizado), "demos": demos_de(optimizado)},
        },
    }
    ruta_prompts = salida / "optimized_prompts.json"
    ruta_prompts.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        optimizado.save(str(salida / "optimized_prompts_module.json"))
    except Exception as e:
        print(f"⚠️ No se pudo guardar el módulo DSPy: {e}")

    print("\n" + "=" * 60 + "\n📊 VALIDACIÓN (juez, 0–1)\n" + "=" * 60)
    print(f"   Semilla (prompt actual): {val_semilla:.3f}")
    print(f"   Optimizado por GEPA:     {val_optimizado:.3f} ({val_optimizado - val_semilla:+.3f})")
    print("\n📝 Instrucción optimizada:\n" + instruccion_de(optimizado))
    print(f"\n💾 {ruta_prompts}")
    print("   Para usarla: copiala a optimizacion/optimized_prompts.json y poné PROMPTS_OPTIMIZADOS=true")
    if val_optimizado <= val_semilla:
        print("⚠️ El optimizado no supera a la semilla en validación: no conviene activarlo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
