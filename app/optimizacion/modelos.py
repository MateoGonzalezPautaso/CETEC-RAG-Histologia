"""
Chequea los modelos de Groq antes de correr el baseline, el juez o GEPA.

Groq cambia seguido los modelos del plan gratuito (p. ej. retiró Llama-4-Scout
y Llama 3.3 70B en 2026). Este script lista los modelos que ve cada key y
avisa si falta alguno de los que usa el proyecto. Con --probar además hace una
llamada corta a cada uno (y una con imagen al generador, que la necesita para
el modo multimodal), con las mismas opciones que el pipeline.

Uso (desde app/):
    uv run python -m optimizacion.modelos
    uv run python -m optimizacion.modelos --probar
"""

import argparse
import asyncio
import base64
import io
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from optimizacion.juez import crear_llm_juez  # noqa: E402  (carga app/.env)
from src.claves import RotadorClaves, _sufijo, crear_chat_groq  # noqa: E402
from src.config import LLM_MODELO  # noqa: E402
from src.llm import invoke_con_reintento  # noqa: E402

MODELO_REFLEXION = "openai/gpt-oss-120b"  # el de optimizacion/gepa.py, sin el prefijo groq/


def listar(claves):
    from groq import Groq

    vistos = None
    for clave in claves:
        try:
            ids = sorted(m.id for m in Groq(api_key=clave).models.list().data)
        except Exception as e:
            print(f"❌ Key {_sufijo(clave)}: {e}")
            continue
        print(f"🔑 Key {_sufijo(clave)}: {len(ids)} modelos")
        for i in ids:
            print(f"     {i}")
        vistos = set(ids) if vistos is None else vistos & set(ids)
    return vistos or set()


def _imagen_prueba() -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 30, 30)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def probar(nombre, llm, contenido):
    from langchain_core.messages import HumanMessage

    try:
        resp = await invoke_con_reintento(llm, [HumanMessage(content=contenido)])
        uso = getattr(resp, "usage_metadata", None) or {}
        print(f"   ✅ {nombre}: {resp.content.strip()[:80]!r} | tokens: {uso.get('total_tokens', '?')}")
        return True
    except Exception as e:
        print(f"   ❌ {nombre}: {e}")
        return False


async def main_async(args) -> int:
    rotador = RotadorClaves.desde_entorno()
    if not len(rotador):
        print("❌ No hay keys: completá GROQ_API_KEYS o GROQ_API_KEY en app/.env")
        return 1

    llm_juez, nombre_juez = crear_llm_juez()
    modelo_juez = nombre_juez.split("/", 1)[1] if nombre_juez.startswith("groq/") else None
    usados = {"generador (LLM_MODELO)": LLM_MODELO, "reflexión GEPA": MODELO_REFLEXION}
    if modelo_juez:
        usados["juez (JUEZ_MODELO)"] = modelo_juez

    disponibles = listar(rotador.claves)
    print("\n📋 Modelos del proyecto:")
    faltan = 0
    for rol, modelo in usados.items():
        ok = modelo in disponibles
        faltan += not ok
        print(f"   {'✅' if ok else '❌'} {rol}: {modelo}")
    if faltan:
        print("\n⚠️ Algún modelo no está disponible para todas las keys. Elegí otro de la lista y "
              "configuralo en app/.env (LLM_MODELO / JUEZ_MODELO).")

    if not args.probar:
        return 1 if faltan else 0

    print("\n🧪 Llamadas de prueba:")
    generador = crear_chat_groq(LLM_MODELO, temperature=0, max_retries=1)
    resultados = [
        await probar("generador, texto", generador, "Respondé solo con la palabra OK."),
        await probar("generador, imagen", generador, [
            {"type": "text", "text": "¿De qué color es esta imagen? Respondé con una palabra."},
            {"type": "image_url", "image_url": {"url": _imagen_prueba()}},
        ]),
        await probar(f"juez ({nombre_juez})", llm_juez, 'Respondé solo con este JSON: {"ok": true}'),
    ]
    if MODELO_REFLEXION != modelo_juez:
        reflexion = crear_chat_groq(MODELO_REFLEXION, temperature=0, max_retries=1)
        resultados.append(await probar("reflexión GEPA", reflexion, "Respondé solo con la palabra OK."))
    return 0 if all(resultados) and not faltan else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Lista y prueba los modelos de Groq del proyecto")
    parser.add_argument("--probar", action="store_true", help="Hace una llamada corta a cada modelo")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
