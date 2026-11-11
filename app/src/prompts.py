"""
Prompt de respuesta optimizable con DSPy GEPA (Etapa 2).

La parte estática del prompt del sistema en modo texto (reglas + estilo) puede
reemplazarse por una instrucción evolucionada por `optimizacion/gepa.py`. Las
partes dinámicas (ontología, continuidad del diálogo, imágenes) se siguen
agregando en tiempo de ejecución.

El archivo usa el mismo formato que el agente de Física de a2a-test-alone
(`optimized_prompts.json`), para poder reutilizarlo en ese repo:

    {"metadata": {...},
     "prompts": {"direct_response": {"instruction": "...", "demos": [...]}}}

Si no está habilitado (PROMPTS_OPTIMIZADOS=false, el default) o el archivo no
existe, el pipeline usa el prompt original sin cambios.
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

CLAVE_RESPUESTA_TEXTO = "direct_response"

INSTRUCCION_BASE_TEXTO = (
    "Eres un asistente experto de histología. Respondés consultas de texto "
    "basándote EXCLUSIVAMENTE en el contenido del manual/base de datos.\n\n"
    "REGLAS:\n"
    "1. Usá SOLO la información de las SECCIONES DEL MANUAL proporcionadas.\n"
    "2. Citá las fuentes con [Manual: archivo].\n"
    "3. NO inventes información que no esté en las secciones proporcionadas.\n"
    "4. Si el tema NO aparece en el manual ni en la ontología, indicalo.\n\n"
)

INSTRUCCION_PROSA = (
    "ESTILO DE RESPUESTA:\n"
    "- Respondé en prosa, como un profesor explicando.\n"
    "- Evitá listas con bullets y formato estructurado rígido.\n"
    "- Primero respondé directamente la pregunta en 1 a 3 frases.\n"
    "- Agregá explicación solo si aporta al punto consultado.\n"
    "- Tono didáctico y natural.\n"
)


def instruccion_texto_default() -> str:
    """Parte estática del prompt de respuesta actual: la semilla que optimiza GEPA."""
    return INSTRUCCION_BASE_TEXTO + INSTRUCCION_PROSA


class PromptsOptimizados:
    """Carga `optimized_prompts.json` una vez y expone instrucciones y demos por clave."""

    def __init__(self, ruta: str, habilitado: bool = False):
        self.ruta = ruta
        self.habilitado = habilitado
        self.prompts: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}
        self.sha256 = ""
        if habilitado:
            self._cargar()

    def _cargar(self) -> None:
        if not os.path.exists(self.ruta):
            print(f"ℹ️ PROMPTS_OPTIMIZADOS activo pero no existe {self.ruta}: se usan los prompts por defecto")
            return
        try:
            with open(self.ruta, "rb") as f:
                crudo = f.read()
            data = json.loads(crudo.decode("utf-8"))
            self.prompts = data.get("prompts", {}) or {}
            self.metadata = data.get("metadata", {}) or {}
            self.sha256 = hashlib.sha256(crudo).hexdigest()
            print(f"🧠 Prompts optimizados cargados desde {os.path.basename(self.ruta)} "
                  f"({', '.join(self.prompts) or 'vacío'}; sha256 {self.sha256[:8]})")
        except Exception as e:
            # Un archivo corrupto no debe tirar el servidor: se sigue con los defaults.
            print(f"⚠️ No se pudieron cargar los prompts optimizados ({e}): se usan los prompts por defecto")
            self.prompts, self.metadata, self.sha256 = {}, {}, ""

    def instruccion(self, clave: str) -> Optional[str]:
        instruccion = (self.prompts.get(clave) or {}).get("instruction", "")
        return instruccion.strip() or None

    def demos_texto(self, clave: str) -> str:
        demos: List[Dict[str, Any]] = (self.prompts.get(clave) or {}).get("demos", []) or []
        if not demos:
            return ""
        bloques = []
        for i, demo in enumerate(demos, 1):
            lineas = [f"--- Ejemplo {i} ---"]
            lineas += [f"{k}: {v}" for k, v in demo.items() if k not in ("reasoning", "rationale", "augmented")]
            bloques.append("\n".join(lineas))
        return "\n\nEJEMPLOS DE REFERENCIA:\n" + "\n\n".join(bloques) + "\n"

    def descripcion(self, clave: str) -> Dict[str, Any]:
        """Qué prompt está activo, para registrarlo en las trazas."""
        if not self.instruccion(clave):
            return {"fuente": "default"}
        return {
            "fuente": "optimizado",
            "archivo": os.path.basename(self.ruta),
            "sha256": self.sha256[:12],
            "optimizador": self.metadata.get("optimizer"),
            "creado": self.metadata.get("created_at"),
        }
