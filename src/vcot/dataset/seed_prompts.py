"""Prompts semilla para generar el dataset V-CoT (IDEA.md §4.3).

Diseño (mentalidad de researcher, optimizado para Qwen3-8B como planner)
-----------------------------------------------------------------------

El dataset captura *decisiones*, no imágenes. Por eso los prompts semilla siguen
dos principios deliberados:

1. **Sub-especificación en las dimensiones que el pipeline debe decidir.** Cada
   prompt da intención, sujeto, entorno y, a lo sumo, un estado de ánimo — pero
   NO dicta layout, composición, iluminación, materiales ni color. Esas son
   justamente las salidas N2–N6 que queremos aprender. Si las pre-cocináramos en
   el prompt, contaminaríamos la señal de decisión (el modelo copiaría en vez de
   decidir). Son *briefs creativos*, no fichas de dirección de arte.

2. **Diversidad estratificada** sobre los ejes que *disparan* esas decisiones:
   género, nº de sujetos (íntimo ↔ multitud), escala (objeto ↔ épico), época,
   hora/atmósfera y registro emocional. Maximiza la cobertura del espacio de
   decisiones con pocas muestras (clave en Smoke/Pilot).

Por qué encaja con Qwen3-8B: es un instruct-follower fuerte con amplio
conocimiento del mundo; prompts breves y evocadores le bastan para producir
planes ricos y coherentes. Mantenerlos cortos también acota el coste de tokens
y evita que el modelo se limite a parafrasear una descripción larga nuestra.

Estratos (3 prompts cada uno): retrato/personaje, paisaje/naturaleza,
arquitectura/interior, bodegón, ciencia-ficción, fantasía/surrealismo,
histórico, calle/documental, acción/movimiento, criatura/animal, comida y
abstracto/conceptual. Para escalas mayores se expande con un generador o un
corpus externo manteniendo esta estratificación.
"""

from __future__ import annotations

from typing import List

SEED_PROMPTS: List[str] = [
    # — retrato / personaje —
    "a weathered lighthouse keeper at the end of a long night shift",
    "a teenage violinist in the seconds before her first solo performance",
    "an off-duty surgeon sitting alone in a hospital stairwell",
    # — paisaje / naturaleza —
    "a solitary oak on a windswept moor as a storm rolls in",
    "a hidden glacial lake walled in by sheer granite peaks",
    "a field of sunflowers during a total solar eclipse",
    # — arquitectura / interior —
    "the reading room of a forgotten national library",
    "a skyscraper abandoned half-finished, open to the sky",
    "a cramped watchmaker's workshop above a busy street",
    # — bodegón / objetos —
    "the contents of a traveler's pockets emptied onto a hotel bed",
    "a banquet table the morning after the celebration",
    "a single origami crane on a windowsill in the rain",
    # — ciencia-ficción —
    "the last surviving greenhouse aboard a generation ship",
    "a roadside diner on a freshly terraformed moon",
    "a lone technician repairing a vast orbital solar array",
    # — fantasía / surrealismo —
    "a library where the books grow on the walls like ivy",
    "a whale migrating through a sky scattered with floating islands",
    "a single ornate door standing alone in an empty desert",
    # — histórico / de época —
    "a telegraph office during a thunderstorm in 1890",
    "a Roman bathhouse in the grey light just before dawn",
    "a jazz club in 1927 as the final set winds down",
    # — calle / documental —
    "a flower vendor closing her stall in heavy monsoon rain",
    "commuters waiting on a platform during a winter blackout",
    "a chess game between two strangers in a crowded park",
    # — acción / movimiento —
    "a cyclist cresting a mountain pass swallowed by fog",
    "a potter throwing clay as the wheel spins fast",
    "a kitesurfer launching off the crest of a wave at golden hour",
    # — criatura / animal —
    "an arctic fox hunting beneath the northern lights",
    "a swarm of fireflies drifting over a still pond at dusk",
    "an old elephant resting in the shade of a lone baobab",
    # — comida / culinario —
    "a street vendor assembling tacos under a tangle of string lights",
    "fresh bread cooling in a village bakery at first light",
    "a quiet tea ceremony in a bare tatami room",
    # — abstracto / conceptual —
    "nostalgia rendered as a physical place you could walk into",
    "the sensation of silence visualized as a landscape",
    "the exact moment a memory begins to fade",
]
