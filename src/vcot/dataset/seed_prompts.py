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

Dos niveles de uso
------------------

- :data:`SEED_PROMPTS` — el **núcleo curado** (36 briefs, 12 estratos × 3),
  calidad garantizada. Es lo que se usa por defecto (sin ``--limit``).
- :func:`generate_prompts` — **expansión estratificada determinista** a cualquier
  ``n`` para Smoke/Pilot. Combina ``subject × context`` por estrato (manteniendo
  la sub-especificación y el balance entre estratos) y antepone siempre el núcleo
  curado. Determinista dado ``seed`` (dataset reproducible). El generador cubre
  cómodamente Smoke (100) y Pilot (low-thousands); para 10k+ se sustituye/expande
  con un corpus externo manteniendo esta misma estratificación.

Estratos (núcleo, 3 prompts cada uno): retrato/personaje, paisaje/naturaleza,
arquitectura/interior, bodegón, ciencia-ficción, fantasía/surrealismo,
histórico, calle/documental, acción/movimiento, criatura/animal, comida y
abstracto/conceptual.
"""

from __future__ import annotations

import random
from typing import Dict, List

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


# --------------------------------------------------------------------------- #
# Expansión estratificada (Smoke/Pilot)
# --------------------------------------------------------------------------- #
#
# Cada estrato declara `subjects` (el actor/elemento — autocontenido) y
# `contexts` (la *condición* situacional: hora, clima, evento, lugar). El brief
# se compone como ``"{subject} {context}"``. Los contexts son deliberadamente
# situacionales (no actividades subject-específicas) para que cualquier
# subject×context del mismo estrato lea con naturalidad y mantenga la
# sub-especificación: describen *cuándo/dónde*, nunca *cómo* iluminar/componer.

STRATA: Dict[str, Dict[str, List[str]]] = {
    "portrait": {
        "subjects": [
            "a weathered lighthouse keeper",
            "a teenage violinist",
            "an off-duty surgeon",
            "a retired sailor with a faded tattoo",
            "a night-shift baker dusted with flour",
            "a war correspondent back from the field",
            "a street fortune-teller between customers",
        ],
        "contexts": [
            "at the end of a long night shift",
            "in the quiet minute before everything changes",
            "during a citywide power outage",
            "lost in thought by a rain-streaked window",
            "caught off guard by an old memory",
            "alone in a room full of strangers",
        ],
    },
    "landscape": {
        "subjects": [
            "a solitary oak on a windswept moor",
            "a hidden glacial lake",
            "a field of sunflowers",
            "a salt flat stretching to the horizon",
            "a terraced rice valley",
            "a basalt coastline battered by surf",
            "an ancient forest of redwoods",
        ],
        "contexts": [
            "as a storm rolls in",
            "during a total solar eclipse",
            "in the last light before nightfall",
            "under the first snowfall of the year",
            "wrapped in low morning fog",
            "beneath a sky of unusual color",
        ],
    },
    "architecture": {
        "subjects": [
            "the reading room of a forgotten national library",
            "a skyscraper abandoned half-finished",
            "a cramped watchmaker's workshop",
            "a derelict subway station",
            "a cathedral mid-restoration under scaffolding",
            "an empty indoor swimming hall",
            "a spiral stairwell in an old apartment block",
        ],
        "contexts": [
            "open to the sky",
            "lit only by a single skylight",
            "the morning after it was emptied",
            "in the silence of a public holiday",
            "as dust drifts through stale air",
            "long after the last visitor left",
        ],
    },
    "still_life": {
        "subjects": [
            "the contents of a traveler's pockets",
            "a banquet table",
            "a single origami crane",
            "a child's collection of pressed flowers",
            "the tools of a watchmaker laid out",
            "a half-finished letter and a cold cup of tea",
            "a bowl of fruit just past its prime",
        ],
        "contexts": [
            "emptied onto a hotel bed",
            "the morning after the celebration",
            "on a windowsill in the rain",
            "in a shaft of late afternoon light",
            "abandoned in a hurry",
            "arranged with quiet care",
        ],
    },
    "scifi": {
        "subjects": [
            "the last surviving greenhouse aboard a generation ship",
            "a roadside diner on a freshly terraformed moon",
            "a lone technician repairing an orbital solar array",
            "a derelict probe drifting past a gas giant",
            "a crowded transit hub on a tidally locked world",
            "a deep-sea research dome on an ice moon",
            "a courier waiting at a wormhole checkpoint",
        ],
        "contexts": [
            "during a long communications blackout",
            "as the artificial dawn cycle begins",
            "with the home star barely a pinprick",
            "the night before the colony votes to leave",
            "while the life-support hums in the dark",
            "as a dust storm closes the horizon",
        ],
    },
    "fantasy": {
        "subjects": [
            "a library where the books grow like ivy",
            "a whale migrating through floating islands",
            "a single ornate door standing in a desert",
            "a clockwork garden tended by no one",
            "a city built inside a hollow mountain",
            "a bridge woven from living roots",
            "a lantern-lit market beneath a frozen sea",
        ],
        "contexts": [
            "in an empty desert",
            "as the second moon rises",
            "the moment the spell begins to fade",
            "long after its makers vanished",
            "in a season that never quite arrives",
            "while everything holds its breath",
        ],
    },
    "historical": {
        "subjects": [
            "a telegraph office",
            "a Roman bathhouse",
            "a jazz club",
            "a frontier trading post",
            "a medieval scriptorium",
            "a Victorian operating theatre",
            "a dockside customs house",
        ],
        "contexts": [
            "during a thunderstorm in 1890",
            "in the grey light just before dawn",
            "as the final set winds down",
            "the day news of the war arrived",
            "by the light of failing candles",
            "an hour before the crowds return",
        ],
    },
    "street": {
        "subjects": [
            "a flower vendor closing her stall",
            "commuters waiting on a platform",
            "a chess game between two strangers",
            "a busker packing up his guitar",
            "a fruit market at the end of the day",
            "a crossing guard at a quiet intersection",
            "a line outside a soup kitchen",
        ],
        "contexts": [
            "in heavy monsoon rain",
            "during a winter blackout",
            "in a crowded park",
            "under flickering neon",
            "as the first commuters appear",
            "while the snow begins to settle",
        ],
    },
    "action": {
        "subjects": [
            "a cyclist cresting a mountain pass",
            "a potter throwing clay",
            "a kitesurfer launching off a wave",
            "a blacksmith mid-strike",
            "a free runner clearing a rooftop gap",
            "a fishing crew hauling a full net",
            "a dancer in the peak of a leap",
        ],
        "contexts": [
            "swallowed by fog",
            "as the wheel spins fast",
            "at golden hour",
            "with sparks flying",
            "against a gathering storm",
            "in the moment of no return",
        ],
    },
    "creature": {
        "subjects": [
            "an arctic fox hunting",
            "a swarm of fireflies",
            "an old elephant",
            "a heron stalking the shallows",
            "a pack of wolves crossing a frozen river",
            "a sea turtle gliding over a reef",
            "a barn owl on a fencepost",
        ],
        "contexts": [
            "beneath the northern lights",
            "drifting over a still pond at dusk",
            "in the shade of a lone baobab",
            "in the hush before the rain",
            "under a brightening moon",
            "as the first frost forms",
        ],
    },
    "food": {
        "subjects": [
            "a street vendor assembling tacos",
            "fresh bread cooling",
            "a quiet tea ceremony",
            "a noodle stall in a night market",
            "a grandmother rolling dumplings",
            "a wood-fired oven at full heat",
            "a market stand piled with citrus",
        ],
        "contexts": [
            "under a tangle of string lights",
            "in a village bakery at first light",
            "in a bare tatami room",
            "as steam fogs the windows",
            "just before the dinner rush",
            "in the last warm light of the day",
        ],
    },
    "abstract": {
        "subjects": [
            "nostalgia",
            "the sensation of silence",
            "the weight of an unspoken apology",
            "the feeling of arriving too late",
            "the space between two heartbeats",
            "the moment doubt becomes certainty",
            "the comfort of a familiar routine",
        ],
        "contexts": [
            "rendered as a physical place you could walk into",
            "visualized as a landscape",
            "imagined as a single room",
            "drawn as weather over a city",
            "shaped into an object you could hold",
            "mapped as a path through fog",
        ],
    },
}

#: Orden de los estratos para el round-robin (mantiene el balance entre géneros).
GENRE_ORDER: List[str] = list(STRATA)


def generate_prompts(n: int, *, seed: int = 0, include_curated: bool = True) -> List[str]:
    """Devuelve ``n`` briefs estratificados y únicos, determinista dado ``seed``.

    Antepone el núcleo curado (:data:`SEED_PROMPTS`) y completa con combinaciones
    ``subject × context`` por estrato, repartidas en *round-robin* para que los
    géneros queden balanceados. Mismo ``seed`` ⇒ misma lista (dataset
    reproducible). Lanza ``ValueError`` si ``n`` excede el espacio disponible.

    Parameters
    ----------
    n:
        Número de prompts a devolver.
    seed:
        Semilla del barajado (reproducibilidad).
    include_curated:
        Si ``True`` (por defecto), los primeros prompts son el núcleo curado.
    """
    if n <= 0:
        return []

    out: List[str] = []
    seen: set[str] = set()

    def add(prompt: str) -> None:
        key = prompt.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(prompt)

    if include_curated:
        for p in SEED_PROMPTS:
            add(p)
            if len(out) >= n:
                return out[:n]

    rng = random.Random(seed)

    # Pools base: todas las combinaciones subject×context por estrato, barajadas.
    base_pools: List[List[str]] = []
    for genre in GENRE_ORDER:
        s = STRATA[genre]
        combos = [f"{subj} {ctx}" for subj in s["subjects"] for ctx in s["contexts"]]
        rng.shuffle(combos)
        base_pools.append(combos)

    if _round_robin_fill(out, add, base_pools, n):
        return out[:n]

    # Si el núcleo se agota, una segunda capa añade un matiz atmosférico a las
    # mismas combinaciones (amplía el rango sin perder la estratificación).
    atmo_pools: List[List[str]] = []
    for combos in base_pools:
        variants = [f"{c}, {a}" for c in combos for a in ATMOSPHERES]
        rng.shuffle(variants)
        atmo_pools.append(variants)
    _round_robin_fill(out, add, atmo_pools, n)

    if len(out) < n:
        raise ValueError(
            f"generate_prompts: pedidos {n} pero el espacio estratificado solo "
            f"produce {len(out)} únicos. Para esta escala usá un corpus externo "
            "(ver docstring del módulo)."
        )
    return out[:n]


def _round_robin_fill(out: List[str], add, pools: List[List[str]], n: int) -> bool:
    """Reparte de a un prompt por estrato y ciclo hasta llegar a ``n`` o agotarse.

    Devuelve ``True`` si se alcanzó ``n``.
    """
    pointers = [0] * len(pools)
    while len(out) < n and any(pointers[i] < len(pools[i]) for i in range(len(pools))):
        for i, pool in enumerate(pools):
            if len(out) >= n:
                return True
            if pointers[i] < len(pool):
                add(pool[pointers[i]])
                pointers[i] += 1
    return len(out) >= n


#: Matices atmosféricos (situacionales, no de dirección de arte) para la segunda
#: capa de expansión. Describen condición/hora, nunca iluminación/composición.
ATMOSPHERES: List[str] = [
    "at dawn",
    "after midnight",
    "in a sudden downpour",
    "during the first frost",
    "in the heat of late summer",
    "as the fog lifts",
]


# --------------------------------------------------------------------------- #
# Clasificación prompt → estrato (para el breakdown por género en eval/report)
# --------------------------------------------------------------------------- #
#
# El núcleo curado va en bloques de 3 en el mismo orden que GENRE_ORDER; los
# generados son "{subject} {context}" así que se clasifican por prefijo de sujeto.

_CURATED_STRATUM = {
    p: GENRE_ORDER[i // 3] for i, p in enumerate(SEED_PROMPTS) if i // 3 < len(GENRE_ORDER)
}
_SUBJECT_TO_GENRE = {
    subj.lower(): genre for genre, s in STRATA.items() for subj in s["subjects"]
}


def prompt_stratum(prompt: str) -> str:
    """Estrato (género) de un prompt; ``'unknown'`` si no se reconoce.

    Best-effort: exacto para el núcleo curado, por prefijo de sujeto para los
    generados. Sirve para el breakdown por género del eval/informe.
    """
    if prompt in _CURATED_STRATUM:
        return _CURATED_STRATUM[prompt]
    low = prompt.lower()
    # Subjects más largos primero ⇒ evita que un prefijo corto gane sobre uno específico.
    for subj in sorted(_SUBJECT_TO_GENRE, key=len, reverse=True):
        if low.startswith(subj):
            return _SUBJECT_TO_GENRE[subj]
    return "unknown"
