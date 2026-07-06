"""Punto di ingresso per generare il dataset: assembla storia + domande in
un record JSONL e gestisce lo split train/eval per seed.

Split per seed (CLAUDE.md, regola non negoziabile #3: "mai addestrare su
seed riservati agli esami"): i seed >= SEED_ESAME_MINIMO sono riservati agli
esami e non vanno mai usati per generare dati di training.
"""
from __future__ import annotations

import json
import random
from typing import Iterable, Iterator

from .domande import genera_domande
from .simulatore import genera_storia

N_TICK_MINIMO = 8
N_TICK_MASSIMO = 22
N_PER_TIPO_DEFAULT = 8

# Confine dello split train/eval: i seed di training vivono sotto questa
# soglia, quelli d'esame sopra. Tenerli in intervalli disgiunti rende lo
# split "mai addestrare sui seed d'esame" un controllo su un numero, non
# una convenzione da ricordare a mano.
SEED_ESAME_MINIMO = 1_000_000


def e_seed_esame(seed: int) -> bool:
    return seed >= SEED_ESAME_MINIMO


def _lunghezza_storia(seed: int) -> int:
    """Lunghezza (in tick) della storia, funzione deterministica del seed
    ma su un flusso RNG indipendente da quello usato dalla politica dei
    personaggi in genera_storia."""
    rng = random.Random(f"lunghezza-{seed}")
    return rng.randint(N_TICK_MINIMO, N_TICK_MASSIMO)


def genera_record(seed: int, n_per_tipo: int = N_PER_TIPO_DEFAULT) -> dict:
    n_tick = _lunghezza_storia(seed)
    storia = genera_storia(seed=seed, n_tick=n_tick)
    rng_domande = random.Random(f"domande-{seed}")
    domande = genera_domande(storia, rng_domande, n_per_tipo=n_per_tipo)
    return {
        "seed": seed,
        "eventi": [e.to_dict() for e in storia.eventi],
        "domande": [d.to_dict() for d in domande],
    }


def genera_record_multipli(seeds: Iterable[int], n_per_tipo: int = N_PER_TIPO_DEFAULT) -> Iterator[dict]:
    for seed in seeds:
        yield genera_record(seed, n_per_tipo=n_per_tipo)


def scrivi_dataset(seeds: Iterable[int], percorso: str, n_per_tipo: int = N_PER_TIPO_DEFAULT) -> int:
    n_scritti = 0
    with open(percorso, "w", encoding="utf-8") as f:
        for record in genera_record_multipli(seeds, n_per_tipo=n_per_tipo):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_scritti += 1
    return n_scritti
