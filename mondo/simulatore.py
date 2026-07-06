"""Punto di ingresso per generare una storia: N tick di simulazione a partire
da un seed esplicito. Stesso seed -> stessa storia, byte per byte.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .motore import avanza_tick, costruisci_stato_iniziale
from .tipi import Evento, StatoMondo


@dataclass(frozen=True)
class Storia:
    seed: int
    eventi: tuple[Evento, ...]
    stato_finale: StatoMondo


def genera_storia(seed: int, n_tick: int = 30) -> Storia:
    """Genera una storia deterministica: `seed` istanzia l'UNICO random.Random
    usato per tutte le scelte della politica dei personaggi (mai `random`
    globale)."""
    rng = random.Random(seed)
    stato = costruisci_stato_iniziale()
    eventi: list[Evento] = []
    for t in range(1, n_tick + 1):
        eventi.extend(avanza_tick(stato, rng, t))
    return Storia(seed=seed, eventi=tuple(eventi), stato_finale=stato)


def storia_a_dict(storia: Storia) -> dict:
    return {
        "seed": storia.seed,
        "eventi": [e.to_dict() for e in storia.eventi],
    }
