"""Riconoscimento: frase -> grafo (FASE1_PIANO.md §2, §4-6).

Speculare a `verbalizza.py`: usa gli stessi stampi per ricostruire
l'`Evento`, poi lo passa a `evento_a_grafo` (mai costruire nodi/archi a
mano, così la numerazione coincide per costruzione col mondo)."""
from __future__ import annotations

from typing import Sequence

from mondo.grafo import Grafo, evento_a_grafo

from . import stampi
from .contesto import StatoDiscorso


def analizza_evento(frase: str, contesto: StatoDiscorso) -> Grafo:
    t, corpo_con_punto = stampi.stacca_prefisso_tempo(frase, contesto)
    if not corpo_con_punto.endswith("."):
        raise ValueError(f"frase-evento senza punto finale: {frase!r}")
    corpo = corpo_con_punto[:-1]
    evento = stampi.riconosci_evento_corpo(corpo, contesto, t)
    contesto.registra_evento(evento)
    return evento_a_grafo(evento)


def analizza_storia(frasi: Sequence[str], contesto: StatoDiscorso | None = None) -> list[Grafo]:
    if contesto is None:
        contesto = StatoDiscorso()
    return [analizza_evento(frase, contesto) for frase in frasi]


def analizza_domanda(frase: str, contesto: StatoDiscorso) -> Grafo:
    return stampi.analizza_domanda(frase, contesto)


def analizza_risposta(frase: str, contesto: StatoDiscorso) -> Grafo:
    return stampi.analizza_risposta(frase, contesto)
