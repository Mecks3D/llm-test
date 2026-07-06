"""Rendering: grafo -> frase (FASE1_PIANO.md §2, §4-6).

Gli stampi (unica fonte della grammatica) vivono in `stampi.py`; questo
modulo si limita a ricostruire l'`Evento`/i campi dal grafo, chiamare lo
stampo giusto e mutare il contesto di discorso.
"""
from __future__ import annotations

from typing import Sequence

from mondo.grafo import Grafo
from mondo.tipi import Evento

from . import stampi
from .contesto import StatoDiscorso


def grafo_a_evento(grafo: Grafo) -> Evento:
    """Inverso di `mondo.grafo.evento_a_grafo`: legge gli archi per
    relazione, non per posizione (FASE1_PIANO.md §2)."""
    radice = grafo.nodi[0]
    per_relazione: dict[str, str] = {}
    for arco in grafo.archi:
        per_relazione[arco.relazione] = grafo.nodi[arco.dipendente].lemma
    return Evento(
        t=int(per_relazione["obl:tempo"]),
        azione=radice.lemma,
        agente=per_relazione["nsubj"],
        oggetto=per_relazione.get("obj"),
        destinatario=per_relazione.get("iobj"),
        luogo=per_relazione.get("obl:luogo"),
        luogo_origine=per_relazione.get("obl:origine"),
        argomento=per_relazione.get("obl:argomento"),
    )


def verbalizza_evento(grafo: Grafo, contesto: StatoDiscorso) -> str:
    evento = grafo_a_evento(grafo)
    prefisso = stampi.prefisso_tempo(evento, contesto)
    corpo = stampi.rendi_evento_corpo(evento, contesto)
    contesto.registra_evento(evento)
    return f"{prefisso}{corpo}."


def verbalizza_storia(grafi: Sequence[Grafo], contesto: StatoDiscorso | None = None) -> list[str]:
    if contesto is None:
        contesto = StatoDiscorso()
    return [verbalizza_evento(grafo, contesto) for grafo in grafi]


def verbalizza_domanda(grafo: Grafo, contesto: StatoDiscorso) -> str:
    return stampi.rendi_domanda(grafo, contesto)


def verbalizza_risposta(grafo: Grafo, contesto: StatoDiscorso) -> str:
    return stampi.rendi_risposta(grafo, contesto)
