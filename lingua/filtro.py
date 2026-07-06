"""Filtro simbolico sui grafi (FASE1_PIANO.md §9): regole vietate ai bordi,
applicate a input e output. Segnaposto architetturale per la Fase 1
(FASE1.md): la lista resta minima e non si espande in questa fase — conta
la sede, non il contenuto.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mondo.grafo import NON_LO_SO, Grafo

from .contesto import estrai_lemma_istanza

_PERCORSO_DEFAULT = Path(__file__).with_name("regole_filtro.txt")


@dataclass(frozen=True)
class RegolaFiltro:
    radice: str
    relazioni: tuple[tuple[str, str], ...]  # (relazione, lemma) da esaurire tutte


@dataclass(frozen=True)
class RisultatoFiltro:
    ammesso: bool
    regola_violata: str | None = None


def _lemma_ai_fini_del_filtro(grafo: Grafo, id_nodo: int) -> str:
    lemma = grafo.nodi[id_nodo].lemma
    istanza = estrai_lemma_istanza(lemma)
    return istanza[0] if istanza is not None else lemma


def _analizza_riga(riga: str) -> RegolaFiltro:
    pezzi = riga.split()
    if not pezzi or not pezzi[0].startswith("radice="):
        raise ValueError(f"riga di filtro mal formata (attesa 'radice=...' per prima): {riga!r}")
    radice = pezzi[0][len("radice="):]
    relazioni: list[tuple[str, str]] = []
    for pezzo in pezzi[1:]:
        relazione, sep, lemma = pezzo.partition("=")
        if not sep:
            raise ValueError(f"riga di filtro mal formata: {riga!r}")
        relazioni.append((relazione, lemma))
    return RegolaFiltro(radice=radice, relazioni=tuple(relazioni))


def _carica_regole(percorso: str | Path) -> tuple[RegolaFiltro, ...]:
    regole = []
    with open(percorso, encoding="utf-8") as f:
        for riga in f:
            riga = riga.strip()
            if not riga or riga.startswith("#"):
                continue
            regole.append(_analizza_riga(riga))
    return tuple(regole)


_REGOLE: tuple[RegolaFiltro, ...] = _carica_regole(_PERCORSO_DEFAULT)


def _regola_scatta(grafo: Grafo, regola: RegolaFiltro) -> bool:
    if grafo.nodi[0].lemma != regola.radice:
        return False
    archi_radice = {
        arco.relazione: _lemma_ai_fini_del_filtro(grafo, arco.dipendente)
        for arco in grafo.archi if arco.testa == 0
    }
    return all(archi_radice.get(relazione) == lemma for relazione, lemma in regola.relazioni)


def _descrivi(regola: RegolaFiltro) -> str:
    pezzi = [f"radice={regola.radice}"] + [f"{relazione}={lemma}" for relazione, lemma in regola.relazioni]
    return " ".join(pezzi)


def filtra(grafo: Grafo) -> RisultatoFiltro:
    if grafo == NON_LO_SO:
        return RisultatoFiltro(ammesso=True)
    for regola in _REGOLE:
        if _regola_scatta(grafo, regola):
            return RisultatoFiltro(ammesso=False, regola_violata=_descrivi(regola))
    return RisultatoFiltro(ammesso=True)
