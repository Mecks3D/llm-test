"""Relazioni di parentela derivate dai fatti di base in dati_mondo.py.

Le relazioni qui sono etichette (stringhe), fatti statici del mondo: la
lingua (Fase 1) le tradurrà in frasi. `relazione_di(a, b)` percorre catene di
al più 4 passi (genitore/figlio/coniuge/fratello), stile CLUTRR; oltre quella
profondità la relazione è dichiarata non nota da questo modulo, il che serve
anche a generare domande di parentela senza risposta determinabile.
"""
from __future__ import annotations

from typing import Optional

from . import dati_mondo as dm


def _genere(persona_id: str) -> str:
    for p in dm.PERSONE:
        if p.id == persona_id:
            return p.genere
    raise KeyError(persona_id)


def _costruisci_genitori_e_figli() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    genitori: dict[str, set[str]] = {p.id: set() for p in dm.PERSONE}
    figli: dict[str, set[str]] = {p.id: set() for p in dm.PERSONE}
    for g, f in dm.GENITORE_DI:
        genitori[f].add(g)
        figli[g].add(f)
    return genitori, figli


def _costruisci_coniugi() -> dict[str, str]:
    coniugi: dict[str, str] = {}
    for a, b in dm.CONIUGE_DI:
        coniugi[a] = b
        coniugi[b] = a
    return coniugi


def _costruisci_fratelli(genitori: dict[str, set[str]]) -> dict[str, set[str]]:
    persone = [p.id for p in dm.PERSONE]
    fratelli: dict[str, set[str]] = {pid: set() for pid in persone}
    for a in persone:
        for b in persone:
            if a != b and genitori[a] and genitori[a] == genitori[b]:
                fratelli[a].add(b)
    return fratelli


_GENITORI, _FIGLI = _costruisci_genitori_e_figli()
_CONIUGI = _costruisci_coniugi()
_FRATELLI = _costruisci_fratelli(_GENITORI)


def relazione_di(a: str, b: str) -> Optional[str]:
    """Ritorna l'etichetta della relazione "a è ___ di b", o None se non
    ricostruibile in <= 4 passi dalle regole coperte."""
    if a == b:
        return None

    # 1 passo
    if a in _GENITORI.get(b, ()):
        return "padre_di" if _genere(a) == "m" else "madre_di"
    if b in _GENITORI.get(a, ()):
        return "figlio_di" if _genere(a) == "m" else "figlia_di"
    if _CONIUGI.get(a) == b:
        return "marito_di" if _genere(a) == "m" else "moglie_di"
    if b in _FRATELLI.get(a, ()):
        return "fratello_di" if _genere(a) == "m" else "sorella_di"

    # 2 passi: nonno/a (genitore, genitore) e nipote-di-nonno (figlio, figlio)
    for x in _FIGLI.get(a, ()):
        if b in _FIGLI.get(x, ()):
            return "nonno_di" if _genere(a) == "m" else "nonna_di"
    for x in _GENITORI.get(a, ()):
        if b in _GENITORI.get(x, ()):
            return "nipote_di"

    # 2 passi: suocero/a (genitore, coniuge) e genero/nuora (coniuge, figlio)
    for x in _FIGLI.get(a, ()):
        if _CONIUGI.get(x) == b:
            return "suocero_di" if _genere(a) == "m" else "suocera_di"
    x = _CONIUGI.get(a)
    if x is not None and b in _GENITORI.get(x, ()):
        return "genero_di" if _genere(a) == "m" else "nuora_di"

    # 3 passi: zio/a (fratello, figlio) e il suo inverso (figlio, fratello)
    for x in _FRATELLI.get(a, ()):
        if b in _FIGLI.get(x, ()):
            return "zio_di" if _genere(a) == "m" else "zia_di"
    for x in _GENITORI.get(a, ()):
        if b in _FRATELLI.get(x, ()):
            return "nipote_di"

    # 4 passi: cugino/a (genitore, fratello, figlio)
    for x in _GENITORI.get(a, ()):
        for y in _FRATELLI.get(x, ()):
            if b in _FIGLI.get(y, ()):
                return "cugino_di" if _genere(a) == "m" else "cugina_di"

    return None


def tutte_le_coppie() -> list[tuple[str, str]]:
    persone = [p.id for p in dm.PERSONE]
    return [(a, b) for a in persone for b in persone if a != b]
