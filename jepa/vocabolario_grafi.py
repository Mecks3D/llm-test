"""Estrattore e mappatore deterministico dal micro-mondo a entità e luoghi per il Graph-JEPA.

Converte le frasi UD (grafi) o gli eventi del simulatore in indici per i tensori
del modello JEPA.
"""
from __future__ import annotations

from typing import Any
from mondo import dati_mondo as dm

# Elenco fisso e deterministico di tutte le entità e luoghi del micro-mondo
PERSONE_ID: tuple[str, ...] = tuple(p.id for p in dm.PERSONE)
OGGETTI_ID: tuple[str, ...] = tuple(o.lemma for o in dm.OGGETTI_UNICI) + tuple(
    v["lemma_unita"] for v in dm.RISORSE.values()
)
LUOGHI_ID: tuple[str, ...] = tuple(l.id for l in dm.LUOGHI)

# Entità totali (persone + oggetti)
ENTITA_ID: tuple[str, ...] = PERSONE_ID + OGGETTI_ID

# Mappe da stringa a indice
ENTITA2ID: dict[str, int] = {ent: i for i, ent in enumerate(ENTITA_ID)}
ID2ENTITA: dict[int, str] = {i: ent for i, ent in enumerate(ENTITA_ID)}

LUOGO2ID: dict[str, int] = {luogo: i for i, luogo in enumerate(LUOGHI_ID)}
ID2LUOGO: dict[int, str] = {i: luogo for i, luogo in enumerate(LUOGHI_ID)}

N_ENTITA: int = len(ENTITA_ID)
N_LUOGHI: int = len(LUOGHI_ID)
N_PERSONE: int = len(PERSONE_ID)
N_OGGETTI: int = len(OGGETTI_ID)

# Azioni supportate dal trasformatore di stato
AZIONI_SUPPORTATE: tuple[str, ...] = (
    "andare",
    "prendere",
    "posare",
    "raccogliere",
    "mettere",
    "estrarre",
    "mangiare",
)
AZIONE2ID: dict[str, int] = {az: i for i, az in enumerate(AZIONI_SUPPORTATE)}


def estrai_evento_da_grafo(g: Any) -> dict[str, str | None]:
    """Estrae l'evento strutturato da un grafo UD emesso dal simulatore o lingua.
    
    Esempio grafo evento:
    ( andare ( nsubj sara ) ( obl:origine cucina ) ( obl:luogo giardino ) ( obl:tempo nove ) )
    """
    radice_lemma = g.nodi[0].lemma
    soggetto = None
    luogo_orig = None
    luogo_dest = None
    oggetto = None

    for arco in g.archi:
        if arco.testa == 0:
            rel = arco.relazione
            lemma = g.nodi[arco.dipendente].lemma.lower()
            if rel == "nsubj":
                soggetto = lemma
            elif rel in ("obl:luogo", "obl:destinazione"):
                luogo_dest = lemma
            elif rel == "obl:origine":
                luogo_orig = lemma
            elif rel in ("obj", "obl:oggetto"):
                oggetto = lemma

    return {
        "azione": radice_lemma,
        "soggetto": soggetto,
        "oggetto": oggetto,
        "origine": luogo_orig,
        "destinazione": luogo_dest,
    }

