"""Allineamento post-hoc fra slot creduti e individui reali.

Serve a risolvere un problema di valutazione che il progetto non aveva mai
affrontato: **le risposte d'oro parlano di `mela_3`, il sistema conosce solo
"quella mela lì"**. Senza un allineamento non si può nemmeno dire se ha
sbagliato.

È lo stesso mestiere della valutazione del multi-object tracking (IDF1 &
compagnia): si conta quante rilevazioni di ciascun individuo reale sono
finite in ciascuno slot, e si sceglie l'accoppiamento migliore. L'accordo
residuo è a sua volta la misura di binding (sonda P2).

Accoppiamento greedy sui conteggi decrescenti: sub-ottimo rispetto
all'ungherese, ma con pochi slot per classe la differenza non si vede e
resta stdlib puro. Se un giorno servisse, si sostituisce solo qui.
"""
from __future__ import annotations

from collections import Counter

from .tracker import TrackerRegole


def conteggi(tracker: TrackerRegole) -> Counter[tuple[int, str]]:
    return Counter(tracker._assorbite)


def allinea_visione(tracker: TrackerRegole) -> dict[str, int]:
    """id d'istanza reale -> slot che lo rappresenta meglio (dal canale visivo)."""
    coppie = conteggi(tracker)
    slot_presi: set[int] = set()
    veri_presi: set[str] = set()
    mappa: dict[str, int] = {}
    for (slot_id, vero), _ in sorted(coppie.items(), key=lambda kv: (-kv[1], kv[0])):
        if slot_id in slot_presi or vero in veri_presi:
            continue
        mappa[vero] = slot_id
        slot_presi.add(slot_id)
        veri_presi.add(vero)
    return mappa


def allinea_lingua(tracker: TrackerRegole) -> dict[str, int]:
    """id d'istanza reale -> slot, dal canale linguistico (nomi univoci)."""
    return dict(tracker._per_lingua)


def allinea(tracker: TrackerRegole) -> dict[str, int]:
    """Allineamento complessivo: la lingua nomina, quindi vince quando c'è."""
    mappa = allinea_visione(tracker)
    mappa.update(allinea_lingua(tracker))
    return mappa


def purezza_binding(tracker: TrackerRegole) -> tuple[float, int]:
    """Frazione di rilevazioni finite nello slot giusto, e loro numero.

    1.0 = ogni rilevazione di `mela_3` è finita nello slot che rappresenta
    `mela_3`. È la sonda P2 nella sua forma più diretta.
    """
    mappa = allinea_visione(tracker)
    coppie = conteggi(tracker)
    totale = sum(coppie.values())
    if totale == 0:
        return 0.0, 0
    giuste = sum(n for (slot_id, vero), n in coppie.items() if mappa.get(vero) == slot_id)
    return giuste / totale, totale
