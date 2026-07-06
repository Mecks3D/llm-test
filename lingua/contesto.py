"""Il contesto di discorso: ciò che il lettore sa finora (FASE1_PIANO.md §3).

Usato sia da `verbalizza.py` sia da `analizza.py`: entrambi lo mutano dopo
ogni frase-evento con `registra_evento`, così le stesse decisioni (definito
vs indefinito, luogo esplicito o no) restano coerenti nelle due direzioni.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from mondo import dati_mondo as dm
from mondo.tipi import Evento

_PERSONE_IDS: frozenset[str] = frozenset(p.id for p in dm.PERSONE)
_LEMMI_RISORSA: frozenset[str] = frozenset(info["lemma_unita"] for info in dm.RISORSE.values())

_RE_ISTANZA = re.compile(r"^([a-z_]+)_(\d+)$")


def estrai_lemma_istanza(entita_id: str) -> tuple[str, int] | None:
    """("mela_2") -> ("mela", 2); None se `entita_id` non è un'istanza di
    risorsa (oggetti unici e persone non lo sono mai, per costruzione)."""
    m = _RE_ISTANZA.match(entita_id)
    if m is None:
        return None
    lemma, indice = m.group(1), int(m.group(2))
    if lemma not in _LEMMI_RISORSA:
        return None
    return lemma, indice


@dataclass
class StatoDiscorso:
    tick_corrente: int | None = None
    max_indice: dict[str, int] = field(default_factory=dict)
    posizione_persone: dict[str, str] = field(default_factory=dict)

    def registra_evento(self, evento: Evento) -> None:
        self.tick_corrente = evento.t

        if evento.oggetto is not None:
            istanza = estrai_lemma_istanza(evento.oggetto)
            if istanza is not None:
                lemma, indice = istanza
                self.max_indice[lemma] = max(self.max_indice.get(lemma, 0), indice)

        if evento.luogo is not None and evento.azione != "bruciare":
            self.posizione_persone[evento.agente] = evento.luogo

        if evento.azione in ("dare", "dire") and evento.destinatario is not None and evento.luogo is not None:
            self.posizione_persone[evento.destinatario] = evento.luogo

        if evento.azione == "guardare" and evento.oggetto is not None and evento.luogo is not None:
            if evento.oggetto in _PERSONE_IDS:
                self.posizione_persone[evento.oggetto] = evento.luogo
