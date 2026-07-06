"""Tipi di dato di base del simulatore: entità e stato del mondo.

Nessuna logica di simulazione qui dentro: solo strutture dati e le poche
funzioni di lettura sullo stato (risoluzione di posizione, testimoni) che
azioni e generatore di domande condividono.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Entità statiche (definite in dati_mondo.py, non cambiano durante la storia)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Luogo:
    id: str
    lemma: str


@dataclass(frozen=True)
class Persona:
    id: str
    lemma: str
    genere: str  # "m" | "f"
    eta: str  # "bambino" | "adulto" | "anziano"
    luogo_preferito: Optional[str] = None  # id luogo, per il bias della politica


@dataclass(frozen=True)
class TipoOggetto:
    """Definisce una CLASSE di oggetto (es. "mela"), non un'istanza.

    Le istanze concrete (mela_1, mela_2, ...) sono generate a runtime quando
    un personaggio prende un'unità da una risorsa finita (melo, pozzo, bosco).
    Gli oggetti unici (palla, libro, cestino, scatola, secchio, pane) esistono
    invece fin dall'inizio come singola istanza con id == lemma.
    """
    lemma: str
    commestibile: bool = False
    contenitore: bool = False
    apribile: bool = False  # solo se contenitore
    risorsa: bool = False  # generato da una fonte finita (melo/pozzo/bosco)
    fisso: bool = False  # arredo: non si può prendere/spostare (es. camino)


# ---------------------------------------------------------------------------
# Stato dinamico
# ---------------------------------------------------------------------------

@dataclass
class StatoOggetto:
    id: str
    lemma: str
    commestibile: bool = False
    contenitore: bool = False
    apribile: bool = False
    fisso: bool = False
    aperto: bool = True  # rilevante solo se contenitore e apribile
    # "dove" dell'oggetto: uno tra
    #   ("luogo", luogo_id)      -> appoggiato in un luogo
    #   ("persona", persona_id)  -> portato/in mano a una persona
    #   ("contenitore", oggetto_id) -> dentro un contenitore
    posizione: tuple[str, str] = field(default=("luogo", ""))
    proprietario: Optional[str] = None  # possesso statico (di chi è), non chi lo porta ora


@dataclass
class StatoPersona:
    id: str
    lemma: str
    genere: str
    eta: str
    luogo_preferito: Optional[str]
    luogo: str
    fame: int = 0
    stanchezza: int = 0
    addormentato: bool = False


@dataclass
class StatoLuogo:
    id: str
    lemma: str
    calore: int = 0  # rilevante solo per il salotto (camino)


@dataclass
class Evento:
    t: int
    azione: str
    agente: str
    oggetto: Optional[str] = None
    destinatario: Optional[str] = None
    luogo: Optional[str] = None
    luogo_origine: Optional[str] = None
    argomento: Optional[str] = None  # extra libero (es. contenitore in mettere_dentro)
    testimoni: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = {
            "t": self.t,
            "azione": self.azione,
            "agente": self.agente,
        }
        if self.oggetto is not None:
            d["oggetto"] = self.oggetto
        if self.destinatario is not None:
            d["destinatario"] = self.destinatario
        if self.luogo is not None:
            d["luogo"] = self.luogo
        if self.luogo_origine is not None:
            d["luogo_origine"] = self.luogo_origine
        if self.argomento is not None:
            d["argomento"] = self.argomento
        d["testimoni"] = list(self.testimoni)
        return d


@dataclass
class StatoMondo:
    t: int
    luoghi: dict[str, StatoLuogo]
    collegamenti: dict[str, frozenset[str]]  # luogo_id -> vicini
    persone: dict[str, StatoPersona]
    oggetti: dict[str, StatoOggetto]
    risorse: dict[str, int]  # "melo" -> mele rimaste, "pozzo" -> acqua, "bosco" -> legna
    prossimo_id_istanza: dict[str, int] = field(default_factory=dict)

    # -- letture sullo stato, usate da azioni.py, motore.py, domande.py -----

    def luogo_effettivo(self, entita_id: str) -> str:
        """Risale la catena di contenimento fino a un luogo fisico."""
        if entita_id in self.persone:
            return self.persone[entita_id].luogo
        ogg = self.oggetti[entita_id]
        tipo, riferimento = ogg.posizione
        if tipo == "luogo":
            return riferimento
        if tipo == "persona":
            return self.persone[riferimento].luogo
        if tipo == "contenitore":
            return self.luogo_effettivo(riferimento)
        raise ValueError(f"posizione sconosciuta per {entita_id}: {ogg.posizione}")

    def testimoni_in(self, luogo_id: str) -> tuple[str, ...]:
        presenti = [p.id for p in self.persone.values() if p.luogo == luogo_id]
        return tuple(sorted(presenti))

    def oggetti_in_luogo(self, luogo_id: str) -> list[str]:
        return [
            oid for oid, o in self.oggetti.items()
            if o.posizione[0] == "luogo" and o.posizione[1] == luogo_id
        ]

    def oggetti_portati_da(self, persona_id: str) -> list[str]:
        return [
            oid for oid, o in self.oggetti.items()
            if o.posizione[0] == "persona" and o.posizione[1] == persona_id
        ]

    def oggetti_dentro(self, contenitore_id: str) -> list[str]:
        return [
            oid for oid, o in self.oggetti.items()
            if o.posizione[0] == "contenitore" and o.posizione[1] == contenitore_id
        ]

    def nuovo_id(self, base: str) -> str:
        n = self.prossimo_id_istanza.get(base, 0) + 1
        self.prossimo_id_istanza[base] = n
        return f"{base}_{n}"
