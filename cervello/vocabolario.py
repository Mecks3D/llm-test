"""Vocabolario del cervello: mappa token <-> id interi, generato dal lessico.

Ordine normativo degli id (FASE2_PIANO.md §3), da non alterare senza
rigenerare `vocabolario.json` e aggiornare i test:

1. id 0-64: i 65 lemmi PRIM, nell'ordine delle righe del lessico (sono il
   contratto di PROGETTO.md: i primitivi occupano i token id 0-64).
2. id 65-71: token speciali, in quest'ordine: [PAD] [STORIA] [DOMANDA]
   [RISPOSTA] [FINE] ( )
3. id 72-86: le 15 relazioni UD usate dai grafi, in ordine alfabetico.
4. id 87-...: tutti gli altri lemmi del lessico (FUNZ, ORD, ecc.),
   nell'ordine delle righe del file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lingua.lessico import N_PRIM
from lingua.lessico import _PERCORSO_DEFAULT as _LESSICO_DEFAULT
from lingua.lessico import carica_lessico

_PERCORSO_DEFAULT = Path(__file__).with_name("vocabolario.json")

TOKEN_SPECIALI: tuple[str, ...] = ("[PAD]", "[STORIA]", "[DOMANDA]", "[RISPOSTA]", "[FINE]", "(", ")")

RELAZIONI_UD: tuple[str, ...] = (
    "advcl:causa", "iobj", "nmod:agente", "nmod:destinatario",
    "nmod:oggetto", "nmod:parentela", "nmod:relativo", "nsubj", "obj",
    "obl:argomento", "obl:luogo", "obl:origine", "obl:quantita",
    "obl:tempo", "quesito",
)


class Vocabolario:
    """Mappa bidirezionale token <-> id, immutabile."""

    def __init__(self, token: tuple[str, ...]) -> None:
        self._token = token
        self._id: dict[str, int] = {}
        for i, t in enumerate(token):
            if t in self._id:
                raise ValueError(f"token duplicato nel vocabolario: {t!r}")
            self._id[t] = i

    @property
    def dimensione(self) -> int:
        return len(self._token)

    def id(self, token: str) -> int:
        return self._id[token]

    def token(self, id_: int) -> str:
        return self._token[id_]

    def token_lista(self) -> tuple[str, ...]:
        return self._token

    def __contains__(self, token: str) -> bool:
        return token in self._id


def sha256_lessico(percorso_lessico: str | Path = _LESSICO_DEFAULT) -> str:
    return hashlib.sha256(Path(percorso_lessico).read_bytes()).hexdigest()


def genera_vocabolario(percorso_lessico: str | Path = _LESSICO_DEFAULT) -> Vocabolario:
    """Costruisce il vocabolario dal lessico, nell'ordine normativo."""
    lessico = carica_lessico(percorso_lessico)
    lessico.valida()

    voci = lessico.voci()
    token: list[str] = [v.lemma for v in voci[:N_PRIM]]
    token.extend(TOKEN_SPECIALI)
    token.extend(RELAZIONI_UD)
    token.extend(v.lemma for v in voci[N_PRIM:])

    return Vocabolario(tuple(token))


def salva_vocabolario(
    vocab: Vocabolario,
    percorso_lessico: str | Path = _LESSICO_DEFAULT,
    percorso: str | Path = _PERCORSO_DEFAULT,
) -> None:
    dati = {"token": list(vocab.token_lista()), "versione_lessico": sha256_lessico(percorso_lessico)}
    Path(percorso).write_text(json.dumps(dati, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def carica_vocabolario(
    percorso: str | Path = _PERCORSO_DEFAULT,
    percorso_lessico: str | Path = _LESSICO_DEFAULT,
) -> Vocabolario:
    """Carica il vocabolario committato, verificando che il lessico non sia
    cambiato da quando è stato generato."""
    dati = json.loads(Path(percorso).read_text(encoding="utf-8"))
    sha_atteso = dati["versione_lessico"]
    sha_attuale = sha256_lessico(percorso_lessico)
    if sha_atteso != sha_attuale:
        raise ValueError(
            "il lessico è cambiato dall'ultima generazione del vocabolario "
            f"(atteso sha256 {sha_atteso}, trovato {sha_attuale}): "
            "rigenerare con `python -m cervello.vocabolario`"
        )
    return Vocabolario(tuple(dati["token"]))


if __name__ == "__main__":
    v = genera_vocabolario()
    salva_vocabolario(v)
    print(f"vocabolario rigenerato: {v.dimensione} token -> {_PERCORSO_DEFAULT}")
