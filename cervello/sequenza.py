"""Linearizzazione grafo <-> sequenza di token (FASE2_PIANO.md §4).

Grammatica normativa (un grafo alla volta, un solo livello di annidamento —
è la forma di TUTTI i grafi prodotti da `mondo/grafo.py`):

    grafo := ( lemma_radice ramo* )
    ramo  := ( relazione lemma [ordinale] )

Un nodo con lemma-istanza (`mela_2`) si scompone SEMPRE in due token: il
lemma base e l'ordinale maschile del lessico (`mela secondo`), anche per
N=1. Non c'è contesto di discorso qui: la scomposizione è locale al
singolo grafo (a differenza del verbalizzatore di Fase 1).

Stdlib puro: nessun import di torch (può girare senza la dipendenza).
"""
from __future__ import annotations

from typing import Sequence

from lingua.lessico import Lessico, carica_lessico
from lingua.morfologia import ordinale, ordinale_inverso
from mondo.grafo import NON_LO_SO, Grafo, evento_a_grafo, grafo_fatto
from mondo.numeri import VALORE_A_LEMMA
from mondo.tipi import Evento

from .vocabolario import RELAZIONI_UD, TOKEN_SPECIALI

PAD, STORIA, DOMANDA, RISPOSTA, FINE, APERTA, CHIUSA = TOKEN_SPECIALI

_LEMMA_A_VALORE_NUM: dict[str, int] = {lemma: valore for valore, lemma in VALORE_A_LEMMA.items()}

_lessico_cache: Lessico | None = None


def _ottieni_lessico() -> Lessico:
    global _lessico_cache
    if _lessico_cache is None:
        lex = carica_lessico()
        lex.valida()
        _lessico_cache = lex
    return _lessico_cache


def _nodo_a_token(lemma: str) -> list[str]:
    lex = _ottieni_lessico()
    if lemma in lex:
        return [lemma]

    base, sep, suffisso = lemma.rpartition("_")
    if not sep or not suffisso.isdigit():
        raise ValueError(f"lemma sconosciuto e non è un'istanza lemma_N: {lemma!r}")
    if base not in lex:
        raise ValueError(f"base d'istanza sconosciuta nel lessico: {lemma!r}")
    return [base, ordinale(int(suffisso), "m")]


def grafo_a_token(grafo: Grafo) -> list[str]:
    """Linearizza un `Grafo` (evento, domanda o risposta) in token stringa."""
    if not grafo.nodi:
        raise ValueError("grafo senza nodi")

    nodi_per_id = {n.id: n for n in grafo.nodi}
    radice = grafo.nodi[0]

    token: list[str] = [APERTA]
    token.extend(_nodo_a_token(radice.lemma))
    for arco in grafo.archi:
        if arco.testa != radice.id:
            raise ValueError(f"arco non radicato nella radice del grafo: {arco}")
        dipendente = nodi_per_id[arco.dipendente]
        token.append(APERTA)
        token.append(arco.relazione)
        token.extend(_nodo_a_token(dipendente.lemma))
        token.append(CHIUSA)
    token.append(CHIUSA)
    return token


class _Cursore:
    def __init__(self, token: Sequence[str]) -> None:
        self._token = token
        self._pos = 0

    def prossimo(self) -> str:
        if self._pos >= len(self._token):
            raise ValueError("sequenza troncata: parentesi non chiusa")
        t = self._token[self._pos]
        self._pos += 1
        return t

    def attendi(self, atteso: str) -> None:
        t = self.prossimo()
        if t != atteso:
            raise ValueError(f"atteso {atteso!r}, trovato {t!r}")

    def esaurito(self) -> bool:
        return self._pos >= len(self._token)

    @property
    def pos(self) -> int:
        return self._pos


def token_a_grafo(token: Sequence[str], famiglia: str) -> Grafo:
    """Ricostruisce un `Grafo` da una sequenza di token.

    `famiglia`: "evento" | "fatto". Solleva `ValueError` con messaggio
    chiaro se la sequenza è malformata (parentesi sbilanciate, relazione
    ignota, ordinale orfano, campo obbligatorio mancante...).
    """
    if famiglia not in ("evento", "fatto"):
        raise ValueError(f"famiglia sconosciuta: {famiglia!r}")

    lex = _ottieni_lessico()
    c = _Cursore(token)

    c.attendi(APERTA)
    radice_lemma = c.prossimo()
    if radice_lemma in (APERTA, CHIUSA):
        raise ValueError(f"lemma radice mancante o non valido: {radice_lemma!r}")
    if radice_lemma not in lex:
        raise ValueError(f"lemma radice sconosciuto nel lessico: {radice_lemma!r}")

    rami: list[tuple[str, str]] = []
    while True:
        t = c.prossimo()
        if t == CHIUSA:
            break
        if t != APERTA:
            raise ValueError(f"atteso {APERTA!r} o {CHIUSA!r} dopo la radice, trovato {t!r}")

        relazione = c.prossimo()
        if relazione not in RELAZIONI_UD:
            raise ValueError(f"relazione sconosciuta: {relazione!r}")

        lemma = c.prossimo()
        if lemma in (APERTA, CHIUSA):
            raise ValueError(f"lemma mancante nel ramo {relazione!r}")

        t2 = c.prossimo()
        if t2 == CHIUSA:
            lemma_completo = lemma
        else:
            try:
                n = ordinale_inverso(t2)
            except ValueError as exc:
                raise ValueError(f"atteso ordinale o {CHIUSA!r} dopo {lemma!r}, trovato {t2!r}") from exc
            if lemma not in lex:
                raise ValueError(f"base d'istanza sconosciuta nel lessico: {lemma!r}")
            lemma_completo = f"{lemma}_{n}"
            c.attendi(CHIUSA)

        rami.append((relazione, lemma_completo))

    if not c.esaurito():
        raise ValueError(f"token in eccesso dopo la chiusura del grafo: {list(token[c.pos:])}")

    if radice_lemma == "non-lo-so":
        if rami:
            raise ValueError("il nodo 'non-lo-so' non ammette rami")
        return NON_LO_SO

    mappa: dict[str, str] = {}
    for relazione, lemma_completo in rami:
        if relazione in mappa:
            raise ValueError(f"relazione ripetuta: {relazione!r}")
        mappa[relazione] = lemma_completo

    if famiglia == "evento":
        return _mappa_a_evento(radice_lemma, mappa)
    return grafo_fatto(radice_lemma, **mappa)


def _mappa_a_evento(azione: str, mappa: dict[str, str]) -> Grafo:
    if "nsubj" not in mappa:
        raise ValueError("campo obbligatorio mancante per un evento: nsubj (agente)")
    if "obl:tempo" not in mappa:
        raise ValueError("campo obbligatorio mancante per un evento: obl:tempo")

    tempo_lemma = mappa["obl:tempo"]
    if tempo_lemma not in _LEMMA_A_VALORE_NUM:
        raise ValueError(f"lemma di tempo non numerico: {tempo_lemma!r}")

    evento = Evento(
        t=_LEMMA_A_VALORE_NUM[tempo_lemma],
        azione=azione,
        agente=mappa["nsubj"],
        oggetto=mappa.get("obj"),
        destinatario=mappa.get("iobj"),
        argomento=mappa.get("obl:argomento"),
        luogo_origine=mappa.get("obl:origine"),
        luogo=mappa.get("obl:luogo"),
    )
    return evento_a_grafo(evento)


def componi_esempio(
    storia: Sequence[Sequence[str]], domanda: Sequence[str], risposta: Sequence[str],
) -> list[str]:
    """`[STORIA] <grafi eventi concatenati> [DOMANDA] <grafo> [RISPOSTA] <grafo> [FINE]`."""
    token: list[str] = [STORIA]
    for grafo_evento in storia:
        token.extend(grafo_evento)
    token.append(DOMANDA)
    token.extend(domanda)
    token.append(RISPOSTA)
    token.extend(risposta)
    token.append(FINE)
    return token
