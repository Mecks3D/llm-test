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

from dataclasses import dataclass
from typing import Sequence

from lingua.lessico import Lessico, carica_lessico
from lingua.morfologia import ordinale, ordinale_inverso
from mondo.grafo import NON_LO_SO, Grafo, evento_a_grafo, grafo_fatto
from mondo.numeri import VALORE_A_LEMMA
from mondo.tipi import Evento

from .vocabolario import RELAZIONI_UD, TOKEN_SPECIALI, TOKEN_STATO

PAD, STORIA, DOMANDA, RISPOSTA, FINE, APERTA, CHIUSA = TOKEN_SPECIALI
STATO = TOKEN_STATO

# Relazione riusata come etichetta di tick nei blocchi [STATO] (Fase B,
# fasi/FASE2_PIANO_STATO.md §2, decisione di Andrea 2026-07-12): l'indice di
# tick è `( obl:tempo <ordinale> )`, con l'ordinale fra i lemmi-numero già in
# vocabolario. Nessun token nuovo oltre a [STATO] (cancello: test "nessun token
# nuovo").
REL_TICK = "obl:tempo"
# Verbo/relazioni delle posizioni di stato: gli stessi già prodotti dalle
# domande di posizione (`trovarsi`/`nsubj`/`obl:luogo`), nessun lessico nuovo.
VERBO_STATO = "trovarsi"

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


# ---------------------------------------------------------------------------
# Fase B: blocchi di stato interlacciati (fasi/FASE2_PIANO_STATO.md §2)
#
# Un blocco stato descrive, a fine tick, la posizione di OGNI persona del cast:
#
#     [STATO] ( obl:tempo <ordinale> )
#             ( trovarsi ( nsubj p1 ) ( obl:luogo l1 ) )
#             ( trovarsi ( nsubj p2 ) ( obl:luogo l2 ) )
#             ...una `trovarsi` per persona, in ordine deterministico...
#
# L'etichetta di tick `( obl:tempo <ordinale> )` è un ramo isolato (non un
# grafo completo): ha una sua analisi dedicata qui sotto, mentre ogni `trovarsi`
# è un normale grafo "fatto" e riusa grafo_a_token/token_a_grafo.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BloccoStato:
    """Stato del mondo a fine tick: etichetta di tick + una posizione per persona.

    `tick_lemma`: lemma-numero dell'ordinale di tick (es. "nove").
    `posizioni`: i grafi `trovarsi(nsubj=persona, obl:luogo=luogo)`, in ordine
    deterministico (l'ordine del cast, deciso da chi genera).
    """
    tick_lemma: str
    posizioni: tuple[Grafo, ...]


@dataclass(frozen=True)
class EsempioStato:
    """Un esempio con stato interlacciato, scomposto nei suoi grafi.

    `segmenti`: per ogni tick con eventi, la coppia (grafi-evento del tick,
    blocco stato di fine tick), nell'ordine della storia.
    """
    segmenti: tuple[tuple[tuple[Grafo, ...], BloccoStato], ...]
    domanda: Grafo
    risposta: Grafo


def grafo_posizione(persona: str, luogo: str) -> Grafo:
    """Grafo `trovarsi(nsubj=persona, obl:luogo=luogo)` di una posizione di stato."""
    return grafo_fatto(VERBO_STATO, nsubj=persona, **{"obl:luogo": luogo})


def blocco_stato_a_token(tick_lemma: str, posizioni: Sequence[tuple[str, str]]) -> list[str]:
    """Linearizza un blocco stato da `(persona, luogo)` in token stringa.

    `tick_lemma`: lemma-numero dell'ordinale di tick. `posizioni`: coppie
    (persona, luogo) nell'ordine deterministico voluto dal generatore.
    """
    if tick_lemma not in _LEMMA_A_VALORE_NUM:
        raise ValueError(f"etichetta di tick non numerica: {tick_lemma!r}")
    if not posizioni:
        raise ValueError("blocco stato senza posizioni")
    token: list[str] = [STATO, APERTA, REL_TICK, tick_lemma, CHIUSA]
    for persona, luogo in posizioni:
        token.extend(grafo_a_token(grafo_posizione(persona, luogo)))
    return token


def _blocco_stato_grafi_a_token(blocco: BloccoStato) -> list[str]:
    token: list[str] = [STATO, APERTA, REL_TICK, blocco.tick_lemma, CHIUSA]
    for g in blocco.posizioni:
        token.extend(grafo_a_token(g))
    return token


def componi_esempio_stato(
    segmenti: Sequence[tuple[Sequence[Sequence[str]], Sequence[str]]],
    domanda: Sequence[str],
    risposta: Sequence[str],
) -> list[str]:
    """Come `componi_esempio`, ma con i blocchi stato interlacciati per tick.

    `segmenti`: per ogni tick con eventi, la coppia (grafi-evento già
    tokenizzati del tick, blocco stato già tokenizzato — vedi
    `blocco_stato_a_token`). Il blocco stato di ogni tick segue gli eventi di
    quel tick. La domanda e la risposta restano DOPO la storia, come sempre.

    Non tocca `componi_esempio` (cancello byte-identico dei default).
    """
    token: list[str] = [STORIA]
    for eventi_tick, blocco_stato in segmenti:
        for grafo_evento in eventi_tick:
            token.extend(grafo_evento)
        token.extend(blocco_stato)
    token.append(DOMANDA)
    token.extend(domanda)
    token.append(RISPOSTA)
    token.extend(risposta)
    token.append(FINE)
    return token


def _estrai_gruppo(token: Sequence[str], i: int) -> tuple[list[str], int]:
    """Estrae il gruppo parentesizzato bilanciato che inizia a `token[i]`.

    Ritorna la sottolista `( ... )` e l'indice del token successivo.
    """
    if token[i] != APERTA:
        raise ValueError(f"atteso {APERTA!r} all'inizio del gruppo, trovato {token[i]!r}")
    profondita = 0
    j = i
    while j < len(token):
        if token[j] == APERTA:
            profondita += 1
        elif token[j] == CHIUSA:
            profondita -= 1
            if profondita == 0:
                return list(token[i : j + 1]), j + 1
        j += 1
    raise ValueError("sequenza troncata: parentesi non chiusa nel gruppo")


def _analizza_blocco_stato(token: Sequence[str], i: int) -> tuple[BloccoStato, int]:
    """Analizza un blocco `[STATO]` a partire da `token[i] == STATO`."""
    if token[i] != STATO:
        raise ValueError(f"atteso {STATO!r}, trovato {token[i]!r}")
    i += 1
    etichetta, i = _estrai_gruppo(token, i)
    if len(etichetta) != 4 or etichetta[1] != REL_TICK:
        raise ValueError(f"etichetta di tick malformata: {etichetta}")
    tick_lemma = etichetta[2]
    if tick_lemma not in _LEMMA_A_VALORE_NUM:
        raise ValueError(f"etichetta di tick non numerica: {tick_lemma!r}")

    # Il blocco non ha un delimitatore di chiusura: finisce quando arriva un
    # token di controllo ([STATO]/[DOMANDA]) o l'inizio del tick successivo,
    # cioè un grafo la cui radice NON è `trovarsi`. La discriminazione per
    # radice è sicura perché `trovarsi` non è mai un'azione-evento (le 16
    # azioni di mondo/azioni.py non lo includono): compare solo come radice di
    # posizioni di stato, domande e risposte. È lo stesso confine previsto per
    # la decodifica interlacciata d'esame (fasi/FASE2_PIANO_STATO.md §5).
    posizioni: list[Grafo] = []
    while i < len(token) and token[i] == APERTA:
        gruppo, j = _estrai_gruppo(token, i)
        if len(gruppo) < 2 or gruppo[1] != VERBO_STATO:
            break  # inizio del tick successivo: il blocco stato è finito
        posizioni.append(token_a_grafo(gruppo, "fatto"))
        i = j
    if not posizioni:
        raise ValueError("blocco stato senza posizioni")
    return BloccoStato(tick_lemma, tuple(posizioni)), i


def analizza_esempio_stato(token: Sequence[str]) -> EsempioStato:
    """Inverso di `componi_esempio_stato`: scompone la sequenza nei suoi grafi.

    Ogni tick con eventi è seguito dal suo blocco `[STATO]`: gli eventi
    accumulati prima di un `[STATO]` formano un segmento con quel blocco. Dopo
    l'ultimo blocco segue `[DOMANDA]`. Solleva `ValueError` su sequenze
    malformate (eventi di coda senza blocco, delimitatori fuori posto...).
    """
    if not token or token[0] != STORIA:
        raise ValueError(f"atteso {STORIA!r} in testa, trovato {token[0] if token else 'vuoto'!r}")

    segmenti: list[tuple[tuple[Grafo, ...], BloccoStato]] = []
    eventi_correnti: list[Grafo] = []
    i = 1
    while i < len(token) and token[i] != DOMANDA:
        t = token[i]
        if t == APERTA:
            gruppo, i = _estrai_gruppo(token, i)
            eventi_correnti.append(token_a_grafo(gruppo, "evento"))
        elif t == STATO:
            blocco, i = _analizza_blocco_stato(token, i)
            segmenti.append((tuple(eventi_correnti), blocco))
            eventi_correnti = []
        else:
            raise ValueError(f"token inatteso nella storia: {t!r}")
    if eventi_correnti:
        raise ValueError("eventi di coda senza blocco [STATO]")
    if i >= len(token) or token[i] != DOMANDA:
        raise ValueError(f"atteso {DOMANDA!r} dopo la storia")

    gruppo, i = _estrai_gruppo(token, i + 1)
    domanda = token_a_grafo(gruppo, "fatto")
    if i >= len(token) or token[i] != RISPOSTA:
        raise ValueError(f"atteso {RISPOSTA!r} dopo la domanda")
    gruppo, i = _estrai_gruppo(token, i + 1)
    risposta = token_a_grafo(gruppo, "fatto")
    if i >= len(token) or token[i] != FINE:
        raise ValueError(f"atteso {FINE!r} in coda")
    if i + 1 != len(token):
        raise ValueError(f"token in eccesso dopo {FINE!r}: {list(token[i + 1 :])}")
    return EsempioStato(tuple(segmenti), domanda, risposta)


def esempio_stato_a_token(esempio: EsempioStato) -> list[str]:
    """Re-linearizza un `EsempioStato` (inverso di `analizza_esempio_stato`)."""
    token: list[str] = [STORIA]
    for eventi, blocco in esempio.segmenti:
        for g in eventi:
            token.extend(grafo_a_token(g))
        token.extend(_blocco_stato_grafi_a_token(blocco))
    token.append(DOMANDA)
    token.extend(grafo_a_token(esempio.domanda))
    token.append(RISPOSTA)
    token.extend(grafo_a_token(esempio.risposta))
    token.append(FINE)
    return token
