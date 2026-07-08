"""Collante senza stato fisso tra `mondo/`, `lingua/`, `cervello/`, `esami/`
per l'interfaccia interattiva (fasi/INTERFACCIA_PIANO.md §4).

Nessuna logica nuova: ogni funzione qui richiama quella già scritta nei
moduli del curriculum. `app.py` non deve mai importare `mondo`/`lingua`/
`cervello`/`esami` direttamente: passa sempre da qui.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from mondo import dati_mondo as dm
from mondo.domande import Domanda, genera_domande
from mondo.generatore import N_PER_TIPO_DEFAULT, e_seed_esame
from mondo.grafo import NON_LO_SO, evento_a_grafo
from mondo.simulatore import Storia, genera_storia

from lingua.contesto import StatoDiscorso
from lingua.verbalizza import verbalizza_domanda, verbalizza_evento, verbalizza_risposta

from cervello.modello import Modello
from cervello.sequenza import (
    DOMANDA as TOK_DOMANDA,
    FINE,
    RISPOSTA as TOK_RISPOSTA,
    STORIA as TOK_STORIA,
    grafo_a_token,
    token_a_grafo,
)
from cervello.vocabolario import Vocabolario, carica_vocabolario

from esami.esamina import _carica_modello, _categoria, decodifica_greedy, dispositivo
from esami.genera import _classifica_domanda_posizione, _n_tick, carica_config

__all__ = [
    "StoriaGenerata", "DomandaMostrata", "EsitoDomanda",
    "verifica_seed", "cast_da_id", "n_tick_auto", "genera_storia_e_testo", "domande_candidate",
    "carica_modello", "chiedi_al_modello",
    "carica_config", "dispositivo",
]


def n_tick_auto(stadio: int, seed: int, config: dict) -> int:
    """Lunghezza della storia secondo la regola dello stadio nel config
    (`storie_corte`: 3-6 tick oppure lunghezza piena — stessa logica di
    `esami.genera._n_tick`, non duplicata)."""
    return _n_tick(stadio, seed, config)


def verifica_seed(seed: int, permetti_seed_esame: bool) -> None:
    """Rifiuta i seed d'esame a meno che non siano esplicitamente permessi
    (fasi/INTERFACCIA_PIANO.md §7.1): stesso spirito di `esami/genera.py`,
    per non "contaminare" a occhio storie che devono restare cieche."""
    if e_seed_esame(seed) and not permetti_seed_esame:
        raise ValueError(
            f"seed {seed} riservato agli esami (>= 1.000.000): rifiutato "
            "(passare permetti_seed_esame=True per esplorarlo consapevolmente)"
        )


def cast_da_id(id_persone: list[str] | None) -> tuple[dm.Persona, ...] | None:
    """Sottoinsieme esplicito di `dati_mondo.PERSONE` scelto nell'interfaccia
    (analogo a `esami.genera._cast_persone`, ma da una lista di id invece
    che dal config: qui il cast è deciso a mano, non dal config caricato)."""
    if id_persone is None:
        return None
    richiesti = set(id_persone)
    persone = tuple(p for p in dm.PERSONE if p.id in richiesti)
    mancanti = richiesti - {p.id for p in persone}
    if mancanti:
        raise ValueError(f"id persona sconosciuti: {sorted(mancanti)}")
    return persone


@dataclass(frozen=True)
class StoriaGenerata:
    storia: Storia
    righe_per_tick: list[str]
    token_eventi: list[list[str]]
    storia_flat: list[str]
    contesto: StatoDiscorso


def genera_storia_e_testo(
    seed: int, n_tick: int, cast: tuple[dm.Persona, ...] | None = None,
) -> StoriaGenerata:
    """Genera la storia e la verbalizza in italiano, tick per tick — stesso
    ciclo di `lingua/__main__.py::_comando_campione_storia` (non
    riscritto, calcato)."""
    storia = genera_storia(seed=seed, n_tick=n_tick, persone=cast)
    contesto = StatoDiscorso()
    righe: list[str] = []
    riga_corrente: list[str] = []
    tick_corrente: int | None = None
    token_eventi: list[list[str]] = []
    for evento in storia.eventi:
        grafo = evento_a_grafo(evento)
        token_eventi.append(grafo_a_token(grafo))
        frase = verbalizza_evento(grafo, contesto)
        if tick_corrente is not None and evento.t != tick_corrente:
            righe.append(" ".join(riga_corrente))
            riga_corrente = []
        riga_corrente.append(frase)
        tick_corrente = evento.t
    if riga_corrente:
        righe.append(" ".join(riga_corrente))
    storia_flat = [t for token in token_eventi for t in token]
    return StoriaGenerata(
        storia=storia, righe_per_tick=righe, token_eventi=token_eventi,
        storia_flat=storia_flat, contesto=contesto,
    )


@dataclass(frozen=True)
class DomandaMostrata:
    domanda: Domanda
    difficolta: str  # "facile" | "difficile" | "non-lo-so" | "-"
    testo_domanda: str
    testo_risposta_oro: str


def domande_candidate(
    storia_gen: StoriaGenerata, seed: int, tipi_ammessi: set[str],
    n_per_tipo: int = N_PER_TIPO_DEFAULT,
) -> list[DomandaMostrata]:
    """Domande candidate filtrate per i tipi che il checkpoint sa gestire,
    con testo in italiano e tag di difficoltà (per "posizione", riusa la
    classificazione anti-scorciatoia di `esami/genera.py`)."""
    rng = random.Random(f"domande-{seed}")
    grezze = genera_domande(storia_gen.storia, rng, n_per_tipo=n_per_tipo)

    mostrate: list[DomandaMostrata] = []
    for d in grezze:
        if d.tipo not in tipi_ammessi:
            continue
        if d.tipo == "posizione":
            difficolta = _classifica_domanda_posizione(storia_gen.storia, d)
        else:
            difficolta = "non-lo-so" if d.grafo_risposta == NON_LO_SO else "-"
        testo_domanda = verbalizza_domanda(d.grafo_domanda, storia_gen.contesto)
        testo_risposta_oro = verbalizza_risposta(d.grafo_risposta, storia_gen.contesto)
        mostrate.append(DomandaMostrata(
            domanda=d, difficolta=difficolta,
            testo_domanda=testo_domanda, testo_risposta_oro=testo_risposta_oro,
        ))
    return mostrate


def carica_modello(config: dict, percorso_checkpoint: str | Path, device: str) -> tuple[Modello, Vocabolario]:
    vocab = carica_vocabolario()
    modello = _carica_modello(config, str(percorso_checkpoint), device)
    return modello, vocab


@dataclass(frozen=True)
class EsitoDomanda:
    categoria: str  # esatto | invenzione | astensione_errata | malformata | errore
    esatto: bool
    testo_risposta_modello: str
    token_domanda: list[str]
    token_risposta_modello: list[str]
    token_risposta_oro: list[str]


def chiedi_al_modello(
    modello: Modello, vocab: Vocabolario, storia_gen: StoriaGenerata,
    domanda: DomandaMostrata, ctx: int, device: str,
) -> EsitoDomanda:
    """Fa rispondere il modello a `domanda` e confronta grafo vs grafo
    (regola non negoziabile #4: mai stringa vs stringa)."""
    tok_domanda = grafo_a_token(domanda.domanda.grafo_domanda)
    tok_risposta_oro = grafo_a_token(domanda.domanda.grafo_risposta)
    prefisso_token = [TOK_STORIA, *storia_gen.storia_flat, TOK_DOMANDA, *tok_domanda, TOK_RISPOSTA]
    prefisso_ids = [vocab.id(t) for t in prefisso_token]

    generati_ids = decodifica_greedy(modello, vocab, prefisso_ids, ctx, device)
    generati_token = [vocab.token(i) for i in generati_ids]
    if generati_token and generati_token[-1] == FINE:
        generati_token = generati_token[:-1]

    grafo_oro = domanda.domanda.grafo_risposta
    try:
        grafo_generato = token_a_grafo(generati_token, "fatto")
    except ValueError:
        grafo_generato = None

    categoria = _categoria(grafo_oro, grafo_generato)
    if grafo_generato is not None:
        testo_modello = verbalizza_risposta(grafo_generato, storia_gen.contesto)
    else:
        testo_modello = f"(sequenza malformata: {' '.join(generati_token)})"

    return EsitoDomanda(
        categoria=categoria, esatto=categoria == "esatto",
        testo_risposta_modello=testo_modello, token_domanda=tok_domanda,
        token_risposta_modello=generati_token, token_risposta_oro=tok_risposta_oro,
    )
