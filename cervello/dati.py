"""JSONL -> batch di tensori con maschera di loss (FASE2_PIANO.md §7).

Carica i record scritti da `esami/genera.py`, compone ogni esempio con
`componi_esempio`, converte in id del vocabolario e impacchetta batch
paddati (a destra, con [PAD]) alla sequenza più lunga del batch. La
maschera di loss è vera solo dalle posizioni della risposta (dopo
[RISPOSTA], fino a [FINE] incluso) — storia e domanda sono date, non si
imparano a pappagallo.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch

from .sequenza import FINE, PAD, RISPOSTA, STATO, componi_esempio, span_blocchi_stato
from .vocabolario import Vocabolario


def carica_esempi(percorso: str | Path) -> list[list[str]]:
    """Legge un JSONL di `esami/genera.py` e compone ogni esempio in una
    sequenza di token completa (`[STORIA]...[RISPOSTA]...[FINE]`)."""
    esempi: list[list[str]] = []
    with open(percorso, encoding="utf-8") as f:
        for riga in f:
            record = json.loads(riga)
            for esempio in record["esempi"]:
                esempi.append(
                    componi_esempio([record["storia"]], esempio["domanda"], esempio["risposta"])
                )
    return esempi


def _maschera_piena(token: Sequence[str]) -> list[bool]:
    """Vera per le posizioni dopo [RISPOSTA] fino a [FINE] incluso."""
    idx_risposta = token.index(RISPOSTA)
    idx_fine = token.index(FINE)
    return [idx_risposta < i <= idx_fine for i in range(len(token))]


def _maschera_stato(token: Sequence[str]) -> list[bool]:
    """Maschera di loss per gli esempi con blocchi [STATO] (Fase B,
    fasi/FASE2_PIANO_STATO.md §3).

    Vera dove il modello deve *generare*: (a) il contenuto di ogni blocco
    [STATO] (etichetta di tick + posizioni `trovarsi`) e (b) la risposta,
    esattamente come `_maschera_piena` (dopo [RISPOSTA] fino a [FINE] incluso).
    Falsa sugli eventi della storia, sulla domanda e sui delimitatori
    [STORIA]/[STATO]/[DOMANDA]/[RISPOSTA] (dati, non imparati a emettere).

    Su un esempio senza blocchi stato coincide con `_maschera_piena`.
    """
    maschera = _maschera_piena(token)
    for inizio, fine in span_blocchi_stato(token):
        for i in range(inizio, fine):
            maschera[i] = True
    return maschera


@dataclass(frozen=True)
class Batch:
    input: torch.Tensor      # (B, T) long
    bersaglio: torch.Tensor  # (B, T) long
    maschera: torch.Tensor   # (B, T) bool — vera dove il bersaglio conta per la loss


def impacchetta_batch(esempi: Sequence[Sequence[str]], vocab: Vocabolario) -> Batch:
    """Converte un gruppo di sequenze-token in un `Batch` paddato alla più
    lunga del gruppo (shift standard: input = seq[:-1], bersaglio = seq[1:])."""
    if not esempi:
        raise ValueError("batch vuoto")

    id_pad = vocab.id(PAD)
    sequenze_id = [[vocab.id(t) for t in token] for token in esempi]
    # Byte-identico sul default: un esempio senza blocchi [STATO] usa
    # esattamente `_maschera_piena` di sempre; solo gli esempi con stato
    # (Fase B) attivano `_maschera_stato`.
    maschere_piene = [
        (_maschera_stato if STATO in token else _maschera_piena)(token)
        for token in esempi
    ]

    lunghezza_max = max(len(s) for s in sequenze_id)
    if lunghezza_max < 2:
        raise ValueError("sequenza troppo corta per uno shift input/bersaglio")
    B, T = len(sequenze_id), lunghezza_max - 1

    input_ids = torch.full((B, T), id_pad, dtype=torch.long)
    bersaglio_ids = torch.full((B, T), id_pad, dtype=torch.long)
    maschera = torch.zeros((B, T), dtype=torch.bool)

    for riga, (ids, m) in enumerate(zip(sequenze_id, maschere_piene)):
        n = len(ids) - 1
        input_ids[riga, :n] = torch.tensor(ids[:-1], dtype=torch.long)
        bersaglio_ids[riga, :n] = torch.tensor(ids[1:], dtype=torch.long)
        maschera[riga, :n] = torch.tensor(m[1:], dtype=torch.bool)

    return Batch(input=input_ids, bersaglio=bersaglio_ids, maschera=maschera)


def genera_batch(
    esempi: Sequence[Sequence[str]], vocab: Vocabolario, batch_size: int, rng: random.Random,
) -> Iterator[Batch]:
    """Mescola `esempi` con `rng` (seedato dal chiamante, tipicamente per
    epoca) e li impacchetta in batch di `batch_size`."""
    ordine = list(range(len(esempi)))
    rng.shuffle(ordine)
    for i in range(0, len(ordine), batch_size):
        indici = ordine[i : i + batch_size]
        yield impacchetta_batch([esempi[j] for j in indici], vocab)
