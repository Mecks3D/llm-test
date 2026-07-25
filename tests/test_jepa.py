"""Test unitari per il modulo GraphWorldJEPA."""
from __future__ import annotations

import torch
import pytest

from jepa.vocabolario_grafi import (
    ENTITA2ID,
    LUOGO2ID,
    ID2LUOGO,
    AZIONI_SUPPORTATE,
    AZIONE2ID,
    estrai_evento_da_grafo,
)
from jepa.modello_jepa import GraphWorldJEPA
from mondo.grafo import evento_a_grafo
from mondo.tipi import Evento


def test_esclusivita_spaziale_softmax():
    """Verifica che per ogni entità la somma delle probabilità spaziali sia esattamente 1.0 (vincolo fisico)."""
    modello = GraphWorldJEPA(d_embed=32)
    logits = modello.inizializza_stato(batch_size=2)
    prob = modello.ottieni_probabilita_posizione(logits)

    # Dimensione prob: [2, N_entita, N_luoghi]
    somme = torch.sum(prob, dim=-1)
    assert torch.allclose(somme, torch.ones_like(somme), atol=1e-6)


def test_aggiornamento_stato_deterministico():
    """Verifica che un'azione di movimento aggiorni i logits dell'entità target in modo stabile."""
    modello = GraphWorldJEPA(d_embed=32)
    logits = modello.inizializza_stato(batch_size=1)

    sara_id = torch.tensor([ENTITA2ID["sara"]])
    andare_id = torch.tensor([AZIONE2ID["andare"]])
    giardino_id = torch.tensor([LUOGO2ID["giardino"]])

    nuovi_logits = modello.aggiorna_stato(logits, sara_id, andare_id, giardino_id)

    # I logits di sara devono essere cambiati
    assert not torch.allclose(logits[0, sara_id, :], nuovi_logits[0, sara_id, :])
    # I logits delle altre entità non devono essere toccati
    piero_id = ENTITA2ID["piero"]
    assert torch.allclose(logits[0, piero_id, :], nuovi_logits[0, piero_id, :])


def test_energy_engine_e_non_lo_so():
    """Verifica che lo stato incerto (uniforme) produca 'non lo so' e che uno stato certo produca il luogo corretto."""
    modello = GraphWorldJEPA(d_embed=32, soglia_non_lo_so=0.45)
    logits = modello.inizializza_stato(batch_size=1)

    sara_idx = ENTITA2ID["sara"]

    # Stato iniziale uniforme: max prob = 1/6 = 0.166 < 0.45 -> 'non lo so'
    energie, risp = modello.calcola_energia_risposte(logits, sara_idx)
    assert risp == "non lo so"
    assert energie["non lo so"] == 0.0

    # Forziamo una posizione certa per Sara su 'cucina'
    cucina_idx = LUOGO2ID["cucina"]
    logits[0, sara_idx, :] = -10.0
    logits[0, sara_idx, cucina_idx] = 10.0

    energie, risp = modello.calcola_energia_risposte(logits, sara_idx)
    assert risp == "cucina"
    assert energie["cucina"] < 0.01
    assert energie["non lo so"] == 1.0


def test_estrazione_evento_da_grafo():
    """Verifica che un evento UD venga decompresso correttamente in soggetto, azione e destinazione."""
    ev = Evento(
        t=9,
        azione="andare",
        agente="sara",
        luogo_origine="cucina",
        luogo="giardino",
    )
    g = evento_a_grafo(ev)
    info = estrai_evento_da_grafo(g)

    assert info["azione"] == "andare"
    assert info["soggetto"] == "sara"
    assert info["destinazione"] == "giardino"
    assert info["origine"] == "cucina"
