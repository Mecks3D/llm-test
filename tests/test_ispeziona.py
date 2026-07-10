"""Test di esami/ispeziona.py (fasi/FASE2_PIANO_DIAGNOSI.md §2, A1)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mondo.grafo import evento_a_grafo, grafo_fatto
from mondo.tipi import Evento

from cervello.modello import ConfigModello, Modello
from cervello.sequenza import DOMANDA, STORIA, grafo_a_token
from cervello.vocabolario import carica_vocabolario

import esami.ispeziona as ispeziona_mod
from esami.ispeziona import (
    _bersaglio_domanda,
    _forward_ispezionato,
    _posizioni_menzione,
    _somma_entita,
    esegui_ispezione,
    induction_esempio,
    pesi_attenzione,
)


@pytest.mark.torch
class TestForwardIspezionato:
    def _config(self):
        vocab = carica_vocabolario()
        return ConfigModello(vocab_size=vocab.dimensione, ctx=64, n_layer=2, n_head=2, d_model=8, d_ff=16, dropout=0.0)

    def test_logits_identici_al_forward_diretto(self):
        torch.manual_seed(0)
        modello = Modello(self._config())
        ids = [3, 7, 1, 9, 2]

        pesi, logits = _forward_ispezionato(modello, ids, "cpu")

        logits_diretti = modello(torch.tensor([ids], dtype=torch.long))[0]
        assert torch.allclose(logits, logits_diretti, atol=1e-5)
        assert len(pesi) == 2
        assert pesi[0].shape == (2, len(ids), len(ids))

    def test_pesi_sono_distribuzioni_causali(self):
        torch.manual_seed(1)
        modello = Modello(self._config())
        ids = [3, 7, 1, 9, 2]
        pesi = pesi_attenzione(modello, ids, "cpu")

        for p in pesi:
            # ogni riga somma a 1 (softmax)
            assert torch.allclose(p.sum(dim=-1), torch.ones(2, len(ids)), atol=1e-5)
            # nessuna attenzione al futuro (maschera causale)
            for h in range(2):
                for t in range(len(ids)):
                    assert torch.all(p[h, t, t + 1:] == 0)

    def test_sequenza_oltre_ctx_solleva(self):
        torch.manual_seed(0)
        modello = Modello(self._config())
        with pytest.raises(ValueError):
            pesi_attenzione(modello, list(range(100)), "cpu")


class TestBersaglioDomanda:
    def test_persona_un_token(self):
        domanda = grafo_a_token(grafo_fatto("trovarsi", nsubj="piero", quesito="dove"))
        trovato = _bersaglio_domanda(domanda)
        assert trovato is not None
        j, bersaglio = trovato
        assert bersaglio == ["piero"]
        assert domanda[j] == "piero"

    def test_istanza_lemma_piu_ordinale(self):
        domanda = grafo_a_token(grafo_fatto("trovarsi", nsubj="mela_2", quesito="dove"))
        trovato = _bersaglio_domanda(domanda)
        assert trovato is not None
        j, bersaglio = trovato
        assert bersaglio == ["mela", "secondo"]
        assert domanda[j:j + 2] == ["mela", "secondo"]

    def test_senza_nsubj_ritorna_none(self):
        domanda = grafo_a_token(grafo_fatto("trovarsi", quesito="dove"))
        assert _bersaglio_domanda(domanda) is None


class TestPosizioniMenzione:
    def test_trova_tutte_le_occorrenze(self):
        storia = ["x", "piero", "y", "piero", "z"]
        assert _posizioni_menzione(storia, ["piero"]) == [1, 3]

    def test_bersaglio_a_due_token(self):
        storia = ["a", "mela", "secondo", "b", "mela", "terzo", "mela", "secondo"]
        assert _posizioni_menzione(storia, ["mela", "secondo"]) == [1, 6]

    def test_nessuna_occorrenza(self):
        assert _posizioni_menzione(["a", "b", "c"], ["piero"]) == []


@pytest.mark.torch
class TestSommaEntita:
    def _esempio_posizione(self, bersaglio="piero", oro="giardino"):
        domanda = grafo_a_token(grafo_fatto("trovarsi", nsubj=bersaglio, quesito="dove"))
        risposta = grafo_a_token(grafo_fatto("essere", nsubj=bersaglio, **{"obl:luogo": oro}))
        return {"tipo": "posizione", "domanda": domanda, "risposta": risposta}

    def test_somma_pesi_sulle_menzioni_giuste(self):
        storia_flat = ["x", "piero", "y", "piero", "z"]
        esempio = self._esempio_posizione()
        _, bersaglio = _bersaglio_domanda(esempio["domanda"])
        assert bersaglio == ["piero"]

        T = 1 + len(storia_flat) + 1 + len(esempio["domanda"])
        pesi = [torch.zeros(1, T, T)]
        j, _ = _bersaglio_domanda(esempio["domanda"])
        pos_query = 1 + len(storia_flat) + 1 + j
        # menzioni di "piero" in storia_flat: indici locali 1 e 3 -> assoluti 2 e 4
        pesi[0][0, pos_query, 2] = 0.3
        pesi[0][0, pos_query, 4] = 0.5

        esito = _somma_entita(pesi, storia_flat, esempio)
        assert esito is not None
        ultima, tutte = esito
        assert ultima.shape == (1, 1)
        assert ultima[0, 0].item() == pytest.approx(0.5)
        assert tutte[0, 0].item() == pytest.approx(0.8)

    def test_non_lo_so_ritorna_none(self):
        domanda = grafo_a_token(grafo_fatto("trovarsi", nsubj="piero", quesito="dove"))
        risposta = ["(", "non-lo-so", ")"]
        esempio = {"tipo": "posizione", "domanda": domanda, "risposta": risposta}
        pesi = [torch.zeros(1, 20, 20)]
        assert _somma_entita(pesi, ["a"] * 10, esempio) is None

    def test_bersaglio_mai_menzionato_ritorna_none(self):
        esempio = self._esempio_posizione()
        pesi = [torch.zeros(1, 20, 20)]
        assert _somma_entita(pesi, ["a", "b", "c"], esempio) is None


class TestInductionEsempio:
    def test_conta_e_somma_le_osservazioni_giuste(self):
        prefisso = ["cane", "gatto", "cane", "topo", "gatto"]
        pesi = [torch.zeros(1, 5, 5)]
        # attesa: (query=2 -> target=1) e (query=4 -> target=2)
        pesi[0][0, 2, 1] = 0.7
        pesi[0][0, 4, 2] = 0.9

        totale, conteggio = induction_esempio(pesi, prefisso)
        assert conteggio == 2
        assert totale[0, 0].item() == pytest.approx(1.6)

    def test_nessuna_ripetizione_conteggio_zero(self):
        prefisso = ["a", "b", "c"]
        pesi = [torch.zeros(1, 3, 3)]
        totale, conteggio = induction_esempio(pesi, prefisso)
        assert conteggio == 0
        assert totale[0, 0].item() == 0.0

    def test_ignora_i_token_strutturali(self):
        # "(" ")" "nsubj" ripetuti non devono generare osservazioni
        prefisso = ["(", "nsubj", "piero", ")", "(", "nsubj", "piero", ")"]
        pesi = [torch.zeros(1, 8, 8)]
        totale, conteggio = induction_esempio(pesi, prefisso)
        # solo "piero" (posizione 6) ripete un token non strutturale (posizione 2)
        assert conteggio == 1


@pytest.mark.torch
class TestEseguiIspezione:
    def _config_modello(self):
        vocab = carica_vocabolario()
        return ConfigModello(vocab_size=vocab.dimensione, ctx=256, n_layer=2, n_head=2, d_model=8, d_ff=16, dropout=0.0)

    def _record_posizione(self, bersaglio, oro, altri_eventi):
        eventi = [*altri_eventi, Evento(t=1, azione="andare", agente=bersaglio, luogo=oro)]
        storia_flat = [t for e in eventi for t in grafo_a_token(evento_a_grafo(e))]
        domanda = grafo_a_token(grafo_fatto("trovarsi", nsubj=bersaglio, quesito="dove"))
        risposta = grafo_a_token(grafo_fatto("essere", nsubj=bersaglio, **{"obl:luogo": oro}))
        return {"seed": 0, "storia": storia_flat, "esempi": [{"tipo": "posizione", "domanda": domanda, "risposta": risposta}]}

    def test_mappa_copre_tutte_le_teste_e_valori_plausibili(self):
        torch.manual_seed(0)
        modello = Modello(self._config_modello())
        vocab = carica_vocabolario()

        record = [
            self._record_posizione("piero", "giardino", [Evento(t=1, azione="andare", agente="anna", luogo="cucina")]),
            self._record_posizione("anna", "cucina", [Evento(t=1, azione="andare", agente="piero", luogo="orto")]),
        ]

        esito = esegui_ispezione(modello, vocab, record, ctx=256, device="cpu")

        assert esito["n_esempi_totali"] == 2
        assert esito["n_esempi_usati"] == 2
        assert esito["n_esempi_entita"] == 2
        assert len(esito["mappa"]) == 2 * 2  # n_layer * n_head
        for d in esito["mappa"]:
            assert 0.0 <= d["stessa_entita_ultima"] <= 1.0
            assert 0.0 <= d["stessa_entita_tutte"] <= 1.0
            assert 0.0 <= d["induction"] <= 1.0
        assert len(esito["top_stessa_entita_ultima"]) == 4
        assert len(esito["top_induction"]) == 4
        valori = [d["stessa_entita_ultima"] for d in esito["top_stessa_entita_ultima"]]
        assert valori == sorted(valori, reverse=True)

    def test_max_esempi_limita_il_campione(self):
        torch.manual_seed(0)
        modello = Modello(self._config_modello())
        vocab = carica_vocabolario()
        record = [
            self._record_posizione("piero", "giardino", []),
            self._record_posizione("anna", "cucina", []),
        ]
        esito = esegui_ispezione(modello, vocab, record, ctx=256, device="cpu", max_esempi=1)
        assert esito["n_esempi_totali"] == 2
        assert esito["n_esempi_usati"] == 1
