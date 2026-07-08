"""Test di interfaccia/ponte.py (fasi/INTERFACCIA_PIANO.md).

Nessun test di `app.py` (finestra tkinter): qui si verifica solo il
collante, che è dove vive tutta la logica non banale."""
from __future__ import annotations

import pytest

from mondo.grafo import grafo_fatto
from cervello.sequenza import grafo_a_token
from cervello.vocabolario import carica_vocabolario

from interfaccia import ponte


class TestVerificaSeed:
    def test_seed_normale_non_solleva(self):
        ponte.verifica_seed(42, permetti_seed_esame=False)

    def test_seed_esame_rifiutato_di_default(self):
        with pytest.raises(ValueError):
            ponte.verifica_seed(1_000_000, permetti_seed_esame=False)

    def test_seed_esame_permesso_esplicitamente(self):
        ponte.verifica_seed(1_000_000, permetti_seed_esame=True)


class TestCastDaId:
    def test_none_restituisce_cast_pieno_implicito(self):
        assert ponte.cast_da_id(None) is None

    def test_sottoinsieme_rispetta_ordine_di_persone(self):
        cast = ponte.cast_da_id(["maria", "anna"])
        assert [p.id for p in cast] == ["anna", "maria"]

    def test_id_sconosciuto_solleva(self):
        with pytest.raises(ValueError):
            ponte.cast_da_id(["anna", "pinco"])


class TestGeneraStoriaETesto:
    def test_deterministico_stesso_seed_stesso_testo(self):
        a = ponte.genera_storia_e_testo(seed=7, n_tick=5)
        b = ponte.genera_storia_e_testo(seed=7, n_tick=5)
        assert a.righe_per_tick == b.righe_per_tick
        assert a.storia_flat == b.storia_flat

    def test_una_riga_per_tick_con_eventi(self):
        storia_gen = ponte.genera_storia_e_testo(seed=7, n_tick=5)
        tick_con_eventi = {e.t for e in storia_gen.storia.eventi}
        assert len(storia_gen.righe_per_tick) == len(tick_con_eventi)

    def test_cast_ridotto_si_riflette_nella_storia(self):
        cast = ponte.cast_da_id(["anna", "piero"])
        storia_gen = ponte.genera_storia_e_testo(seed=7, n_tick=5, cast=cast)
        agenti = {e.agente for e in storia_gen.storia.eventi}
        assert agenti <= {"anna", "piero"}


class TestDomandeCandidate:
    def test_filtra_per_tipo_ammesso(self):
        storia_gen = ponte.genera_storia_e_testo(seed=7, n_tick=8)
        mostrate = ponte.domande_candidate(storia_gen, seed=7, tipi_ammessi={"posizione"})
        assert mostrate
        assert all(m.domanda.tipo == "posizione" for m in mostrate)

    def test_tipo_non_ammesso_restituisce_vuoto(self):
        storia_gen = ponte.genera_storia_e_testo(seed=7, n_tick=8)
        mostrate = ponte.domande_candidate(storia_gen, seed=7, tipi_ammessi=set())
        assert mostrate == []

    def test_difficolta_posizione_tra_le_etichette_note(self):
        storia_gen = ponte.genera_storia_e_testo(seed=7, n_tick=8)
        mostrate = ponte.domande_candidate(storia_gen, seed=7, tipi_ammessi={"posizione"})
        assert {m.difficolta for m in mostrate} <= {"facile", "difficile", "non-lo-so"}

    def test_testo_domanda_e_risposta_non_vuoti(self):
        storia_gen = ponte.genera_storia_e_testo(seed=7, n_tick=8)
        mostrate = ponte.domande_candidate(storia_gen, seed=7, tipi_ammessi={"posizione"})
        for m in mostrate:
            assert m.testo_domanda
            assert m.testo_risposta_oro


class TestNTickAuto:
    def _config(self, storie_corte: bool):
        return {"stadi": {1: {"tipi": ["posizione"], "soglia": 0.95, "storie_corte": storie_corte}}}

    def test_storia_corta_tra_3_e_6(self):
        n = ponte.n_tick_auto(1, seed=7, config=self._config(True))
        assert 3 <= n <= 6

    def test_storia_piena_usa_lunghezza_storia(self):
        from mondo.generatore import _lunghezza_storia
        n = ponte.n_tick_auto(1, seed=7, config=self._config(False))
        assert n == _lunghezza_storia(7)


torch = pytest.importorskip("torch")


class _ModelloIniettato:
    """Copia locale del finto modello di test_esami.py::_ModelloIniettato:
    ad ogni chiamata restituisce logits il cui argmax riproduce, in ordine,
    `sequenza_output`."""

    def __init__(self, vocab, sequenza_output):
        self._ids_output = [vocab.id(t) for t in sequenza_output]
        self._vocab_size = vocab.dimensione
        self._chiamate = 0
        self._T_precedente = None
        self.training = False

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def __call__(self, x):
        B, T = x.shape
        if self._T_precedente is None or T <= self._T_precedente:
            self._chiamate = 0
        self._T_precedente = T
        idx = min(self._chiamate, len(self._ids_output) - 1)
        prossimo_id = self._ids_output[idx]
        self._chiamate += 1
        logits = torch.full((B, T, self._vocab_size), -10.0)
        logits[0, -1, prossimo_id] = 10.0
        return logits


@pytest.mark.torch
class TestChiediAlModello:
    def _storia_gen_e_domanda(self, oro_grafo):
        from lingua.contesto import StatoDiscorso
        storia_gen = ponte.StoriaGenerata(
            storia=None, righe_per_tick=[], token_eventi=[], storia_flat=[],
            contesto=StatoDiscorso(),
        )
        from mondo.domande import Domanda
        domanda_grezza = Domanda(
            tipo="posizione",
            grafo_domanda=grafo_fatto("trovarsi", nsubj="sara", quesito="dove"),
            grafo_risposta=oro_grafo,
        )
        domanda = ponte.DomandaMostrata(
            domanda=domanda_grezza, difficolta="facile", testo_domanda="?", testo_risposta_oro="?",
        )
        return storia_gen, domanda

    def test_esatto(self):
        vocab = carica_vocabolario()
        oro = grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "cucina"})
        storia_gen, domanda = self._storia_gen_e_domanda(oro)
        modello = _ModelloIniettato(vocab, [*grafo_a_token(oro), "[FINE]"])
        esito = ponte.chiedi_al_modello(modello, vocab, storia_gen, domanda, ctx=200, device="cpu")
        assert esito.categoria == "esatto"
        assert esito.esatto is True
        assert "cucina" in esito.testo_risposta_modello.lower()

    def test_errore(self):
        vocab = carica_vocabolario()
        oro = grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "cucina"})
        generato = grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "giardino"})
        storia_gen, domanda = self._storia_gen_e_domanda(oro)
        modello = _ModelloIniettato(vocab, [*grafo_a_token(generato), "[FINE]"])
        esito = ponte.chiedi_al_modello(modello, vocab, storia_gen, domanda, ctx=200, device="cpu")
        assert esito.categoria == "errore"
        assert esito.esatto is False

    def test_malformata_non_solleva(self):
        vocab = carica_vocabolario()
        oro = grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "cucina"})
        storia_gen, domanda = self._storia_gen_e_domanda(oro)
        modello = _ModelloIniettato(vocab, ["(", "essere", "(", "nsubj", "sara"])
        esito = ponte.chiedi_al_modello(modello, vocab, storia_gen, domanda, ctx=200, device="cpu")
        assert esito.categoria == "malformata"
        assert esito.esatto is False
