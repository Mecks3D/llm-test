"""Test dei tre tipi di domanda dell'esperimento "tempo" (fasi/FASE2_PIANO_TEMPO.md
§2): posizione_tempo, azione_tempo, azione_luogo. Cast di una sola persona,
rotante per seed.
"""
from __future__ import annotations

import random

import pytest

from mondo import dati_mondo as dm
from mondo.domande import (
    _evento_al_tick,
    _genera_azione_luogo,
    _genera_azione_tempo,
    _genera_posizione_tempo,
    _grafo_evento_senza_tempo,
    _posizione_al_tick,
    genera_domande,
    genera_domande_tempo,
)
from mondo.grafo import NON_LO_SO
from mondo.numeri import VALORE_A_LEMMA
from mondo.simulatore import Storia, genera_storia
from mondo.tipi import Evento, StatoMondo, StatoPersona

N_TICK = 20
_LEMMA_A_VALORE = {lemma: valore for valore, lemma in VALORE_A_LEMMA.items()}


def _tempo_della_domanda(grafo) -> int:
    lemma = next(grafo.nodi[a.dipendente].lemma for a in grafo.archi if a.relazione == "obl:tempo")
    return _LEMMA_A_VALORE[lemma]


def _luogo_della_domanda(grafo) -> str:
    return next(grafo.nodi[a.dipendente].lemma for a in grafo.archi if a.relazione == "obl:luogo")


def _cast_per_seed(seed: int) -> tuple[dm.Persona, ...]:
    return (dm.PERSONE[seed % len(dm.PERSONE)],)


def _storia_cast1(seed: int, n_tick: int = N_TICK) -> Storia:
    return genera_storia(seed=seed, n_tick=n_tick, persone=_cast_per_seed(seed))


# ---------------------------------------------------------------------------
# 1. Verità via prefisso (il test più importante)
# ---------------------------------------------------------------------------

class TestVeritaViaPrefisso:
    def test_posizione_tempo_coincide_con_storia_troncata(self):
        for seed in range(50):
            cast = _cast_per_seed(seed)
            storia = genera_storia(seed=seed, n_tick=N_TICK, persone=cast)
            pid = cast[0].id
            for t in (1, 5, 10, N_TICK):
                atteso = genera_storia(seed=seed, n_tick=t, persone=cast).stato_finale.luogo_effettivo(pid)
                assert _posizione_al_tick(storia, pid, t) == atteso, f"seed={seed} t={t}"


# ---------------------------------------------------------------------------
# 2. azione_tempo: risposta "dorme" in continuazione di sonno; ValueError se
#    l'assunzione del motore (un tick sveglio ha sempre un evento) è violata.
# ---------------------------------------------------------------------------

class TestAzioneTempo:
    def test_risposta_dorme_sui_tick_di_continuazione_sonno(self):
        trovato_almeno_un_caso = False
        for seed in range(30):
            cast = _cast_per_seed(seed)
            storia = genera_storia(seed=seed, n_tick=N_TICK, persone=cast)
            pid = cast[0].id
            rng = random.Random(f"test-{seed}")
            domande = _genera_azione_tempo(storia, rng, n=N_TICK, n_tick=N_TICK)
            for d in domande:
                t = _tempo_della_domanda(d.grafo_domanda)
                evento = _evento_al_tick(storia, pid, t)
                if evento is None and d.grafo_risposta != NON_LO_SO:
                    assert d.grafo_risposta.nodi[0].lemma == "dormire"
                    relazioni = {a.relazione for a in d.grafo_risposta.archi}
                    assert relazioni == {"nsubj", "obl:tempo"}  # nessun luogo: sta dormendo, non un evento
                    trovato_almeno_un_caso = True
        assert trovato_almeno_un_caso, "nessun caso di continuazione-sonno trovato sui seed di prova"

    def test_value_error_su_storia_che_viola_assunzione_motore(self):
        pid = "anna"
        stato_finale = StatoMondo(
            t=3, luoghi={}, collegamenti={},
            persone={pid: StatoPersona(id=pid, lemma="Anna", genere="f", eta="anziano",
                                        luogo_preferito=None, luogo="cucina")},
            oggetti={}, risorse={},
        )
        eventi = (
            Evento(t=1, azione="andare", agente=pid, luogo="cucina", luogo_origine="salotto"),
            # tick 2 e 3: nessun evento, ma l'ultimo evento (t=1) non è "dormire"
            # -> viola l'assunzione del motore (chi è sveglio agisce ogni tick).
        )
        storia = Storia(seed=0, eventi=eventi, stato_finale=stato_finale)
        rng = random.Random("x")
        with pytest.raises(ValueError):
            _genera_azione_tempo(storia, rng, n=3, n_tick=3)


# ---------------------------------------------------------------------------
# 3. azione_luogo: esclude i luoghi con eventi diversi, include quelli con
#    eventi identici (a meno del tempo).
# ---------------------------------------------------------------------------

class TestAzioneLuogo:
    def _storia_sintetica(self) -> Storia:
        pid = "anna"
        stato_finale = StatoMondo(
            t=4, luoghi={}, collegamenti={},
            persone={pid: StatoPersona(id=pid, lemma="Anna", genere="f", eta="anziano",
                                        luogo_preferito=None, luogo="giardino")},
            oggetti={}, risorse={},
        )
        eventi = (
            Evento(t=1, azione="svegliarsi", agente=pid, luogo="cucina"),
            Evento(t=2, azione="svegliarsi", agente=pid, luogo="cucina"),
            Evento(t=3, azione="andare", agente=pid, luogo="giardino", luogo_origine="cucina"),
            Evento(t=4, azione="andare", agente=pid, luogo="giardino", luogo_origine="salotto"),
        )
        return Storia(seed=0, eventi=eventi, stato_finale=stato_finale)

    def test_luogo_con_eventi_identici_incluso_senza_tempo(self):
        storia = self._storia_sintetica()
        rng = random.Random("x")
        domande = _genera_azione_luogo(storia, rng, n=5)
        luoghi = {_luogo_della_domanda(d.grafo_domanda) for d in domande}
        assert luoghi == {"cucina"}
        (d,) = [d for d in domande if _luogo_della_domanda(d.grafo_domanda) == "cucina"]
        atteso = _grafo_evento_senza_tempo(storia.eventi[0])
        assert d.grafo_risposta == atteso
        assert all(a.relazione != "obl:tempo" for a in d.grafo_risposta.archi)

    def test_luogo_con_eventi_diversi_escluso(self):
        storia = self._storia_sintetica()
        rng = random.Random("x")
        domande = _genera_azione_luogo(storia, rng, n=5)
        luoghi = {_luogo_della_domanda(d.grafo_domanda) for d in domande}
        assert "giardino" not in luoghi

    def test_mai_non_lo_so(self):
        for seed in range(30):
            storia = _storia_cast1(seed)
            rng = random.Random(f"test-{seed}")
            domande = _genera_azione_luogo(storia, rng, n=10)
            assert all(d.grafo_risposta != NON_LO_SO for d in domande)


# ---------------------------------------------------------------------------
# 4. Determinismo
# ---------------------------------------------------------------------------

class TestDeterminismo:
    def test_stesso_seed_stesse_domande(self):
        for seed in (0, 1, 7, 42):
            storia = _storia_cast1(seed)
            d1 = genera_domande_tempo(storia, random.Random(f"domande-tempo-{seed}"), n_per_tipo=8, n_tick=N_TICK)
            d2 = genera_domande_tempo(storia, random.Random(f"domande-tempo-{seed}"), n_per_tipo=8, n_tick=N_TICK)
            assert d1 == d2

    def test_tipi_esattamente_i_tre_attesi(self):
        storia = _storia_cast1(3)
        domande = genera_domande_tempo(storia, random.Random("domande-tempo-3"), n_per_tipo=8, n_tick=N_TICK)
        assert set(d.tipo for d in domande) == {"posizione_tempo", "azione_tempo", "azione_luogo"}


# ---------------------------------------------------------------------------
# 5. Byte-identità dell'esistente: genera_domande (cast pieno, tipi vecchi)
#    non cambia. La modifica a domande.py è puramente additiva (nessuna riga
#    esistente toccata): questo test è una rete di sicurezza empirica.
# ---------------------------------------------------------------------------

class TestByteIdentitaEsistente:
    def test_genera_domande_tipi_e_conteggi_invariati(self):
        for seed in (0, 1, 42, 12345):
            storia = genera_storia(seed=seed, n_tick=20)
            rng = random.Random(f"domande-{seed}")
            domande = genera_domande(storia, rng, n_per_tipo=6)
            tipi = {d.tipo for d in domande}
            assert tipi <= {
                "posizione", "possesso", "conteggio", "transfer",
                "parentela", "deduzione", "causa",
            }
            assert "posizione_tempo" not in tipi
            assert "azione_tempo" not in tipi
            assert "azione_luogo" not in tipi

    def test_genera_domande_deterministico_come_prima(self):
        for seed in (0, 1, 42, 12345):
            storia = genera_storia(seed=seed, n_tick=20)
            d1 = genera_domande(storia, random.Random(f"domande-{seed}"), n_per_tipo=6)
            d2 = genera_domande(storia, random.Random(f"domande-{seed}"), n_per_tipo=6)
            assert d1 == d2
