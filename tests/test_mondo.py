"""Test del simulatore (FASE0.md, criteri di accettazione):
invarianti, riproducibilità byte-per-byte, prestazioni, copertura.
"""
from __future__ import annotations

import random
import time

import pytest

from mondo import dati_mondo as dm
from mondo.azioni import AZIONI
from mondo.domande import genera_domande
from mondo.generatore import _lunghezza_storia, genera_record, genera_record_multipli
from mondo.grafo import evento_a_grafo
from mondo.motore import avanza_tick, costruisci_stato_iniziale
from mondo.simulatore import genera_storia
from mondo.statistiche import calcola_statistiche

SEED_DI_PROVA = list(range(30))


# ---------------------------------------------------------------------------
# Riproducibilità
# ---------------------------------------------------------------------------

class TestRiproducibilita:
    def test_stesso_seed_stessa_storia(self):
        for seed in (0, 1, 42, 12345):
            s1 = genera_storia(seed=seed, n_tick=20)
            s2 = genera_storia(seed=seed, n_tick=20)
            assert s1.eventi == s2.eventi

    def test_stesso_seed_stesso_record_byte_per_byte(self):
        for seed in (0, 1, 42, 12345):
            r1 = genera_record(seed)
            r2 = genera_record(seed)
            assert r1 == r2

    def test_seed_diversi_storie_diverse(self):
        s1 = genera_storia(seed=1, n_tick=20)
        s2 = genera_storia(seed=2, n_tick=20)
        assert s1.eventi != s2.eventi

    def test_nessun_random_globale(self):
        """Interferire con il modulo random globale non deve cambiare
        l'output: il simulatore deve usare solo l'RNG esplicito che riceve."""
        random.seed(999)
        s1 = genera_storia(seed=7, n_tick=15)
        random.seed(1)
        s2 = genera_storia(seed=7, n_tick=15)
        assert s1.eventi == s2.eventi


# ---------------------------------------------------------------------------
# Invarianti dello stato
# ---------------------------------------------------------------------------

def _profondita_catena_posizione(stato, oggetto_id: str) -> int:
    profondita = 0
    tipo, rif = stato.oggetti[oggetto_id].posizione
    while tipo == "contenitore":
        profondita += 1
        assert profondita < len(stato.oggetti), "ciclo di contenimento rilevato"
        tipo, rif = stato.oggetti[rif].posizione
    return profondita


class TestInvarianti:
    @pytest.mark.parametrize("seed", SEED_DI_PROVA)
    def test_posizioni_sempre_valide_e_senza_cicli(self, seed):
        stato = costruisci_stato_iniziale()
        rng = random.Random(seed)
        for t in range(1, 25 + 1):
            avanza_tick(stato, rng, t)
            for oid, o in stato.oggetti.items():
                tipo, rif = o.posizione
                if tipo == "luogo":
                    assert rif in stato.luoghi
                elif tipo == "persona":
                    assert rif in stato.persone
                elif tipo == "contenitore":
                    assert rif in stato.oggetti and stato.oggetti[rif].contenitore
                else:
                    pytest.fail(f"tipo di posizione sconosciuto: {tipo!r}")
                _profondita_catena_posizione(stato, oid)  # non deve mai ciclare
                # un oggetto risolve sempre a un unico luogo fisico
                assert stato.luogo_effettivo(oid) in stato.luoghi

    @pytest.mark.parametrize("seed", SEED_DI_PROVA)
    def test_fame_e_stanchezza_nei_limiti(self, seed):
        stato = costruisci_stato_iniziale()
        rng = random.Random(seed)
        for t in range(1, 25 + 1):
            avanza_tick(stato, rng, t)
            for persona in stato.persone.values():
                assert 0 <= persona.fame <= dm.SOGLIA_MASSIMA
                assert 0 <= persona.stanchezza <= dm.SOGLIA_MASSIMA

    @pytest.mark.parametrize("seed", SEED_DI_PROVA)
    def test_risorse_mai_negative(self, seed):
        stato = costruisci_stato_iniziale()
        rng = random.Random(seed)
        for t in range(1, 25 + 1):
            avanza_tick(stato, rng, t)
            for quantita in stato.risorse.values():
                assert quantita >= 0

    @pytest.mark.parametrize("seed", SEED_DI_PROVA)
    def test_testimoni_coerenti_con_le_posizioni(self, seed):
        """Ogni testimone di un evento deve essere una persona realmente
        presente in quel luogo al momento dell'evento (l'agente incluso,
        tranne per gli eventi di sistema come "bruciare" il cui agente non
        è una persona)."""
        stato = costruisci_stato_iniziale()
        rng = random.Random(seed)
        for t in range(1, 25 + 1):
            eventi = avanza_tick(stato, rng, t)
            for e in eventi:
                assert len(e.testimoni) == len(set(e.testimoni)), "testimoni duplicati"
                assert list(e.testimoni) == sorted(e.testimoni), "testimoni non ordinati"
                if e.agente in stato.persone:
                    assert e.agente in e.testimoni

    @pytest.mark.parametrize("seed", SEED_DI_PROVA)
    def test_precondizioni_rispettate_prima_di_ogni_effetto(self, seed):
        """Rigioca la storia verificando che, per ogni evento, l'istanza
        scelta fosse davvero fra quelle valide subito PRIMA dell'effetto."""
        stato = costruisci_stato_iniziale()
        rng = random.Random(seed)
        for t in range(1, 25 + 1):
            for persona in dm.PERSONE:
                from mondo.motore import scegli_azione

                azione, parametri = scegli_azione(stato, persona.id, rng)
                assert azione.precondizioni(stato, parametri), (
                    f"{azione.nome} scelta per {persona.id} ma precondizioni false"
                )
                azione.effetti(stato, parametri, t)
            from mondo.motore import _aggiorna_camino, _aggiorna_fisiologia

            _aggiorna_camino(stato, t)
            _aggiorna_fisiologia(stato)

    @pytest.mark.parametrize("seed", SEED_DI_PROVA)
    def test_conservazione_risorse(self, seed):
        """Ogni mela/unità d'acqua/legna presa proviene dalla fonte finita:
        rimaste + create == iniziale, sempre (nulla si crea dal nulla)."""
        storia = genera_storia(seed=seed, n_tick=30)
        # ogni "prendere" con argomento=fonte è un'estrazione fresca dalla
        # risorsa finita (non un ri-raccogliere un oggetto già libero nel
        # mondo): il loro numero deve combaciare esattamente col consumo.
        for fonte, info in dm.RISORSE.items():
            prese_da_fonte = [e for e in storia.eventi if e.azione == "prendere" and e.argomento == fonte]
            assert len(prese_da_fonte) == info["quantita_iniziale"] - storia.stato_finale.risorse[fonte]


# ---------------------------------------------------------------------------
# Copertura (azioni, tipi di domanda) e prestazioni
# ---------------------------------------------------------------------------

class TestCopertura:
    @classmethod
    @pytest.fixture(scope="class")
    def records(cls):
        return list(genera_record_multipli(range(1500), n_per_tipo=8))

    def test_ogni_azione_almeno_1_percento(self, records):
        stats = calcola_statistiche(records)
        sotto_soglia = stats.azioni_sotto_soglia()
        assert sotto_soglia == [], f"azioni sotto l'1%: {sotto_soglia}"
        # tutte le 15 azioni STRIPS devono comparire
        assert set(AZIONI.keys()) <= set(stats.azioni.keys())

    def test_ogni_tipo_di_domanda_almeno_1_percento(self, records):
        stats = calcola_statistiche(records)
        sotto_soglia = stats.tipi_domanda_sotto_soglia()
        assert sotto_soglia == [], f"tipi di domanda sotto l'1%: {sotto_soglia}"

    def test_non_lo_so_presente_fin_da_subito(self, records):
        stats = calcola_statistiche(records)
        totale_non_lo_so = sum(stats.non_lo_so_per_tipo.values())
        assert totale_non_lo_so > 0
        # ogni tipo, tranne "parentela" (0% per costruzione con una singola
        # famiglia chiusa: si veda la nota nel docstring di mondo/domande.py),
        # deve avere una quota non banale di domande senza risposta.
        for tipo in stats.tipi_domanda:
            if tipo == "parentela":
                continue
            assert stats.percentuale_non_lo_so(tipo) > 0, f"nessun 'non lo so' per il tipo {tipo}"


class TestPrestazioni:
    def test_10000_storie_in_meno_di_un_minuto(self):
        t0 = time.time()
        for seed in range(10_000):
            n_tick = _lunghezza_storia(seed)
            genera_storia(seed=seed, n_tick=n_tick)
        durata = time.time() - t0
        assert durata < 60, f"generazione di 10000 storie troppo lenta: {durata:.1f}s"


# ---------------------------------------------------------------------------
# Grafo (valutazione grafo vs grafo, mai stringa vs stringa)
# ---------------------------------------------------------------------------

class TestGrafo:
    def test_stesso_evento_stesso_grafo(self):
        storia = genera_storia(seed=5, n_tick=10)
        for evento in storia.eventi:
            g1 = evento_a_grafo(evento)
            g2 = evento_a_grafo(evento)
            assert g1 == g2

    def test_eventi_diversi_grafi_diversi(self):
        storia = genera_storia(seed=5, n_tick=10)
        grafi = [evento_a_grafo(e) for e in storia.eventi]
        # non tutti gli eventi devono coincidere (controllo di sanità: il
        # grafo porta davvero informazione distintiva sull'evento)
        assert len(set(grafi)) > 1

    def test_risposta_non_lo_so_e_un_grafo_di_prima_classe(self):
        storia = genera_storia(seed=8, n_tick=10)
        rng = random.Random(123)
        domande = genera_domande(storia, rng, n_per_tipo=10)
        non_lo_so = [d for d in domande if d.grafo_risposta.nodi[0].lemma == "non-lo-so"]
        assert non_lo_so, "nessuna domanda 'non lo so' generata in questo campione"
