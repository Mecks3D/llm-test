"""Round-trip lingua/ per i tre tipi dell'esperimento "tempo" (fasi/
FASE2_PIANO_TEMPO.md §3.3, cancelli duri): storia a cast singolo, grafo ->
testo -> grafo identico, su ≥300 seed.
"""
from __future__ import annotations

import random

from mondo import dati_mondo as dm
from mondo.domande import genera_domande_tempo
from mondo.grafo import Grafo, evento_a_grafo, grafo_fatto
from mondo.simulatore import genera_storia
from mondo.tipi import Evento

from lingua.analizza import analizza_domanda, analizza_risposta, analizza_storia
from lingua.contesto import StatoDiscorso
from lingua.verbalizza import verbalizza_domanda, verbalizza_risposta
from lingua.verbalizza import verbalizza_storia as _verbalizza_storia

N_TICK = 20
N_SEED = 320


def _cast_per_seed(seed: int) -> tuple[dm.Persona, ...]:
    return (dm.PERSONE[seed % len(dm.PERSONE)],)


CASI_GOLDEN = [
    (grafo_fatto("trovarsi", nsubj="anna", **{"obl:tempo": "due"}, quesito="dove"),
     "Dove si trova Anna alle due?"),
    (grafo_fatto("essere", nsubj="anna", **{"obl:luogo": "cucina", "obl:tempo": "due"}),
     "Alle due Anna è in cucina."),
    (grafo_fatto("fare", nsubj="anna", **{"obl:tempo": "due"}, quesito="che-cosa"),
     "Che cosa fa Anna alle due?"),
    (grafo_fatto("dormire", nsubj="anna", **{"obl:tempo": "due"}),
     "Alle due Anna dorme."),
    (grafo_fatto("fare", nsubj="anna", **{"obl:luogo": "giardino"}, quesito="che-cosa"),
     "Che cosa fa Anna in giardino?"),
]


class TestGoldenTempo:
    def test_verbalizza_e_analizza(self):
        cv, cp = StatoDiscorso(), StatoDiscorso()
        for grafo, frase_attesa in CASI_GOLDEN:
            e_domanda = frase_attesa.endswith("?")
            rendi = verbalizza_domanda if e_domanda else verbalizza_risposta
            analizza = analizza_domanda if e_domanda else analizza_risposta
            frase = rendi(grafo, cv)
            assert frase == frase_attesa, f"{grafo} -> {frase!r} != {frase_attesa!r}"
            ricostruito = analizza(frase_attesa, cp)
            assert ricostruito == grafo, f"{frase_attesa!r} -> {ricostruito} != {grafo}"

    def test_risposta_azione_tempo_evento_pieno(self):
        c = StatoDiscorso()
        evento = Evento(t=2, azione="andare", agente="anna", luogo="giardino", luogo_origine="cucina")
        frase = verbalizza_risposta(evento_a_grafo(evento), c)
        assert frase == "Alle due Anna va dalla cucina in giardino."
        assert analizza_risposta(frase, StatoDiscorso()) == evento_a_grafo(evento)

    def test_risposta_azione_luogo_senza_tempo(self):
        c = StatoDiscorso()
        evento = Evento(t=5, azione="prendere", agente="anna", oggetto="palla", luogo="orto")
        g = evento_a_grafo(evento)
        g_senza_tempo = Grafo(nodi=g.nodi[:-1], archi=g.archi[:-1])
        frase = verbalizza_risposta(g_senza_tempo, c)
        assert frase == "Anna prende la palla nell'orto."
        assert analizza_risposta(frase, StatoDiscorso()) == g_senza_tempo


class TestRoundTripTempo:
    def test_round_trip_domande_tempo_seed_0_319(self):
        for seed in range(N_SEED):
            cast = _cast_per_seed(seed)
            storia = genera_storia(seed=seed, n_tick=N_TICK, persone=cast)
            grafi_eventi = [evento_a_grafo(e) for e in storia.eventi]

            # Narra prima l'intera storia (come farebbe la pipeline vera):
            # solo così max_indice/posizione_persone sono popolati quando si
            # verbalizzano/analizzano le domande (vedi lingua/__main__.py).
            cv = StatoDiscorso()
            frasi_eventi = _verbalizza_storia(grafi_eventi, cv)
            cp = StatoDiscorso()
            assert analizza_storia(frasi_eventi, cp) == grafi_eventi, f"seed {seed}: round-trip eventi fallito"

            rng = random.Random(f"domande-tempo-{seed}")
            domande = genera_domande_tempo(storia, rng, n_per_tipo=8, n_tick=N_TICK)
            assert domande, f"seed {seed}: nessuna domanda tempo generata"
            for d in domande:
                fd = verbalizza_domanda(d.grafo_domanda, cv)
                ottenuto_d = analizza_domanda(fd, cp)
                assert ottenuto_d == d.grafo_domanda, f"seed {seed} {d.tipo}: domanda {fd!r}"

                fr = verbalizza_risposta(d.grafo_risposta, cv)
                ottenuto_r = analizza_risposta(fr, cp)
                assert ottenuto_r == d.grafo_risposta, f"seed {seed} {d.tipo}: risposta {fr!r}"

    def test_determinismo_verbalizzazione(self):
        cast = _cast_per_seed(3)
        storia = genera_storia(seed=3, n_tick=N_TICK, persone=cast)
        domande = genera_domande_tempo(storia, random.Random("domande-tempo-3"), n_per_tipo=8, n_tick=N_TICK)
        cv1, cv2 = StatoDiscorso(), StatoDiscorso()
        for d in domande:
            assert verbalizza_domanda(d.grafo_domanda, cv1) == verbalizza_domanda(d.grafo_domanda, cv2)
            assert verbalizza_risposta(d.grafo_risposta, cv1) == verbalizza_risposta(d.grafo_risposta, cv2)
