"""Test di esami/diagnosi.py (fasi/FASE2_PIANO_ANTISCORCIATOIA.md §5.3, §6.8).

Le proprietà D1/D2/D3 sono già testate a fondo su storie sintetiche in
test_esami.py::TestClassificazioneDifficolta; qui si verifica invece che
`esegui_diagnosi` AGGREGHI correttamente baseline/condizionata/anatomia/
per-entità su un "dataset giocattolo" con esiti costruiti a mano, usando un
modello iniettato (nessun training reale) e `genera_storia` stubbato per
poter controllare per intero gli eventi delle due storie di esempio.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mondo.grafo import Grafo, evento_a_grafo, grafo_fatto
from mondo.simulatore import Storia
from mondo.tipi import Evento, StatoMondo, StatoPersona

from cervello.sequenza import grafo_a_token
from cervello.vocabolario import carica_vocabolario

import esami.diagnosi as diagnosi_mod


def _persona(id_: str, luogo: str) -> StatoPersona:
    return StatoPersona(id=id_, lemma=id_, genere="f", eta="adulto", luogo_preferito=None, luogo=luogo)


def _stato(persone: dict[str, str]) -> StatoMondo:
    return StatoMondo(
        t=0, luoghi={}, collegamenti={},
        persone={id_: _persona(id_, luogo) for id_, luogo in persone.items()},
        oggetti={}, risorse={},
    )


# Storia A (seed 100): bersaglio "piero", ultima menzione a t=3 (giardino),
# ma un evento di interferenza a t=4 (cucina, distanza_coda=1) e cucina è il
# luogo più frequente della storia: D1 vero (oro=giardino != piu_frequente=
# cucina), D2 falso, D3 falso (oro == luogo dell'ultima menzione).
_STORIA_A = Storia(
    seed=100,
    eventi=(
        Evento(t=1, azione="andare", agente="anna", luogo="cucina"),
        Evento(t=2, azione="andare", agente="maria", luogo="cucina"),
        Evento(t=3, azione="andare", agente="piero", luogo="giardino"),
        Evento(t=4, azione="andare", agente="anna", luogo="cucina"),
    ),
    stato_finale=_stato({"anna": "cucina", "maria": "cucina", "piero": "giardino"}),
)

# Storia B (seed 200): bersaglio "sara", 5 eventi tutti in "orto": D1 falso
# (oro == piu_frequente), D2 vero (distanza_coda=4), D3 falso.
_STORIA_B = Storia(
    seed=200,
    eventi=(
        Evento(t=1, azione="andare", agente="sara", luogo="orto"),
        Evento(t=2, azione="andare", agente="anna", luogo="orto"),
        Evento(t=3, azione="andare", agente="anna", luogo="orto"),
        Evento(t=4, azione="andare", agente="anna", luogo="orto"),
        Evento(t=5, azione="andare", agente="anna", luogo="orto"),
    ),
    stato_finale=_stato({"sara": "orto", "anna": "orto"}),
)

_STORIE = {100: _STORIA_A, 200: _STORIA_B}


def _domanda_e_risposta(bersaglio: str, oro: str) -> tuple[list[str], list[str]]:
    dom = grafo_a_token(grafo_fatto("trovarsi", nsubj=bersaglio, quesito="dove"))
    ris = grafo_a_token(grafo_fatto("essere", nsubj=bersaglio, **{"obl:luogo": oro}))
    return dom, ris


class _ModelloMultiRisposta:
    """Come `_ModelloIniettato` di test_esami.py, ma con una risposta
    diversa per ogni sessione di decodifica successiva (una per esempio),
    nell'ordine passato al costruttore."""

    def __init__(self, vocab, risposte_token: list[list[str]]):
        self._risposte_ids = [[vocab.id(t) for t in r] for r in risposte_token]
        self._vocab_size = vocab.dimensione
        self._sessione = -1
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
            self._sessione += 1
            self._chiamate = 0
        self._T_precedente = T
        ids_output = self._risposte_ids[self._sessione]
        idx = min(self._chiamate, len(ids_output) - 1)
        prossimo_id = ids_output[idx]
        self._chiamate += 1
        logits = torch.full((B, T, self._vocab_size), -10.0)
        logits[0, -1, prossimo_id] = 10.0
        return logits


@pytest.mark.torch
class TestEseguiDiagnosi:
    def test_metriche_2_a_4_calcolate_a_mano(self, monkeypatch):
        monkeypatch.setattr(diagnosi_mod, "genera_storia", lambda seed, n_tick, persone: _STORIE[seed])
        vocab = carica_vocabolario()

        dom_a, ris_a = _domanda_e_risposta("piero", "giardino")
        dom_b, ris_b = _domanda_e_risposta("sara", "orto")
        record = [
            {"seed": 100, "storia": [], "esempi": [{"tipo": "posizione", "domanda": dom_a, "risposta": ris_a}]},
            {"seed": 200, "storia": [], "esempi": [{"tipo": "posizione", "domanda": dom_b, "risposta": ris_b}]},
        ]

        # esempio A: il modello risponde "cucina" (il luogo più frequente
        # della storia A), sbagliando l'oro "giardino" -> categoria "errore".
        # esempio B: il modello risponde correttamente "orto".
        _, generato_a = _domanda_e_risposta("piero", "cucina")
        risposte = [[*generato_a, "[FINE]"], [*ris_b, "[FINE]"]]
        modello = _ModelloMultiRisposta(vocab, risposte)

        config = {"stadi": {1: {"tipi": ["posizione"], "soglia": 0.95, "storie_corte": True}}, "dataset": {}}
        esito = diagnosi_mod.esegui_diagnosi(modello, vocab, record, config, stadio=1, ctx=200, device="cpu")

        assert esito["n_esempi"] == 2
        assert esito["conteggi"] == {"esatto": 1, "invenzione": 0, "astensione_errata": 0, "malformata": 0, "errore": 1}
        assert esito["esattezza"] == pytest.approx(0.5)
        assert esito["n_posizione_oro_noto"] == 2

        # metrica 2: baseline euristiche sul sottoinsieme oro noto (n=2)
        assert esito["baseline"]["ultima_menzione"] == pytest.approx(1.0)  # entrambe: oro == ultima menzione
        assert esito["baseline"]["piu_frequente"] == pytest.approx(0.5)   # solo B
        assert esito["baseline"]["ultimo_evento"] == pytest.approx(0.5)   # solo B (ultimo evento storia B = orto)
        assert esito["baseline"]["modello"] == pytest.approx(0.5)         # solo B esatto

        # metrica 3: esattezza condizionata
        cond = esito["condizionata"]
        assert cond["oro_uguale_piu_frequente"]["si"] == {"esattezza": pytest.approx(1.0), "n": 1}  # B
        assert cond["oro_uguale_piu_frequente"]["no"] == {"esattezza": pytest.approx(0.0), "n": 1}  # A
        assert cond["distanza_coda"]["1-2"] == {"esattezza": pytest.approx(0.0), "n": 1}  # A: distanza 1
        assert cond["distanza_coda"]["3-5"] == {"esattezza": pytest.approx(1.0), "n": 1}  # B: distanza 4
        assert cond["distanza_coda"]["0"] == {"esattezza": 0.0, "n": 0}
        assert cond["distanza_coda"][">=6"] == {"esattezza": 0.0, "n": 0}
        assert cond["d3_tracking_puro"]["no"] == {"esattezza": pytest.approx(0.5), "n": 2}  # A e B: D3 falso
        assert cond["d3_tracking_puro"]["si"] == {"esattezza": 0.0, "n": 0}

        # metrica 4: anatomia degli errori (solo la categoria "errore": A)
        assert esito["anatomia_errori"] == {"piu_frequente": 1}

        # metrica 5: esattezza per entità
        assert esito["per_entita"]["piero"] == {"esattezza": pytest.approx(0.0), "n": 1}
        assert esito["per_entita"]["sara"] == {"esattezza": pytest.approx(1.0), "n": 1}

    def test_max_esempi_limita_il_conteggio(self, monkeypatch):
        monkeypatch.setattr(diagnosi_mod, "genera_storia", lambda seed, n_tick, persone: _STORIE[seed])
        vocab = carica_vocabolario()
        dom_a, ris_a = _domanda_e_risposta("piero", "giardino")
        dom_b, ris_b = _domanda_e_risposta("sara", "orto")
        record = [
            {"seed": 100, "storia": [], "esempi": [{"tipo": "posizione", "domanda": dom_a, "risposta": ris_a}]},
            {"seed": 200, "storia": [], "esempi": [{"tipo": "posizione", "domanda": dom_b, "risposta": ris_b}]},
        ]
        modello = _ModelloMultiRisposta(vocab, [[*ris_a, "[FINE]"], [*ris_b, "[FINE]"]])
        config = {"stadi": {1: {"tipi": ["posizione"], "soglia": 0.95, "storie_corte": True}}, "dataset": {}}

        esito = diagnosi_mod.esegui_diagnosi(
            modello, vocab, record, config, stadio=1, ctx=200, device="cpu", max_esempi=1,
        )
        assert esito["n_esempi"] == 1
        assert esito["n_posizione_oro_noto"] == 1


_STORIA_TEMPO = Storia(
    seed=300,
    eventi=(
        Evento(t=1, azione="andare", agente="anna", luogo="giardino", luogo_origine="cucina"),
        Evento(t=2, azione="andare", agente="anna", luogo="orto", luogo_origine="giardino"),
        Evento(t=3, azione="andare", agente="anna", luogo="cucina", luogo_origine="orto"),
        Evento(t=4, azione="andare", agente="anna", luogo="salotto", luogo_origine="cucina"),
        Evento(t=5, azione="andare", agente="anna", luogo="camera", luogo_origine="salotto"),
    ),
    stato_finale=_stato({"anna": "camera"}),
)


@pytest.mark.torch
class TestEseguiDiagnosiTrackingTempo:
    """esami/diagnosi.py --split tracking-tempo (fasi/FASE2_PIANO_TEMPO.md
    §4.3): esegui_diagnosi resta generico (metrica 1 sempre corretta), le
    metriche 2-5 (specifiche di "posizione") restano a zero perché il tipo
    è "posizione_tempo", non "posizione" — vedi commento nel CLI."""

    def test_metrica_1_corretta_metriche_posizione_a_zero(self, monkeypatch):
        monkeypatch.setattr(diagnosi_mod, "genera_storia", lambda seed, n_tick, persone: _STORIA_TEMPO)
        vocab = carica_vocabolario()

        # posizione_tempo a t=1 (oro "giardino"): un solo esempio, cosi'
        # niente ambiguita' di sessione nel modello iniettato (T monotona).
        dom = grafo_a_token(grafo_fatto("trovarsi", nsubj="anna", **{"obl:tempo": "uno"}, quesito="dove"))
        ris = grafo_a_token(grafo_fatto("essere", nsubj="anna", **{"obl:luogo": "giardino", "obl:tempo": "uno"}))
        record = [{"seed": 300, "storia": [], "esempi": [{"tipo": "posizione_tempo", "domanda": dom, "risposta": ris}]}]

        modello = _ModelloMultiRisposta(vocab, [["(", "non-lo-so", ")", "[FINE]"]])
        config = {
            "stadi": {1: {"tipi": ["posizione_tempo"], "soglia": 0.95, "storie_corte": True}},
            "dataset": {},
        }
        esito = diagnosi_mod.esegui_diagnosi(modello, vocab, record, config, stadio=1, ctx=200, device="cpu")

        assert esito["n_esempi"] == 1
        assert esito["conteggi"] == {"esatto": 0, "invenzione": 0, "astensione_errata": 1, "malformata": 0, "errore": 0}
        assert esito["esattezza"] == 0.0
        # metriche specifiche di "posizione": mai popolate, il tipo qui è
        # "posizione_tempo".
        assert esito["n_posizione_oro_noto"] == 0
        assert esito["per_entita"] == {}
        assert esito["anatomia_errori"] == {}
        # sezione "tempo": popolata (astensione_errata entra in n_oro_noto,
        # niente anatomia perché non è categoria "errore").
        assert esito["tempo"]["posizione_tempo"]["n_oro_noto"] == 1
        assert esito["tempo"]["posizione_tempo"]["anatomia_errori"] == {}


def _grafo_evento_test(evento: Evento, *, con_tempo: bool = True) -> list[str]:
    g = evento_a_grafo(evento)
    if not con_tempo:
        g = Grafo(nodi=g.nodi[:-1], archi=g.archi[:-1])
    return grafo_a_token(g)


@pytest.mark.torch
class TestEseguiDiagnosiSezioneTempo:
    """Anatomia degli errori dei tipi "tempo" (nota aperta di
    FASE2_PIANO_TEMPO.md §8): errori di tracking (contenuto di un altro
    tick/luogo) vs errori di generazione (verbo/argomenti sbagliati),
    verificata su esiti costruiti a mano sopra `_STORIA_TEMPO` (5 tick,
    `storie_corte: {min: 5, max: 5}` così `n_tick` è deterministico)."""

    _CONFIG = {
        "stadi": {1: {"tipi": ["posizione_tempo", "azione_tempo", "azione_luogo"], "soglia": 0.95,
                      "storie_corte": {"min": 5, "max": 5}}},
        "dataset": {},
    }

    def _esegui(self, monkeypatch, record, risposte):
        monkeypatch.setattr(diagnosi_mod, "genera_storia", lambda seed, n_tick, persone: _STORIA_TEMPO)
        vocab = carica_vocabolario()
        modello = _ModelloMultiRisposta(vocab, [[*r, "[FINE]"] for r in risposte])
        return diagnosi_mod.esegui_diagnosi(
            modello, vocab, record, self._CONFIG, stadio=1, ctx=200, device="cpu",
        )

    def test_posizione_tempo_tick_vicino_e_distanza_coda(self, monkeypatch):
        # t=4 (oro salotto): il modello risponde cucina = posizione a t=3
        # -> "posizione_tick_vicino", distanza tick 1. t=2 (oro orto): esatto.
        dom4 = grafo_a_token(grafo_fatto("trovarsi", nsubj="anna", **{"obl:tempo": "quattro"}, quesito="dove"))
        ris4 = grafo_a_token(grafo_fatto("essere", nsubj="anna", **{"obl:luogo": "salotto", "obl:tempo": "quattro"}))
        gen4 = grafo_a_token(grafo_fatto("essere", nsubj="anna", **{"obl:luogo": "cucina", "obl:tempo": "quattro"}))
        dom2 = grafo_a_token(grafo_fatto("trovarsi", nsubj="anna", **{"obl:tempo": "due"}, quesito="dove"))
        ris2 = grafo_a_token(grafo_fatto("essere", nsubj="anna", **{"obl:luogo": "orto", "obl:tempo": "due"}))
        record = [
            {"seed": 300, "storia": [], "esempi": [{"tipo": "posizione_tempo", "domanda": dom4, "risposta": ris4}]},
            {"seed": 300, "storia": [], "esempi": [{"tipo": "posizione_tempo", "domanda": dom2, "risposta": ris2}]},
        ]
        esito = self._esegui(monkeypatch, record, [gen4, ris2])

        sezione = esito["tempo"]["posizione_tempo"]
        assert sezione["n_oro_noto"] == 2
        assert sezione["esattezza"] == pytest.approx(0.5)
        assert sezione["anatomia_errori"] == {"posizione_tick_vicino": 1}
        assert sezione["distanza_tick_generato"] == {"1-2": 1}
        # distanza dalla coda: t=4 -> n_tick-t=1 ("1-2", errore); t=2 -> 3 ("3-5", esatto)
        assert sezione["per_distanza_coda"]["1-2"] == {"esattezza": pytest.approx(0.0), "n": 1}
        assert sezione["per_distanza_coda"]["3-5"] == {"esattezza": pytest.approx(1.0), "n": 1}

    def test_azione_tempo_altro_tick_origine_verbo(self, monkeypatch):
        eventi = _STORIA_TEMPO.eventi
        # t=3: il modello genera il contenuto dell'evento di t=4 (con il
        # tempo della domanda) -> "evento_di_altro_tick", distanza 1.
        dom3 = grafo_a_token(grafo_fatto("fare", nsubj="anna", **{"obl:tempo": "tre"}, quesito="che-cosa"))
        ris3 = _grafo_evento_test(eventi[2])
        gen3 = grafo_a_token(grafo_fatto(
            "andare", nsubj="anna", **{"obl:origine": "cucina", "obl:luogo": "salotto", "obl:tempo": "tre"},
        ))
        # t=2: tutto giusto tranne obl:origine, duplicata sul luogo generato
        # -> "solo_origine_sbagliata" + origine_uguale_luogo_generato.
        dom2 = grafo_a_token(grafo_fatto("fare", nsubj="anna", **{"obl:tempo": "due"}, quesito="che-cosa"))
        ris2 = _grafo_evento_test(eventi[1])
        gen2 = grafo_a_token(grafo_fatto(
            "andare", nsubj="anna", **{"obl:origine": "orto", "obl:luogo": "orto", "obl:tempo": "due"},
        ))
        # t=5: verbo sbagliato ("dormire" invece di "andare").
        dom5 = grafo_a_token(grafo_fatto("fare", nsubj="anna", **{"obl:tempo": "cinque"}, quesito="che-cosa"))
        ris5 = _grafo_evento_test(eventi[4])
        gen5 = grafo_a_token(grafo_fatto("dormire", nsubj="anna", **{"obl:tempo": "cinque"}))
        record = [
            {"seed": 300, "storia": [], "esempi": [{"tipo": "azione_tempo", "domanda": dom3, "risposta": ris3}]},
            {"seed": 300, "storia": [], "esempi": [{"tipo": "azione_tempo", "domanda": dom2, "risposta": ris2}]},
            {"seed": 300, "storia": [], "esempi": [{"tipo": "azione_tempo", "domanda": dom5, "risposta": ris5}]},
        ]
        esito = self._esegui(monkeypatch, record, [gen3, gen2, gen5])

        sezione = esito["tempo"]["azione_tempo"]
        assert sezione["n_oro_noto"] == 3
        assert sezione["esattezza"] == pytest.approx(0.0)
        assert sezione["anatomia_errori"] == {
            "evento_di_altro_tick": 1, "solo_origine_sbagliata": 1, "verbo_sbagliato": 1,
        }
        assert sezione["distanza_tick_generato"] == {"1-2": 1}
        assert sezione["errori_verbo_giusto"] == 2  # gen3 e gen2 ("andare")
        assert sezione["origine_uguale_luogo_generato"] == 1
        # oro: tutti eventi copiabili (nessun "dorme" derivato), 4 archi ciascuno
        assert sezione["per_oro"]["evento"] == {"esattezza": pytest.approx(0.0), "n": 3}
        assert sezione["per_oro"]["dormire_derivato"] == {"esattezza": 0.0, "n": 0}
        assert sezione["per_n_archi_oro"] == {"4": {"esattezza": pytest.approx(0.0), "n": 3}}

    def test_azione_luogo_evento_di_altro_luogo(self, monkeypatch):
        eventi = _STORIA_TEMPO.eventi
        # luogo chiesto "orto" (oro = evento t=2 senza tempo): il modello
        # genera l'evento di t=3 senza tempo (luogo cucina != orto).
        dom = grafo_a_token(grafo_fatto("fare", nsubj="anna", **{"obl:luogo": "orto"}, quesito="che-cosa"))
        ris = _grafo_evento_test(eventi[1], con_tempo=False)
        gen = _grafo_evento_test(eventi[2], con_tempo=False)
        record = [{"seed": 300, "storia": [], "esempi": [{"tipo": "azione_luogo", "domanda": dom, "risposta": ris}]}]
        esito = self._esegui(monkeypatch, record, [gen])

        sezione = esito["tempo"]["azione_luogo"]
        assert sezione["n_oro_noto"] == 1
        assert sezione["anatomia_errori"] == {"evento_di_altro_luogo": 1}
        assert sezione["errori_verbo_giusto"] == 1
        assert sezione["luogo_richiesto_nel_generato"] == 0
        assert sezione["per_n_archi_oro"] == {"3": {"esattezza": pytest.approx(0.0), "n": 1}}
        # "azione_luogo" non ha metriche legate al tick
        assert "per_distanza_coda" not in sezione
        assert "per_oro" not in sezione

    def test_run_senza_tipi_tempo_sezione_vuota(self, monkeypatch):
        monkeypatch.setattr(diagnosi_mod, "genera_storia", lambda seed, n_tick, persone: _STORIE[100])
        vocab = carica_vocabolario()
        dom, ris = _domanda_e_risposta("piero", "giardino")
        record = [{"seed": 100, "storia": [], "esempi": [{"tipo": "posizione", "domanda": dom, "risposta": ris}]}]
        modello = _ModelloMultiRisposta(vocab, [[*ris, "[FINE]"]])
        config = {"stadi": {1: {"tipi": ["posizione"], "soglia": 0.95, "storie_corte": True}}, "dataset": {}}
        esito = diagnosi_mod.esegui_diagnosi(modello, vocab, record, config, stadio=1, ctx=200, device="cpu")
        assert esito["tempo"] == {}


@pytest.mark.torch
class TestSezioneStato:
    """Fase B §6.2: cuore metrico della diagnosi dei blocchi [STATO] (puro,
    nessun modello). posizioni_per_tick sintetico, blocchi generati a mano."""

    # anna: cucina, giardino, salotto, salotto ; piero: camera, camera, bosco, cucina
    POSIZIONI = {
        1: {"anna": "cucina", "piero": "camera"},
        2: {"anna": "giardino", "piero": "camera"},
        3: {"anna": "salotto", "piero": "bosco"},   # oro al tick 3
        4: {"anna": "salotto", "piero": "cucina"},
    }

    def _blocco(self, tick_lemma, coppie):
        tok = ["(", "obl:tempo", tick_lemma, ")"]
        for pid, luogo in coppie:
            tok += grafo_a_token(grafo_fatto("trovarsi", nsubj=pid, **{"obl:luogo": luogo}))
        return tok

    def test_leggi_blocco_generato_tollerante(self):
        from esami.diagnosi import _leggi_blocco_generato
        tick, pos = _leggi_blocco_generato(self._blocco("tre", [("anna", "salotto"), ("piero", "bosco")]))
        assert tick == "tre"
        assert pos == {"anna": "salotto", "piero": "bosco"}
        # token spuri prima/dopo: saltati, non fanno crashare
        tick2, pos2 = _leggi_blocco_generato(["[FINE]"] + self._blocco("due", [("anna", "cucina")]))
        assert tick2 == "due" and pos2 == {"anna": "cucina"}

    def test_anatomia_errori_stato(self):
        from esami.diagnosi import _diagnosi_blocco_stato, _nuovo_accumulatore_stato
        acc = _nuovo_accumulatore_stato()
        # blocco 1: anna->giardino (sua posizione al tick 2, vicino) = tick_vicino;
        #           piero->salotto (posizione dell'ALTRA persona anna al tick 3) = altra_persona
        _diagnosi_blocco_stato(acc, self.POSIZIONI, 3, 4, self._blocco("tre", [("anna", "giardino"), ("piero", "salotto")]))
        # blocco 2: anna->salotto (giusto); piero OMESSO = mancante
        _diagnosi_blocco_stato(acc, self.POSIZIONI, 3, 4, self._blocco("tre", [("anna", "salotto")]))
        # blocco 3: anna->camera (né sua vicina né di altri al tick 3) = altro; piero->bosco (giusto)
        _diagnosi_blocco_stato(acc, self.POSIZIONI, 3, 4, self._blocco("tre", [("anna", "camera"), ("piero", "bosco")]))

        assert acc["n_blocchi"] == 3
        assert acc["n_posizioni"] == 6
        assert acc["posizioni_esatte"] == 2
        assert acc["tick_esatti"] == 3  # etichetta "tre" sempre corretta
        assert dict(acc["anatomia_errori"]) == {"tick_vicino": 1, "altra_persona": 1, "mancante": 1, "altro": 1}
        # tick 3 su n_tick 4 -> distanza coda 1 -> bucket "1-2"
        assert acc["per_distanza_coda"]["1-2"]["n"] == 6
        assert acc["per_distanza_coda"]["1-2"]["esatto"] == 2

    def test_blocco_malformato_conta_mancanti(self):
        from esami.diagnosi import _diagnosi_blocco_stato, _nuovo_accumulatore_stato
        acc = _nuovo_accumulatore_stato()
        _diagnosi_blocco_stato(acc, self.POSIZIONI, 3, 4, ["(", "obl:tempo", "tre", ")"])  # nessuna posizione
        assert acc["malformati"] == 1
        assert dict(acc["anatomia_errori"]) == {"mancante": 2}  # anna e piero entrambe assenti
        assert acc["blocchi_esatti"] == 0
