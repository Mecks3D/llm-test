"""Test del modulo esami/ (fasi/FASE2_PIANO.md)."""
from __future__ import annotations

import json

import pytest

from cervello.sequenza import componi_esempio
from esami.genera import (
    _n_tick,
    _verifica_seed,
    carica_config,
    finestra_seed,
    genera_dataset,
    genera_record,
    percorso_dataset,
    scrivi_dataset,
)


def _config_piccolo(tmp_path, **override_dataset):
    dataset = {
        "ctx": 3072,
        "train_storie": 4,
        "dev_storie": 3,
        "esame_storie": 3,
        "n_per_tipo": 8,
    }
    dataset.update(override_dataset)
    return {
        "percorsi": {"dati_dir": str(tmp_path / "dati")},
        "dataset": dataset,
        "stadi": {
            1: {"tipi": ["posizione"], "soglia": 0.95, "storie_corte": True},
            3: {"tipi": ["possesso", "conteggio", "transfer", "parentela"], "soglia": 0.90, "storie_corte": False},
        },
    }


# ---------------------------------------------------------------------------
# Gruppo 3: esami/genera.py
# ---------------------------------------------------------------------------

class TestFinestreSeed:
    def test_finestre_conformi_alla_tabella(self, tmp_path):
        config = _config_piccolo(tmp_path)
        assert finestra_seed(1, "train", config) == range(0, 4)
        assert finestra_seed(2, "train", config) == range(100_000, 100_004)
        assert finestra_seed(1, "dev", config) == range(800_000, 800_003)
        assert finestra_seed(2, "dev", config) == range(810_000, 810_003)
        assert finestra_seed(1, "esame", config) == range(1_000_000, 1_000_003)
        assert finestra_seed(2, "esame", config) == range(1_010_000, 1_010_003)

    def test_split_sconosciuto_solleva(self, tmp_path):
        config = _config_piccolo(tmp_path)
        with pytest.raises(ValueError):
            finestra_seed(1, "boh", config)


class TestVerificaSeed:
    def test_train_rifiuta_seed_esame(self):
        with pytest.raises(ValueError):
            _verifica_seed(1_000_000, "train")

    def test_dev_rifiuta_seed_esame(self):
        with pytest.raises(ValueError):
            _verifica_seed(1_000_000, "dev")

    def test_esame_rifiuta_seed_sotto_soglia(self):
        with pytest.raises(ValueError):
            _verifica_seed(999_999, "esame")

    def test_seed_validi_non_sollevano(self):
        _verifica_seed(0, "train")
        _verifica_seed(999_999, "dev")
        _verifica_seed(1_000_000, "esame")


class TestRifiutoSeed:
    def test_train_con_seed_riservato_non_scrive_file(self, tmp_path):
        # stadio 11: inizio finestra train = 100_000*10 = 1.000.000 (riservato)
        config = _config_piccolo(tmp_path)
        config["stadi"][11] = {"tipi": ["posizione"], "soglia": 0.95, "storie_corte": True}
        with pytest.raises(ValueError, match="riservat"):
            scrivi_dataset(11, "train", config)
        assert not percorso_dataset(11, "train", config).exists()

    def test_esame_con_seed_sotto_soglia_non_scrive_file(self, tmp_path):
        config = _config_piccolo(tmp_path, esame_storie=3)
        # stadio 0 fittizio: inizio finestra esame = 1.000.000 + 10.000*(-1) < 1.000.000
        config["stadi"][0] = {"tipi": ["posizione"], "soglia": 0.95, "storie_corte": True}
        with pytest.raises(ValueError, match="riservat"):
            scrivi_dataset(0, "esame", config)
        assert not percorso_dataset(0, "esame", config).exists()


class TestStadio1StorieCorteSolaPosizione:
    def test_n_tick_in_3_6(self, tmp_path):
        config = _config_piccolo(tmp_path)
        for seed in range(20):
            n = _n_tick(1, seed, config)
            assert 3 <= n <= 6

    def test_solo_domande_posizione(self, tmp_path):
        config = _config_piccolo(tmp_path)
        for seed in range(10):
            record = genera_record(1, seed, config)
            for esempio in record["esempi"]:
                assert esempio["tipo"] == "posizione"


class TestSequenzaComposta:
    def test_ogni_composto_entro_ctx(self, tmp_path):
        config = _config_piccolo(tmp_path, ctx=3072)
        for stadio in (1, 3):
            for seed in finestra_seed(stadio, "train", config):
                record = genera_record(stadio, seed, config)
                for esempio in record["esempi"]:
                    composto = componi_esempio(
                        [record["storia"]], esempio["domanda"], esempio["risposta"]
                    )
                    assert len(composto) <= config["dataset"]["ctx"]

    def test_ctx_troppo_piccolo_solleva(self, tmp_path):
        config = _config_piccolo(tmp_path, ctx=5)
        with pytest.raises(ValueError, match="ctx"):
            genera_dataset(1, "train", config)


class TestDeterminismo:
    def test_generazione_byte_identica(self, tmp_path):
        config1 = _config_piccolo(tmp_path / "a")
        config2 = _config_piccolo(tmp_path / "b")
        p1 = scrivi_dataset(1, "train", config1)
        p2 = scrivi_dataset(1, "train", config2)
        assert p1.read_bytes() == p2.read_bytes()

    def test_riscrittura_byte_identica(self, tmp_path):
        config = _config_piccolo(tmp_path)
        p = scrivi_dataset(1, "train", config)
        contenuto1 = p.read_bytes()
        p2 = scrivi_dataset(1, "train", config)
        assert p2.read_bytes() == contenuto1


class TestCaricaConfig:
    def test_config_reale_si_carica(self):
        config = carica_config()
        assert config["dataset"]["ctx"] == 3072
        assert config["stadi"][1]["tipi"] == ["posizione"]
        assert config["stadi"][3]["soglia"] == 0.90


# ---------------------------------------------------------------------------
# Gruppo 7: esami/esamina.py — decodifica greedy + confronto grafo vs grafo
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:
    torch = None

from cervello.vocabolario import carica_vocabolario
from cervello.sequenza import grafo_a_token
from mondo.grafo import NON_LO_SO, grafo_fatto


class _ModelloIniettato:
    """Finto modello per `decodifica_greedy`: ad ogni chiamata restituisce
    logits il cui argmax riproduce, in ordine, `sequenza_output` (una lista
    di token già nel vocabolario). Permette di testare esamina.py senza
    addestrare nulla (FASE2_PIANO.md §9.7: "con risposte iniettate")."""

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
            self._chiamate = 0  # nuova sessione di decodifica (nuovo esempio)
        self._T_precedente = T
        idx = min(self._chiamate, len(self._ids_output) - 1)
        prossimo_id = self._ids_output[idx]
        self._chiamate += 1
        logits = torch.full((B, T, self._vocab_size), -10.0)
        logits[0, -1, prossimo_id] = 10.0
        return logits


@pytest.mark.torch
class TestCategoria:
    def _valuta_con_iniezione(self, oro_grafo, sequenza_generata, tipo="posizione"):
        from esami.esamina import valuta_esempio

        vocab = carica_vocabolario()
        modello = _ModelloIniettato(vocab, [*sequenza_generata, "[FINE]"])
        esempio = {"tipo": tipo, "domanda": ["(", "trovarsi", "(", "quesito", "dove", ")", ")"],
                   "risposta": grafo_a_token(oro_grafo)}
        return valuta_esempio(modello, vocab, storia_flat=[], esempio=esempio, ctx=200, device="cpu")

    def test_esatto(self):
        oro = grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "cucina"})
        esito = self._valuta_con_iniezione(oro, grafo_a_token(oro))
        assert esito.categoria == "esatto"
        assert esito.esatto is True

    def test_invenzione(self):
        # oro = non-lo-so, il modello risponde con un fatto inventato
        generato = grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "cucina"})
        esito = self._valuta_con_iniezione(NON_LO_SO, grafo_a_token(generato))
        assert esito.categoria == "invenzione"
        assert esito.esatto is False

    def test_astensione_errata(self):
        # oro determinabile, il modello risponde non-lo-so
        oro = grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "cucina"})
        esito = self._valuta_con_iniezione(oro, grafo_a_token(NON_LO_SO))
        assert esito.categoria == "astensione_errata"
        assert esito.esatto is False

    def test_malformata(self):
        oro = grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "cucina"})
        # sequenza sbilanciata: non ricostruisce un grafo valido
        esito = self._valuta_con_iniezione(oro, ["(", "essere", "(", "nsubj", "sara"])
        assert esito.categoria == "malformata"
        assert esito.esatto is False

    def test_errore_generico(self):
        # oro determinabile, il modello risponde con un fatto diverso ma valido
        oro = grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "cucina"})
        generato = grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "giardino"})
        esito = self._valuta_con_iniezione(oro, grafo_a_token(generato))
        assert esito.categoria == "errore"
        assert esito.esatto is False


@pytest.mark.torch
class TestValutaDataset:
    def test_conteggi_e_esattezza(self):
        from esami.esamina import valuta_dataset

        vocab = carica_vocabolario()
        oro_esatto = grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "cucina"})
        oro_determinabile = grafo_fatto("essere", nsubj="piero", **{"obl:luogo": "orto"})
        generato_sbagliato = grafo_fatto("essere", nsubj="piero", **{"obl:luogo": "cucina"})

        def _record(tipo, oro, sequenza_generata):
            return {
                "storia": [],
                "esempi": [{
                    "tipo": tipo,
                    "domanda": ["(", "trovarsi", "(", "quesito", "dove", ")", ")"],
                    "risposta": grafo_a_token(oro),
                }],
            }, [*sequenza_generata, "[FINE]"]

        # 4 esempi: 1 esatto, 1 invenzione, 1 astensione_errata, 1 errore
        casi = [
            _record("posizione", oro_esatto, grafo_a_token(oro_esatto)),
            _record("posizione", NON_LO_SO, grafo_a_token(oro_esatto)),
            _record("possesso", oro_determinabile, grafo_a_token(NON_LO_SO)),
            _record("possesso", oro_determinabile, grafo_a_token(generato_sbagliato)),
        ]

        # ogni caso ha una risposta iniettata diversa: serve uno stub per
        # caso. L'aggregazione multi-esempio con un unico modello è testata
        # a parte in test_valuta_dataset_stesso_modello_su_piu_esempi_identici.
        from esami.esamina import valuta_esempio

        totali = {"esatto": 0, "invenzione": 0, "astensione_errata": 0, "malformata": 0, "errore": 0}
        per_tipo = {}
        for record, sequenza in casi:
            modello = _ModelloIniettato(vocab, sequenza)
            esempio = record["esempi"][0]
            esito = valuta_esempio(modello, vocab, record["storia"], esempio, ctx=200, device="cpu")
            totali[esito.categoria] += 1
            per_tipo.setdefault(esito.tipo, {"esatto": 0, "n": 0})
            per_tipo[esito.tipo]["n"] += 1
            if esito.esatto:
                per_tipo[esito.tipo]["esatto"] += 1

        assert totali == {"esatto": 1, "invenzione": 1, "astensione_errata": 1, "malformata": 0, "errore": 1}
        assert per_tipo["posizione"] == {"esatto": 1, "n": 2}
        assert per_tipo["possesso"] == {"esatto": 0, "n": 2}

    def test_valuta_dataset_stesso_modello_su_piu_esempi_identici(self):
        # con un unico modello iniettato che risponde sempre correttamente
        # allo stesso identico esempio ripetuto, valuta_dataset deve dare
        # esattezza 1.0 (verifica l'aggregazione reale, non solo per-esempio).
        from esami.esamina import valuta_dataset

        vocab = carica_vocabolario()
        oro = grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "cucina"})
        modello = _ModelloIniettato(vocab, [*grafo_a_token(oro), "[FINE]"])
        record = [{
            "storia": [],
            "esempi": [
                {"tipo": "posizione", "domanda": ["(", "trovarsi", "(", "quesito", "dove", ")", ")"],
                 "risposta": grafo_a_token(oro)}
                for _ in range(3)
            ],
        }]
        esito = valuta_dataset(modello, vocab, record, ctx=200, device="cpu")
        assert esito["n_esempi"] == 3
        assert esito["esattezza"] == 1.0
        assert esito["conteggi"]["esatto"] == 3
        assert esito["conteggi"]["malformata"] == 0
        assert esito["campioni_non_esatti"] == []  # tutte esatte: niente campioni

    def test_campioni_non_esatti_riportati_per_diagnosi(self):
        # un modello che risponde sempre la stessa cosa sbagliata: il JSON
        # deve riportare i token generati e l'oro dei primi casi non esatti.
        from esami.esamina import valuta_dataset

        vocab = carica_vocabolario()
        oro = grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "cucina"})
        generato = grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "giardino"})
        modello = _ModelloIniettato(vocab, [*grafo_a_token(generato), "[FINE]"])
        record = [{
            "storia": [],
            "esempi": [
                {"tipo": "posizione", "domanda": ["(", "trovarsi", "(", "quesito", "dove", ")", ")"],
                 "risposta": grafo_a_token(oro)}
                for _ in range(3)
            ],
        }]
        esito = valuta_dataset(modello, vocab, record, ctx=200, device="cpu")
        assert esito["esattezza"] == 0.0
        assert len(esito["campioni_non_esatti"]) == 3
        campione = esito["campioni_non_esatti"][0]
        assert campione["categoria"] == "errore"
        assert campione["generato"] == grafo_a_token(generato)
        assert campione["oro"] == grafo_a_token(oro)


@pytest.mark.torch
class TestDecodificaGreedy:
    def test_si_ferma_a_fine(self):
        from esami.esamina import decodifica_greedy

        vocab = carica_vocabolario()
        modello = _ModelloIniettato(vocab, ["(", "non-lo-so", ")", "[FINE]"])
        prefisso = [vocab.id(t) for t in ["[STORIA]", "[DOMANDA]", "[RISPOSTA]"]]
        generati = decodifica_greedy(modello, vocab, prefisso, ctx=200, device="cpu")
        token_generati = [vocab.token(i) for i in generati]
        assert token_generati == ["(", "non-lo-so", ")", "[FINE]"]

    def test_si_ferma_al_tetto_ctx(self):
        from esami.esamina import decodifica_greedy

        vocab = carica_vocabolario()
        # sequenza che non emette mai [FINE]
        modello = _ModelloIniettato(vocab, ["(", "non-lo-so"] * 50)
        prefisso = [vocab.id(t) for t in ["[STORIA]", "[DOMANDA]", "[RISPOSTA]"]]
        generati = decodifica_greedy(modello, vocab, prefisso, ctx=10, device="cpu")
        assert len(prefisso) + len(generati) == 10
