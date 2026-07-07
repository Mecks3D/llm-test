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
