"""Test del modulo esami/ (fasi/FASE2_PIANO.md)."""
from __future__ import annotations

import json

import pytest

from cervello.sequenza import componi_esempio
from mondo import dati_mondo as dm
from esami.genera import (
    _cast_persone,
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

    def test_storie_corte_dict_min_max_personalizzato(self, tmp_path):
        config = _config_piccolo(tmp_path)
        config["stadi"][1]["storie_corte"] = {"min": 1, "max": 3}
        for seed in range(30):
            n = _n_tick(1, seed, config)
            assert 1 <= n <= 3

    def test_storie_corte_true_resta_3_6_byte_identico(self, tmp_path):
        config_bool = _config_piccolo(tmp_path)
        config_dict = _config_piccolo(tmp_path)
        config_dict["stadi"][1]["storie_corte"] = {"min": 3, "max": 6}
        for seed in range(30):
            assert _n_tick(1, seed, config_bool) == _n_tick(1, seed, config_dict)


class TestCastPersone:
    def test_assente_ritorna_none(self, tmp_path):
        config = _config_piccolo(tmp_path)
        assert _cast_persone(config) is None

    def test_cast_valido_filtra_preservando_ordine(self, tmp_path):
        config = _config_piccolo(tmp_path, cast=["maria", "anna"])
        persone = _cast_persone(config)
        assert [p.id for p in persone] == ["anna", "maria"]  # ordine di dm.PERSONE

    def test_cast_con_id_sconosciuto_solleva(self, tmp_path):
        config = _config_piccolo(tmp_path, cast=["anna", "pinco"])
        with pytest.raises(ValueError, match="pinco"):
            _cast_persone(config)

    def test_genera_record_rispetta_il_cast(self, tmp_path):
        cast_ids = {"anna", "piero", "maria"}
        config = _config_piccolo(tmp_path, cast=sorted(cast_ids))
        for seed in range(10):
            record = genera_record(1, seed, config)
            entita_storia = {tok for tok in record["storia"] if tok in {p.id for p in dm.PERSONE}}
            assert entita_storia <= cast_ids


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
# Gruppo 8: esperimento anti-scorciatoia
# (fasi/FASE2_PIANO_ANTISCORCIATOIA.md §2-§4, §6)
# ---------------------------------------------------------------------------

from mondo.domande import Domanda
from mondo.grafo import NON_LO_SO, grafo_fatto
from mondo.simulatore import Storia
from mondo.tipi import Evento, StatoMondo
from esami.genera import (
    TIPI_TEMPO,
    _cast_per_seed,
    _classifica_domanda_posizione,
    _record_per_seed,
    _seleziona_posizione,
    _tracking_puro,
    _tracking_puro_tempo,
    genera_esame_tracking,
    genera_tracking_tempo,
    percorso_esame_tracking,
    percorso_tracking_tempo,
    scrivi_esame_tracking,
    scrivi_tracking_tempo,
)


def _stato_con_persone(*id_persone: str) -> StatoMondo:
    return StatoMondo(t=0, luoghi={}, collegamenti={}, persone={p: None for p in id_persone}, oggetti={}, risorse={})


def _storia_sintetica(eventi: list[Evento], persone: tuple[str, ...] = ("anna", "piero", "maria")) -> Storia:
    return Storia(seed=0, eventi=tuple(eventi), stato_finale=_stato_con_persone(*persone))


def _domanda_posizione(bersaglio: str, oro_luogo: str | None) -> Domanda:
    grafo_domanda = grafo_fatto("trovarsi", nsubj=bersaglio, quesito="dove")
    if oro_luogo is None:
        risposta = NON_LO_SO
    else:
        risposta = grafo_fatto("essere", nsubj=bersaglio, **{"obl:luogo": oro_luogo})
    return Domanda("posizione", grafo_domanda, risposta)


class TestClassificazioneDifficolta:
    def test_non_lo_so(self):
        storia = _storia_sintetica([Evento(t=1, azione="andare", agente="anna", luogo="cucina")])
        domanda = _domanda_posizione("piero", None)
        assert _classifica_domanda_posizione(storia, domanda) == "non-lo-so"

    def test_d1_oro_diverso_dal_luogo_piu_frequente(self):
        eventi = [
            Evento(t=1, azione="andare", agente="anna", luogo="cucina"),
            Evento(t=2, azione="andare", agente="maria", luogo="cucina"),
            Evento(t=3, azione="andare", agente="piero", luogo="giardino"),
            Evento(t=4, azione="andare", agente="anna", luogo="cucina"),
        ]
        storia = _storia_sintetica(eventi)
        domanda = _domanda_posizione("piero", "giardino")
        assert _classifica_domanda_posizione(storia, domanda) == "difficile"

    def test_d2_distanza_coda_lunga(self):
        eventi = [
            Evento(t=1, azione="andare", agente="piero", luogo="giardino"),
            Evento(t=2, azione="andare", agente="anna", luogo="giardino"),
            Evento(t=3, azione="andare", agente="anna", luogo="giardino"),
            Evento(t=4, azione="andare", agente="anna", luogo="giardino"),
            Evento(t=5, azione="andare", agente="anna", luogo="giardino"),
        ]
        storia = _storia_sintetica(eventi)
        domanda = _domanda_posizione("piero", "giardino")
        assert _classifica_domanda_posizione(storia, domanda) == "difficile"

    def test_d3_oggetto_trasportato_senza_ricomenzione(self):
        eventi = [
            Evento(t=1, azione="prendere", agente="anna", oggetto="palla", luogo="cucina"),
            Evento(t=2, azione="andare", agente="anna", luogo="salotto", luogo_origine="cucina"),
        ]
        storia = _storia_sintetica(eventi)
        domanda = _domanda_posizione("palla", "salotto")
        assert _classifica_domanda_posizione(storia, domanda) == "difficile"

    def test_facile(self):
        eventi = [
            Evento(t=1, azione="andare", agente="anna", luogo="cucina"),
            Evento(t=2, azione="andare", agente="maria", luogo="cucina"),
            Evento(t=3, azione="andare", agente="piero", luogo="cucina"),
        ]
        storia = _storia_sintetica(eventi)
        domanda = _domanda_posizione("piero", "cucina")
        assert _classifica_domanda_posizione(storia, domanda) == "facile"

    def test_cercare_non_conta_come_menzione_di_un_oggetto(self):
        # "cercare X" dice che X NON è lì: non stabilisce la posizione, quindi
        # non deve contare come "ultima menzione" (stessa esclusione di
        # mondo/domande.py::_oggetti_con_posizione_nota).
        eventi = [
            Evento(t=1, azione="prendere", agente="anna", oggetto="palla", luogo="cucina"),
            Evento(t=2, azione="andare", agente="anna", luogo="salotto", luogo_origine="cucina"),
            Evento(t=3, azione="cercare", agente="piero", oggetto="palla", luogo="giardino"),
        ]
        storia = _storia_sintetica(eventi)
        domanda = _domanda_posizione("palla", "salotto")
        # l'ultima menzione VALIDA resta t=1 (prendere), non t=3 (cercare):
        # stessa distanza_coda e stesso esito di test_d3 sopra.
        assert _classifica_domanda_posizione(storia, domanda) == "difficile"


# ---------------------------------------------------------------------------
# Gruppo 9: A3, split "tracking puro" (fasi/FASE2_PIANO_DIAGNOSI.md §2)
# ---------------------------------------------------------------------------

class TestTrackingPuro:
    def test_non_lo_so_falso(self):
        storia = _storia_sintetica([Evento(t=1, azione="andare", agente="anna", luogo="cucina")])
        domanda = _domanda_posizione("piero", None)
        assert _tracking_puro(storia, domanda) is False

    def test_solo_d1_vero_non_basta(self):
        # stesso caso di test_d1_oro_diverso_dal_luogo_piu_frequente: D1 vero,
        # ma D2 falso (distanza_coda=1) e D3 falso (oro == luogo dell'ultima
        # menzione) -> non è tracking puro, serve la congiunzione.
        eventi = [
            Evento(t=1, azione="andare", agente="anna", luogo="cucina"),
            Evento(t=2, azione="andare", agente="maria", luogo="cucina"),
            Evento(t=3, azione="andare", agente="piero", luogo="giardino"),
            Evento(t=4, azione="andare", agente="anna", luogo="cucina"),
        ]
        storia = _storia_sintetica(eventi)
        domanda = _domanda_posizione("piero", "giardino")
        assert _classifica_domanda_posizione(storia, domanda) == "difficile"  # almeno una vera
        assert _tracking_puro(storia, domanda) is False  # non tutte e tre

    def test_d1_e_d2_e_d3_tutte_vere(self):
        eventi = [
            Evento(t=1, azione="andare", agente="piero", luogo="giardino"),
            Evento(t=2, azione="andare", agente="anna", luogo="cucina"),
            Evento(t=3, azione="andare", agente="maria", luogo="cucina"),
            Evento(t=4, azione="andare", agente="anna", luogo="cucina"),
            Evento(t=5, azione="andare", agente="maria", luogo="cucina"),
        ]
        storia = _storia_sintetica(eventi)
        # oro disgiunto sia dal luogo più frequente (cucina) sia dal luogo
        # dell'ultima menzione di piero (giardino, t=1): D1, D2 (distanza=4),
        # D3 tutte vere.
        domanda = _domanda_posizione("piero", "orto")
        assert _tracking_puro(storia, domanda) is True


class TestEsameTracking:
    def test_ogni_domanda_e_gia_nellesame_ufficiale(self, tmp_path):
        config = _config_piccolo(tmp_path, esame_storie=30)
        tracking = genera_esame_tracking(1, config)
        assert tracking  # il campione di prova produce almeno una storia utile

        esame = {r["seed"]: r for r in genera_dataset(1, "esame", config)}
        for r in tracking:
            assert r["seed"] in esame
            assert r["storia"] == esame[r["seed"]]["storia"]
            esempi_ufficiali = esame[r["seed"]]["esempi"]
            for esempio in r["esempi"]:
                assert esempio["tipo"] == "posizione"
                assert esempio in esempi_ufficiali

    def test_seed_solo_dalla_finestra_esame(self, tmp_path):
        config = _config_piccolo(tmp_path, esame_storie=30)
        finestra = set(finestra_seed(1, "esame", config))
        for r in genera_esame_tracking(1, config):
            assert r["seed"] in finestra

    def test_tipo_non_ammesso_ritorna_vuoto(self, tmp_path):
        config = _config_piccolo(tmp_path, esame_storie=10)
        config["stadi"][1]["tipi"] = []
        assert genera_esame_tracking(1, config) == []

    def test_scrittura_deterministica(self, tmp_path):
        config = _config_piccolo(tmp_path, esame_storie=30)
        p1 = scrivi_esame_tracking(1, config)
        contenuto1 = p1.read_bytes()
        assert p1 == percorso_esame_tracking(1, config)
        p2 = scrivi_esame_tracking(1, config)
        assert p2.read_bytes() == contenuto1


# ---------------------------------------------------------------------------
# Esperimento "tempo" (fasi/FASE2_PIANO_TEMPO.md §4): cast rotante,
# integrazione in genera_record, split diagnostico tracking_tempo.jsonl.
# ---------------------------------------------------------------------------

def _config_piccolo_tempo(tmp_path, **override_dataset):
    override_dataset.setdefault("cast_rotante", True)
    config = _config_piccolo(tmp_path, **override_dataset)
    config["stadi"][1]["tipi"] = ["posizione", "posizione_tempo", "azione_tempo", "azione_luogo"]
    config["stadi"][1]["storie_corte"] = {"min": 12, "max": 24}
    return config


class TestCastRotante:
    def test_assente_come_prima_byte_identico(self, tmp_path):
        config_vecchio = _config_piccolo(tmp_path / "vecchio")
        config_nuovo = _config_piccolo(tmp_path / "nuovo")  # niente cast_rotante
        for seed in range(10):
            assert _cast_per_seed(config_nuovo, seed) == _cast_persone(config_nuovo)
            r1 = genera_record(1, seed, config_vecchio)
            r2 = genera_record(1, seed, config_nuovo)
            assert r1 == r2

    def test_rotazione_deterministica_su_seed_mod_6(self, tmp_path):
        config = _config_piccolo_tempo(tmp_path)
        for seed in range(12):
            cast = _cast_per_seed(config, seed)
            assert len(cast) == 1
            assert cast[0] == dm.PERSONE[seed % len(dm.PERSONE)]

    def test_cast_e_cast_rotante_insieme_solleva(self, tmp_path):
        config = _config_piccolo(tmp_path, cast=["anna"], cast_rotante=True)
        with pytest.raises(ValueError, match="cast_rotante"):
            _cast_per_seed(config, 0)

    def test_genera_record_rispetta_il_cast_rotante(self, tmp_path):
        config = _config_piccolo_tempo(tmp_path)
        for seed in range(8):
            record = genera_record(1, seed, config)
            entita_storia = {tok for tok in record["storia"] if tok in {p.id for p in dm.PERSONE}}
            atteso = {dm.PERSONE[seed % len(dm.PERSONE)].id}
            assert entita_storia <= atteso


class TestIntegrazioneDomandeTempo:
    def test_tipi_tempo_compaiono_quando_ammessi(self, tmp_path):
        config = _config_piccolo_tempo(tmp_path)
        tipi_visti: set[str] = set()
        for seed in range(15):
            record = genera_record(1, seed, config)
            tipi_visti |= {e["tipo"] for e in record["esempi"]}
        assert tipi_visti & TIPI_TEMPO

    def test_rng_separato_non_cambia_le_estrazioni_esistenti(self, tmp_path):
        config_solo_posizione = _config_piccolo_tempo(tmp_path / "a", n_per_tipo=6)
        config_solo_posizione["stadi"][1]["tipi"] = ["posizione"]
        config_con_tempo = _config_piccolo_tempo(tmp_path / "b", n_per_tipo=6)
        for seed in range(10):
            r1 = genera_record(1, seed, config_solo_posizione)
            r2 = genera_record(1, seed, config_con_tempo)
            esempi_posizione_2 = [e for e in r2["esempi"] if e["tipo"] == "posizione"]
            assert r1["esempi"] == esempi_posizione_2
            assert r1["storia"] == r2["storia"]

    def test_tipi_tempo_assenti_se_non_ammessi(self, tmp_path):
        config = _config_piccolo_tempo(tmp_path)
        config["stadi"][1]["tipi"] = ["posizione"]
        for seed in range(10):
            record = genera_record(1, seed, config)
            assert not ({e["tipo"] for e in record["esempi"]} & TIPI_TEMPO)


class TestTrackingTempo:
    def test_ogni_domanda_e_gia_nellesame_ufficiale(self, tmp_path):
        config = _config_piccolo_tempo(tmp_path, esame_storie=40)
        tracking = genera_tracking_tempo(1, config)
        assert tracking  # il campione di prova produce almeno una storia utile

        esame = {r["seed"]: r for r in genera_dataset(1, "esame", config)}
        for r in tracking:
            assert r["seed"] in esame
            assert r["storia"] == esame[r["seed"]]["storia"]
            esempi_ufficiali = esame[r["seed"]]["esempi"]
            for esempio in r["esempi"]:
                assert esempio["tipo"] == "posizione_tempo"
                assert esempio in esempi_ufficiali

    def test_seed_solo_dalla_finestra_esame(self, tmp_path):
        config = _config_piccolo_tempo(tmp_path, esame_storie=40)
        finestra = set(finestra_seed(1, "esame", config))
        for r in genera_tracking_tempo(1, config):
            assert r["seed"] in finestra

    def test_tipo_non_ammesso_ritorna_vuoto(self, tmp_path):
        config = _config_piccolo_tempo(tmp_path, esame_storie=10)
        config["stadi"][1]["tipi"] = ["posizione"]
        assert genera_tracking_tempo(1, config) == []

    def test_cast_non_rotante_solleva(self, tmp_path):
        config = _config_piccolo_tempo(tmp_path, esame_storie=10)
        del config["dataset"]["cast_rotante"]
        with pytest.raises(ValueError, match="cast_rotante"):
            genera_tracking_tempo(1, config)

    def test_scrittura_deterministica(self, tmp_path):
        config = _config_piccolo_tempo(tmp_path, esame_storie=40)
        p1 = scrivi_tracking_tempo(1, config)
        contenuto1 = p1.read_bytes()
        assert p1 == percorso_tracking_tempo(1, config)
        p2 = scrivi_tracking_tempo(1, config)
        assert p2.read_bytes() == contenuto1


class TestSelezioneAntiScorciatoia:
    def test_quota_difficili_e_non_lo_so_rispettate_su_storie_reali(self, tmp_path):
        config = _config_piccolo(
            tmp_path, train_storie=50,
            anti_scorciatoia={"quota_difficili": 0.6, "candidate_per_tipo": 999},
        )
        conteggi = {"difficile": 0, "facile": 0, "non-lo-so": 0}
        for seed in finestra_seed(1, "train", config):
            record = genera_record(1, seed, config)
            for esempio in record["esempi"]:
                assert "difficolta" in esempio
                conteggi[esempio["difficolta"]] += 1

        totale = sum(conteggi.values())
        noti = conteggi["difficile"] + conteggi["facile"]
        assert conteggi["difficile"] / noti >= 0.6 - 0.15  # tolleranza: storie povere di candidate
        assert 0.10 <= conteggi["non-lo-so"] / totale <= 0.30

    def test_senza_anti_scorciatoia_niente_campo_difficolta(self, tmp_path):
        config = _config_piccolo(tmp_path)
        record = genera_record(1, 0, config)
        for esempio in record["esempi"]:
            assert "difficolta" not in esempio

    def test_dev_ed_esame_non_toccati_dalla_selezione(self, tmp_path):
        config_base = _config_piccolo(tmp_path / "base")
        config_anti = _config_piccolo(
            tmp_path / "anti",
            anti_scorciatoia={"quota_difficili": 0.6, "candidate_per_tipo": 999},
            troncamenti=True,
        )
        for split in ("dev", "esame"):
            record_base = genera_dataset(1, split, config_base)
            record_anti = genera_dataset(1, split, config_anti)
            assert record_base == record_anti


class TestStorieTroncate:
    def test_solo_record_pieno_senza_troncamenti(self, tmp_path):
        config = _config_piccolo(tmp_path)
        for seed in range(5):
            assert len(_record_per_seed(1, seed, config, "train")) == 1

    def test_conta_i_record_attesi(self, tmp_path):
        config = _config_piccolo(tmp_path, troncamenti=True)
        for seed in range(20):
            n_tick_pieno = _n_tick(1, seed, config)
            record = _record_per_seed(1, seed, config, "train")
            assert len(record) == max(1, n_tick_pieno - 3 + 1)
            attesi = {None} | set(range(3, n_tick_pieno))
            assert {r.get("troncamento") for r in record} == attesi

    def test_eventi_troncati_sono_il_prefisso_esatto(self, tmp_path):
        # round-trip prefisso già verificato nel piano su 180/180 semi: qui
        # si controlla che il campo "storia" (token-evento) della versione
        # troncata sia esattamente il prefisso di quella piena.
        config = _config_piccolo(tmp_path, troncamenti=True)
        for seed in range(10):
            record = _record_per_seed(1, seed, config, "train")
            pieno = next(r for r in record if r.get("troncamento") is None)
            for r in record:
                if r.get("troncamento") is None:
                    continue
                assert pieno["storia"][: len(r["storia"])] == r["storia"]

    def test_dev_ed_esame_ignorano_troncamenti(self, tmp_path):
        config_base = _config_piccolo(tmp_path / "base")
        config_tr = _config_piccolo(tmp_path / "tr", troncamenti=True)
        for split in ("dev", "esame"):
            assert genera_dataset(1, split, config_base) == genera_dataset(1, split, config_tr)

    def test_oro_puo_cambiare_tra_troncamenti(self, tmp_path):
        # cerca programmaticamente un seed dove la stessa entita ha oro
        # diverso in due troncamenti diversi (piano §6.4d).
        config = _config_piccolo(tmp_path, troncamenti=True)
        trovato = False
        for seed in range(200):
            record = _record_per_seed(1, seed, config, "train")
            per_troncamento: dict[int, dict[str, str | None]] = {}
            for r in record:
                k = r.get("troncamento", -1)
                oro_per_entita: dict[str, str | None] = {}
                for esempio in r["esempi"]:
                    bersaglio = _lemma_bersaglio(esempio["domanda"])
                    oro_per_entita[bersaglio] = _lemma_oro(esempio["risposta"])
                per_troncamento[k] = oro_per_entita
            entita_comuni = set.intersection(*(set(d) for d in per_troncamento.values())) if per_troncamento else set()
            for entita in entita_comuni:
                valori = {d[entita] for d in per_troncamento.values() if entita in d}
                if len(valori) > 1:
                    trovato = True
                    break
            if trovato:
                break
        assert trovato, "nessun seed nei primi 200 mostra un oro che cambia tra troncamenti"


def _entita_dopo(token: list[str], relazione: str) -> tuple[str, ...]:
    """Estrae lo span di lemmi (1 o 2, per le istanze lemma+ordinale) tra la
    relazione data e la parentesi chiusa che la segue."""
    idx = token.index(relazione)
    fine = token.index(")", idx)
    return tuple(token[idx + 1 : fine])


def _lemma_bersaglio(tok_domanda: list[str]) -> tuple[str, ...]:
    # domanda = ( trovarsi ( nsubj BERSAGLIO ) ( quesito dove ) )
    return _entita_dopo(tok_domanda, "nsubj")


def _lemma_oro(tok_risposta: list[str]) -> tuple[str, ...] | None:
    if tok_risposta[:2] == ["(", "non-lo-so"]:
        return None
    return _entita_dopo(tok_risposta, "obl:luogo")


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


# ---------------------------------------------------------------------------
# Fase B T3: generazione dello stato-oro (fasi/FASE2_PIANO_STATO.md §4)
# ---------------------------------------------------------------------------

class TestStatoOro:
    def test_blocchi_stato_oro_coincidono_col_simulatore(self, tmp_path):
        from mondo.numeri import lemma_numero
        from mondo.simulatore import genera_storia
        from esami.genera import _blocchi_stato_oro, _n_tick

        config = _config_piccolo(tmp_path, stato=True)
        seed = 0
        n_tick = _n_tick(1, seed, config)
        storia = genera_storia(seed=seed, n_tick=n_tick, persone=None)
        blocchi = _blocchi_stato_oro(seed, n_tick, None, storia)

        tick_con_eventi = sorted({e.t for e in storia.eventi})
        assert sorted(blocchi) == tick_con_eventi
        cast_ids = [p.id for p in dm.PERSONE]
        for t, (tick_lemma, posizioni) in blocchi.items():
            assert tick_lemma == lemma_numero(t)
            # una posizione per persona del cast, nell'ordine del cast
            assert [pid for pid, _ in posizioni] == cast_ids
            # ogni posizione == stato del simulatore troncato al tick t (§1.9)
            atteso = genera_storia(seed=seed, n_tick=t, persone=None).stato_finale
            for pid, luogo in posizioni:
                assert luogo == atteso.luogo_effettivo(pid)

    def test_record_train_interlaccia_lo_stato(self, tmp_path):
        from mondo.simulatore import genera_storia
        from esami.genera import _blocchi_stato_oro, _n_tick
        from cervello.sequenza import analizza_esempio_stato, grafo_posizione

        config = _config_piccolo(tmp_path, stato=True)
        seed = 0
        record = genera_record(1, seed, config, split="train")
        assert "[STATO]" in record["storia"]

        n_tick = _n_tick(1, seed, config)
        storia = genera_storia(seed=seed, n_tick=n_tick, persone=None)
        blocchi = _blocchi_stato_oro(seed, n_tick, None, storia)

        # ricompongo un esempio e lo analizzo: i blocchi combaciano con l'oro
        esempio = record["esempi"][0]
        seq = componi_esempio([record["storia"]], esempio["domanda"], esempio["risposta"])
        analizzato = analizza_esempio_stato(seq)
        assert len(analizzato.segmenti) == len(blocchi)
        for (_, blocco), t in zip(analizzato.segmenti, sorted(blocchi)):
            tick_lemma, posizioni = blocchi[t]
            assert blocco.tick_lemma == tick_lemma
            assert list(blocco.posizioni) == [grafo_posizione(pid, luogo) for pid, luogo in posizioni]

    def test_stato_oro_coincide_su_300_storie(self, tmp_path):
        # Cancello T3: stato-oro == simulatore su 300 storie campione. Ogni
        # persona ha sempre una posizione (mai None, anche senza aver agito).
        from mondo.simulatore import genera_storia
        from esami.genera import _blocchi_stato_oro, _n_tick

        config = _config_piccolo(tmp_path, stato=True)
        for seed in range(300):
            n_tick = _n_tick(1, seed, config)
            storia = genera_storia(seed=seed, n_tick=n_tick, persone=None)
            blocchi = _blocchi_stato_oro(seed, n_tick, None, storia)
            for t, (_, posizioni) in blocchi.items():
                atteso = genera_storia(seed=seed, n_tick=t, persone=None).stato_finale
                for pid, luogo in posizioni:
                    assert luogo is not None, f"seed {seed} tick {t}: {pid} senza posizione"
                    assert luogo == atteso.luogo_effettivo(pid), f"seed {seed} tick {t} {pid}"

    def test_dev_ed_esame_restano_senza_stato(self, tmp_path):
        config = _config_piccolo(tmp_path, stato=True)
        for split, seed in (("dev", 800_000), ("esame", 1_000_000)):
            record = genera_record(1, seed, config, split=split)
            assert "[STATO]" not in record["storia"]

    def test_default_senza_stato_byte_identico(self, tmp_path):
        seed = 0
        senza_chiave = genera_record(1, seed, _config_piccolo(tmp_path), split="train")
        con_false = genera_record(1, seed, _config_piccolo(tmp_path, stato=False), split="train")
        assert senza_chiave == con_false
        assert "[STATO]" not in senza_chiave["storia"]


@pytest.mark.torch
class TestValutaEsempioStato:
    """Fase B T4: decodifica interlacciata d'esame (fasi/FASE2_PIANO_STATO.md §5)."""

    def _storia_due_tick(self):
        ev1 = grafo_a_token(grafo_fatto("andare", nsubj="sara", **{"obl:luogo": "cucina", "obl:tempo": "uno"}))
        ev2 = grafo_a_token(grafo_fatto("andare", nsubj="sara", **{"obl:luogo": "giardino", "obl:tempo": "due"}))
        return ev1, ev2

    def _posizione(self, luogo):
        return grafo_a_token(grafo_fatto("trovarsi", nsubj="sara", **{"obl:luogo": luogo}))

    def test_raggruppa_eventi_per_tick(self):
        from esami.esamina import _raggruppa_eventi_per_tick

        ev1, ev2 = self._storia_due_tick()
        per_tick = _raggruppa_eventi_per_tick(ev1 + ev2)
        assert [t for t, _ in per_tick] == ["uno", "due"]
        assert per_tick[0][1] == [ev1]
        assert per_tick[1][1] == [ev2]

    def test_interlacciata_stop_su_controllo(self):
        # Il modello genera un blocco per tick e lo chiude emettendo un token di
        # controllo (non "("); gli eventi sono teacher-forced; poi la risposta.
        from esami.esamina import valuta_esempio_stato

        vocab = carica_vocabolario()
        ev1, ev2 = self._storia_due_tick()
        storia_flat = ev1 + ev2
        risposta = grafo_a_token(grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "giardino"}))
        esempio = {"tipo": "posizione", "domanda": grafo_a_token(grafo_fatto("trovarsi", nsubj="sara", quesito="dove")), "risposta": risposta}

        blocco1 = ["(", "obl:tempo", "uno", ")"] + self._posizione("cucina")
        blocco2 = ["(", "obl:tempo", "due", ")"] + self._posizione("giardino")
        script = blocco1 + ["[DOMANDA]"] + blocco2 + ["[DOMANDA]"] + risposta + ["[FINE]"]
        modello = _ModelloIniettato(vocab, script)

        esito = valuta_esempio_stato(modello, vocab, storia_flat, esempio, ctx=500, device="cpu")
        assert esito.esatto
        assert esito.token_generati == risposta
        assert esito.blocchi_generati == [blocco1, blocco2]

    def test_interlacciata_stop_su_radice_diversa(self):
        # Dopo le posizioni il modello "inizia il tick successivo" (gruppo con
        # radice != trovarsi): il blocco si chiude e quel gruppo NON è incluso.
        from esami.esamina import valuta_esempio_stato

        vocab = carica_vocabolario()
        ev1, _ = self._storia_due_tick()
        risposta = grafo_a_token(grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "cucina"}))
        esempio = {"tipo": "posizione", "domanda": grafo_a_token(grafo_fatto("trovarsi", nsubj="sara", quesito="dove")), "risposta": risposta}

        blocco1 = ["(", "obl:tempo", "uno", ")"] + self._posizione("cucina")
        # la decodifica riconosce l'inizio del tick successivo dopo "(" + radice:
        # bastano questi due token perché il blocco si chiuda scartandoli.
        stray = ["(", "andare"]
        script = blocco1 + stray + risposta + ["[FINE]"]
        modello = _ModelloIniettato(vocab, script)

        esito = valuta_esempio_stato(modello, vocab, ev1, esempio, ctx=500, device="cpu")
        assert esito.esatto
        assert esito.blocchi_generati == [blocco1]  # il gruppo stray non è incluso
