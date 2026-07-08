"""Test del modulo cervello/ (fasi/FASE2_PIANO.md)."""
from __future__ import annotations

import pytest

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # i test @pytest.mark.torch vengono saltati da tests/conftest.py

from lingua.lessico import ORDINE_PRIM, N_PRIM
from cervello.vocabolario import (
    RELAZIONI_UD,
    TOKEN_SPECIALI,
    _PERCORSO_DEFAULT,
    carica_vocabolario,
    genera_vocabolario,
    salva_vocabolario,
    sha256_lessico,
)


# ---------------------------------------------------------------------------
# Gruppo 1: vocabolario
# ---------------------------------------------------------------------------

class TestVocabolario:
    def test_prim_occupano_id_0_64_nellordine_del_lessico(self):
        v = genera_vocabolario()
        for i, lemma_atteso in enumerate(ORDINE_PRIM):
            assert v.token(i) == lemma_atteso
            assert v.id(lemma_atteso) == i

    def test_speciali_agli_id_normativi(self):
        v = genera_vocabolario()
        for offset, token in enumerate(TOKEN_SPECIALI):
            id_atteso = N_PRIM + offset
            assert v.token(id_atteso) == token
            assert v.id(token) == id_atteso

    def test_relazioni_agli_id_normativi(self):
        v = genera_vocabolario()
        base = N_PRIM + len(TOKEN_SPECIALI)
        assert list(RELAZIONI_UD) == sorted(RELAZIONI_UD)
        for offset, relazione in enumerate(RELAZIONI_UD):
            id_atteso = base + offset
            assert v.token(id_atteso) == relazione
            assert v.id(relazione) == id_atteso

    def test_resto_del_lessico_dopo_le_relazioni(self):
        v = genera_vocabolario()
        base = N_PRIM + len(TOKEN_SPECIALI) + len(RELAZIONI_UD)
        # un lemma NOME qualsiasi noto (persona del mondo) deve stare oltre
        # la sezione delle relazioni.
        assert v.id("anna") >= base

    def test_dimensione_coerente(self):
        v = genera_vocabolario()
        assert v.dimensione == len(v.token_lista())
        assert v.dimensione == N_PRIM + len(TOKEN_SPECIALI) + len(RELAZIONI_UD) + (
            v.dimensione - N_PRIM - len(TOKEN_SPECIALI) - len(RELAZIONI_UD)
        )

    def test_nessun_token_duplicato(self):
        v = genera_vocabolario()
        assert len(set(v.token_lista())) == v.dimensione

    def test_rigenerazione_byte_identica_al_committato(self):
        assert _PERCORSO_DEFAULT.exists(), "vocabolario.json va committato (rigenerare con python -m cervello.vocabolario)"
        atteso = _PERCORSO_DEFAULT.read_bytes()

        v = genera_vocabolario()
        tmp = _PERCORSO_DEFAULT.with_suffix(".tmp_test.json")
        try:
            salva_vocabolario(v, percorso=tmp)
            assert tmp.read_bytes() == atteso
        finally:
            tmp.unlink(missing_ok=True)

    def test_carica_vocabolario_combacia_con_generato(self):
        v_generato = genera_vocabolario()
        v_caricato = carica_vocabolario()
        assert v_generato.token_lista() == v_caricato.token_lista()

    def test_carica_vocabolario_rifiuta_lessico_alterato(self, tmp_path):
        lessico_alterato = tmp_path / "lessico.tsv"
        lessico_alterato.write_text("# lessico alterato per test\n", encoding="utf-8")

        with pytest.raises(ValueError, match="lessico è cambiato"):
            carica_vocabolario(percorso_lessico=lessico_alterato)

    def test_sha256_lessico_deterministico(self):
        assert sha256_lessico() == sha256_lessico()


# ---------------------------------------------------------------------------
# Gruppo 2: sequenza (linearizzazione grafo <-> token)
# ---------------------------------------------------------------------------

import random

from mondo.domande import genera_domande
from mondo.generatore import _lunghezza_storia
from mondo.grafo import NON_LO_SO, evento_a_grafo, grafo_fatto
from mondo.simulatore import genera_storia
from mondo.tipi import Evento
from cervello.sequenza import componi_esempio, grafo_a_token, token_a_grafo


class TestSequenzaGolden:
    def test_golden_evento(self):
        e = Evento(t=9, azione="andare", agente="sara", luogo_origine="cucina", luogo="giardino")
        g = evento_a_grafo(e)
        atteso = (
            "( andare ( nsubj sara ) ( obl:origine cucina ) "
            "( obl:luogo giardino ) ( obl:tempo nove ) )"
        ).split()
        assert grafo_a_token(g) == atteso
        assert token_a_grafo(atteso, "evento") == g

    def test_golden_domanda_istanza(self):
        g = grafo_fatto("trovarsi", nsubj="mela_1", quesito="dove")
        atteso = "( trovarsi ( nsubj mela primo ) ( quesito dove ) )".split()
        assert grafo_a_token(g) == atteso
        assert token_a_grafo(atteso, "fatto") == g

    def test_non_lo_so(self):
        atteso = ["(", "non-lo-so", ")"]
        assert grafo_a_token(NON_LO_SO) == atteso
        assert token_a_grafo(atteso, "fatto") == NON_LO_SO

    def test_istanza_qualunque_ordinale_anche_uno(self):
        g = grafo_fatto("avere", obj="legna_1", quesito="chi")
        tok = grafo_a_token(g)
        assert tok == "( avere ( obj legna primo ) ( quesito chi ) )".split()
        assert token_a_grafo(tok, "fatto") == g

    def test_ordinale_oltre_30_solleva(self):
        with pytest.raises(ValueError):
            grafo_a_token(grafo_fatto("avere", obj="mela_31", quesito="chi"))


class TestSequenzaMalformata:
    def test_parentesi_troncata(self):
        with pytest.raises(ValueError):
            token_a_grafo(["(", "andare", "(", "nsubj", "sara", ")"], "evento")

    def test_token_in_eccesso(self):
        with pytest.raises(ValueError):
            token_a_grafo(["(", "andare", ")", ")"], "evento")

    def test_relazione_ignota(self):
        with pytest.raises(ValueError):
            token_a_grafo(["(", "andare", "(", "boh:relazione", "sara", ")", ")"], "evento")

    def test_ordinale_orfano(self):
        with pytest.raises(ValueError):
            token_a_grafo(
                ["(", "andare", "(", "nsubj", "mela", "primo", "extra", ")", ")"], "evento"
            )

    def test_radice_sconosciuta(self):
        with pytest.raises(ValueError):
            token_a_grafo(["(", "nsubj", "sara", ")"], "evento")

    def test_famiglia_sconosciuta(self):
        with pytest.raises(ValueError):
            token_a_grafo(["(", "andare", ")"], "boh")

    def test_evento_senza_agente(self):
        with pytest.raises(ValueError):
            token_a_grafo(["(", "andare", "(", "obl:tempo", "nove", ")", ")"], "evento")


class TestComponiEsempio:
    def test_struttura(self):
        storia = [["(", "andare", ")"], ["(", "dormire", ")"]]
        domanda = ["(", "trovarsi", ")"]
        risposta = ["(", "non-lo-so", ")"]
        atteso = (
            ["[STORIA]"] + storia[0] + storia[1]
            + ["[DOMANDA]"] + domanda
            + ["[RISPOSTA]"] + risposta
            + ["[FINE]"]
        )
        assert componi_esempio(storia, domanda, risposta) == atteso


class TestRoundTripMassa:
    def test_round_trip_eventi_e_domande_seed_0_299(self):
        vocab = carica_vocabolario()
        for seed in range(300):
            n_tick = _lunghezza_storia(seed)
            storia = genera_storia(seed=seed, n_tick=n_tick)
            grafi = [evento_a_grafo(e) for e in storia.eventi]
            for g in grafi:
                tok = grafo_a_token(g)
                for t in tok:
                    assert t in vocab, f"seed {seed}: token {t!r} fuori vocabolario"
                assert token_a_grafo(tok, "evento") == g, f"seed {seed}: round-trip evento fallito"

            rng = random.Random(f"domande-{seed}")
            for d in genera_domande(storia, rng, n_per_tipo=8):
                for g in (d.grafo_domanda, d.grafo_risposta):
                    tok = grafo_a_token(g)
                    for t in tok:
                        assert t in vocab, f"seed {seed} {d.tipo}: token {t!r} fuori vocabolario"
                    assert token_a_grafo(tok, "fatto") == g, (
                        f"seed {seed} {d.tipo}: round-trip fatto fallito"
                    )


# ---------------------------------------------------------------------------
# Gruppo 4: dati (cervello/dati.py) — richiede torch
# ---------------------------------------------------------------------------

_ESEMPIO_GOLDEN = (
    "[STORIA] ( andare ( nsubj sara ) ( obl:luogo cucina ) ( obl:tempo uno ) ) "
    "[DOMANDA] ( trovarsi ( nsubj sara ) ( quesito dove ) ) "
    "[RISPOSTA] ( essere ( nsubj sara ) ( obl:luogo cucina ) ) [FINE]"
).split()

# esempio più corto (storia vuota, risposta non-lo-so), ma con la stessa
# struttura [STORIA]...[RISPOSTA]...[FINE] di un vero esempio composto.
_ESEMPIO_CORTO = (
    "[STORIA] [DOMANDA] ( trovarsi ( nsubj sara ) ( quesito dove ) ) "
    "[RISPOSTA] ( non-lo-so ) [FINE]"
).split()


@pytest.mark.torch
class TestDatiMaschera:
    def test_esempio_golden_composto_correttamente(self):
        e = Evento(t=1, azione="andare", agente="sara", luogo="cucina")
        storia_tok = grafo_a_token(evento_a_grafo(e))
        domanda_tok = grafo_a_token(grafo_fatto("trovarsi", nsubj="sara", quesito="dove"))
        risposta_tok = grafo_a_token(
            grafo_fatto("essere", nsubj="sara", **{"obl:luogo": "cucina"})
        )
        from cervello.sequenza import componi_esempio
        assert componi_esempio([storia_tok], domanda_tok, risposta_tok) == _ESEMPIO_GOLDEN

    def test_maschera_vera_solo_dopo_risposta_fino_a_fine(self):
        from cervello.dati import _maschera_piena

        maschera = _maschera_piena(_ESEMPIO_GOLDEN)
        assert len(maschera) == len(_ESEMPIO_GOLDEN) == 41
        idx_risposta = _ESEMPIO_GOLDEN.index("[RISPOSTA]")
        idx_fine = _ESEMPIO_GOLDEN.index("[FINE]")
        assert idx_risposta == 28 and idx_fine == 40

        atteso = [False] * 29 + [True] * 12  # indici 0-28 False, 29-40 True
        assert maschera == atteso
        assert not any(maschera[: idx_risposta + 1])
        assert all(maschera[idx_risposta + 1 : idx_fine + 1])


@pytest.mark.torch
class TestImpacchettaBatch:
    def test_padding_e_shift_corretti(self):
        from cervello.dati import impacchetta_batch

        vocab = carica_vocabolario()
        corto = _ESEMPIO_CORTO
        lungo = _ESEMPIO_GOLDEN

        batch = impacchetta_batch([corto, lungo], vocab)
        T = len(lungo) - 1
        assert batch.input.shape == (2, T)
        assert batch.bersaglio.shape == (2, T)
        assert batch.maschera.shape == (2, T)

        # riga 0 (corto, 18 token -> 17 posizioni di shift), poi tutto [PAD]
        id_pad = vocab.id("[PAD]")
        n_corto = len(corto) - 1
        assert batch.input[0, 0].item() == vocab.id("[STORIA]")
        ids_attesi = [vocab.id(t) for t in corto]
        assert batch.input[0, :n_corto].tolist() == ids_attesi[:-1]
        assert batch.bersaglio[0, :n_corto].tolist() == ids_attesi[1:]
        assert batch.maschera[0, :n_corto].sum().item() == 4  # ( non-lo-so ) [FINE]
        assert torch.all(batch.input[0, n_corto:] == id_pad)
        assert torch.all(batch.bersaglio[0, n_corto:] == id_pad)
        assert not torch.any(batch.maschera[0, n_corto:])  # padding esclude dalla loss

        # riga 1 (lungo): shift esatto su tutta la riga, nessun padding
        assert batch.input[1, 0].item() == vocab.id("[STORIA]")
        assert batch.bersaglio[1, -1].item() == vocab.id("[FINE]")
        assert batch.maschera[1].sum().item() == 12  # le 12 posizioni di risposta+[FINE]

    def test_batch_vuoto_solleva(self):
        from cervello.dati import impacchetta_batch

        with pytest.raises(ValueError):
            impacchetta_batch([], carica_vocabolario())


@pytest.mark.torch
class TestGeneraBatch:
    def test_mescola_e_copre_tutti_gli_esempi(self):
        from cervello.dati import genera_batch

        vocab = carica_vocabolario()
        esempi = [_ESEMPIO_GOLDEN, _ESEMPIO_CORTO, _ESEMPIO_GOLDEN, _ESEMPIO_CORTO, _ESEMPIO_GOLDEN]
        rng = random.Random("epoca-0")
        batch = list(genera_batch(esempi, vocab, batch_size=2, rng=rng))
        assert sum(b.input.shape[0] for b in batch) == len(esempi)
        assert len(batch) == 3  # 2 + 2 + 1

    def test_determinismo_stesso_rng_seed(self):
        from cervello.dati import genera_batch

        vocab = carica_vocabolario()
        esempi = [_ESEMPIO_GOLDEN, _ESEMPIO_CORTO, _ESEMPIO_GOLDEN]
        b1 = list(genera_batch(esempi, vocab, batch_size=2, rng=random.Random("seme-x")))
        b2 = list(genera_batch(esempi, vocab, batch_size=2, rng=random.Random("seme-x")))
        for x, y in zip(b1, b2):
            assert torch.equal(x.input, y.input)
            assert torch.equal(x.bersaglio, y.bersaglio)
            assert torch.equal(x.maschera, y.maschera)


# ---------------------------------------------------------------------------
# Gruppo 5: modello (cervello/modello.py) — richiede torch
# ---------------------------------------------------------------------------

@pytest.mark.torch
class TestModello:
    def _config(self):
        from cervello.modello import ConfigModello
        vocab = carica_vocabolario()
        return ConfigModello(vocab_size=vocab.dimensione, ctx=3072)

    def test_forward_shape(self):
        from cervello.modello import Modello
        torch.manual_seed(0)
        m = Modello(self._config())
        x = torch.randint(0, self._config().vocab_size, (2, 17))
        logits = m(x)
        assert logits.shape == (2, 17, self._config().vocab_size)

    def test_supera_ctx_solleva(self):
        from cervello.modello import Modello
        torch.manual_seed(0)
        cfg = self._config()
        m = Modello(cfg)
        x = torch.randint(0, cfg.vocab_size, (1, cfg.ctx + 1))
        with pytest.raises(ValueError):
            m(x)

    def test_numero_parametri_circa_7_3m(self):
        from cervello.modello import Modello
        torch.manual_seed(0)
        m = Modello(self._config())
        n = m.numero_parametri()
        atteso = 7_300_000
        assert abs(n - atteso) / atteso <= 0.05, f"{n} parametri, atteso ~{atteso} (±5%)"

    def test_determinismo_stesso_seed(self):
        from cervello.modello import Modello
        cfg = self._config()
        x = torch.randint(0, cfg.vocab_size, (2, 12))

        torch.use_deterministic_algorithms(True)
        try:
            torch.manual_seed(123)
            m1 = Modello(cfg)
            logits1 = m1(x)

            torch.manual_seed(123)
            m2 = Modello(cfg)
            logits2 = m2(x)

            assert torch.equal(logits1, logits2)
        finally:
            torch.use_deterministic_algorithms(False)


# ---------------------------------------------------------------------------
# Gruppo 6: canary di apprendimento — cancello duro (FASE2_PIANO.md §9.6)
#
# Marcato "slow": escluso dalla suite di default (pytest.ini: -m "not slow")
# perché su CPU è impraticabile (misurato: >30s/step con batch=32, ctx~360).
# Va eseguito esplicitamente, tipicamente su GPU (Colab: vedi
# colab_training.ipynb, cella dedicata al canary).
# ---------------------------------------------------------------------------

def _decodifica_greedy_canario(modello, vocab, id_fine, ids_prefisso, max_nuovi, device):
    modello.eval()
    ids = list(ids_prefisso)
    with torch.no_grad():
        for _ in range(max_nuovi):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits = modello(x)
            prossimo = int(torch.argmax(logits[0, -1]).item())
            ids.append(prossimo)
            if prossimo == id_fine:
                break
    modello.train()
    return ids[len(ids_prefisso):]


@pytest.mark.torch
@pytest.mark.slow
class TestCanarioApprendimento:
    def test_overfit_32_esempi_stadio1(self):
        from esami.genera import carica_config, genera_dataset
        from cervello.dati import componi_esempio, impacchetta_batch
        from cervello.modello import ConfigModello, Modello

        vocab = carica_vocabolario()
        config = carica_config()
        config = {**config, "dataset": {**config["dataset"], "train_storie": 40}}
        record = genera_dataset(1, "train", config)

        esempi: list[list[str]] = []
        for r in record:
            for e in r["esempi"]:
                esempi.append(componi_esempio([r["storia"]], e["domanda"], e["risposta"]))
        esempi.sort(key=len)
        esempi = esempi[:32]
        assert len(esempi) == 32

        idx_risposta = [e.index("[RISPOSTA]") for e in esempi]
        idx_fine = [e.index("[FINE]") for e in esempi]
        # il grafo-verità è la sotto-sequenza tra [RISPOSTA] e [FINE] (esclusi):
        # è esattamente l'output di grafo_a_token(grafo_risposta).
        risposte_oro = [
            token_a_grafo(e[ir + 1 : ifi], "fatto")
            for e, ir, ifi in zip(esempi, idx_risposta, idx_fine)
        ]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = ConfigModello(vocab_size=vocab.dimensione, ctx=3072, dropout=0.0)
        torch.manual_seed(0)
        modello = Modello(cfg).to(device)
        ottim = torch.optim.AdamW(modello.parameters(), lr=1e-3, betas=(0.9, 0.95))

        batch = impacchetta_batch(esempi, vocab)
        input_ids = batch.input.to(device)
        bersaglio = batch.bersaglio.to(device)
        maschera = batch.maschera.to(device).float()

        id_fine = vocab.id("[FINE]")
        max_step = 1000
        esatti = 0
        step_raggiunto = 0
        for step in range(max_step):
            step_raggiunto = step + 1
            modello.train()
            logits = modello(input_ids)
            loss_tok = F.cross_entropy(
                logits.reshape(-1, vocab.dimensione), bersaglio.reshape(-1), reduction="none"
            )
            loss = (loss_tok * maschera.reshape(-1)).sum() / maschera.sum()
            ottim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modello.parameters(), 1.0)
            ottim.step()

            if step_raggiunto % 20 != 0 and step_raggiunto != max_step:
                continue

            esatti = 0
            for e, idx, ifi, oro in zip(esempi, idx_risposta, idx_fine, risposte_oro):
                prefisso = [vocab.id(t) for t in e[: idx + 1]]
                max_nuovi = (ifi - idx) + 5
                generati = _decodifica_greedy_canario(
                    modello, vocab, id_fine, prefisso, max_nuovi, device
                )
                token_generati = [vocab.token(i) for i in generati]
                if token_generati and token_generati[-1] == "[FINE]":
                    token_generati = token_generati[:-1]
                try:
                    grafo_generato = token_a_grafo(token_generati, "fatto")
                except ValueError:
                    continue
                if grafo_generato == oro:
                    esatti += 1
            if esatti == len(esempi):
                break

        assert esatti == len(esempi), (
            f"canary non converge: {esatti}/{len(esempi)} esatti dopo {step_raggiunto} step "
            "su 1000 (qualcosa è scollegato nella pipeline dati->modello->loss: NON procedere "
            "alle tappe successive finché questo non passa)"
        )


# ---------------------------------------------------------------------------
# Gruppo 7 (parte addestra): ripresa con --stadio N dal checkpoint N-1
# ---------------------------------------------------------------------------

@pytest.mark.torch
class TestRipresaDaCheckpoint:
    """Con --stadio N (N non minimo) il training riparte dal checkpoint
    dello stadio N-1, mai da pesi casuali (FASE2_PIANO.md §7)."""

    def _config(self, tmp_path):
        return {
            "nome_run": "run_test",
            "device": "cpu",
            "seed_torch": 0,
            "percorsi": {
                "dati_dir": str(tmp_path / "dati"),
                "risultati_dir": str(tmp_path / "risultati"),
            },
            "dataset": {"ctx": 64, "train_storie": 1, "dev_storie": 1,
                        "esame_storie": 1, "n_per_tipo": 1},
            "stadi": {
                1: {"tipi": ["posizione"], "soglia": 0.95, "storie_corte": True},
                2: {"tipi": ["posizione"], "soglia": 0.95, "storie_corte": False},
            },
            "modello": {"n_layer": 1, "n_head": 2, "d_model": 8, "d_ff": 16,
                        "dropout": 0.0},
            "training": {"batch": 1, "accumulo": 1, "lr": 3.0e-4, "beta1": 0.9,
                         "beta2": 0.95, "weight_decay": 0.1, "warmup_step": 1,
                         "grad_clip": 1.0, "max_step": 1,
                         "intervallo_valutazione": 1, "dev_campione": 1},
        }

    def _percorso_config(self, tmp_path, config):
        import yaml
        percorso = tmp_path / "config.yaml"
        percorso.write_text(yaml.safe_dump(config), encoding="utf-8")
        return percorso

    def test_stadio_2_senza_checkpoint_stadio_1_solleva(self, tmp_path):
        from cervello.addestra import esegui_curriculum
        config = self._config(tmp_path)
        percorso_config = self._percorso_config(tmp_path, config)
        with pytest.raises(FileNotFoundError, match="stadio 1"):
            esegui_curriculum(config, percorso_config, solo_stadio=2)

    def test_stadio_2_con_checkpoint_lo_carica_e_prosegue(self, tmp_path):
        from cervello.addestra import esegui_curriculum
        from cervello.modello import ConfigModello, Modello
        config = self._config(tmp_path)
        percorso_config = self._percorso_config(tmp_path, config)

        vocab = carica_vocabolario()
        cfg = ConfigModello(vocab_size=vocab.dimensione, ctx=64, **config["modello"])
        torch.manual_seed(0)
        dir_risultati = tmp_path / "risultati" / "run_test"
        dir_risultati.mkdir(parents=True)
        torch.save({"modello": Modello(cfg).state_dict()}, dir_risultati / "stadio1.pt")

        # Il checkpoint viene caricato e si prosegue fino ai dataset (qui
        # assenti): l'errore atteso riguarda train.jsonl, NON il checkpoint.
        with pytest.raises(FileNotFoundError, match="train.jsonl"):
            esegui_curriculum(config, percorso_config, solo_stadio=2)


@pytest.mark.torch
class TestPesiIniziali:
    """--pesi-iniziali fa ripartire il PRIMO stadio di una run da un
    checkpoint esterno (es. curriculum "facile" che riparte dai pesi di
    una run diversa), invece che da pesi casuali."""

    def _config(self, tmp_path):
        return TestRipresaDaCheckpoint._config(self, tmp_path)

    def _percorso_config(self, tmp_path, config):
        return TestRipresaDaCheckpoint._percorso_config(self, tmp_path, config)

    def test_valido_solo_per_il_primo_stadio(self, tmp_path):
        from cervello.addestra import esegui_curriculum
        config = self._config(tmp_path)
        percorso_config = self._percorso_config(tmp_path, config)
        with pytest.raises(ValueError, match="primo stadio"):
            esegui_curriculum(
                config, percorso_config, solo_stadio=2,
                pesi_iniziali=tmp_path / "non_esiste.pt",
            )

    def test_carica_i_pesi_e_prosegue(self, tmp_path):
        from cervello.addestra import esegui_curriculum
        from cervello.modello import ConfigModello, Modello
        config = self._config(tmp_path)
        percorso_config = self._percorso_config(tmp_path, config)

        vocab = carica_vocabolario()
        cfg = ConfigModello(vocab_size=vocab.dimensione, ctx=64, **config["modello"])
        torch.manual_seed(0)
        percorso_pesi = tmp_path / "esterno.pt"
        torch.save({"modello": Modello(cfg).state_dict()}, percorso_pesi)

        # I pesi vengono caricati e si prosegue fino ai dataset (qui
        # assenti): l'errore atteso riguarda train.jsonl, NON i pesi.
        with pytest.raises(FileNotFoundError, match="train.jsonl"):
            esegui_curriculum(
                config, percorso_config, solo_stadio=1, pesi_iniziali=percorso_pesi,
            )


# ---------------------------------------------------------------------------
# Gruppo 7 (parte addestra): checkpoint intra-stadio e ripresa da dentro
# lo stadio (run interrotto a metà, es. Colab morto a 6500/20000 step)
# ---------------------------------------------------------------------------

@pytest.mark.torch
class TestCheckpointIntraStadio:
    def _config(self, tmp_path):
        return {
            "nome_run": "run_test",
            "device": "cpu",
            "seed_torch": 0,
            "percorsi": {
                "dati_dir": str(tmp_path / "dati"),
                "risultati_dir": str(tmp_path / "risultati"),
            },
            "dataset": {"ctx": 1024, "train_storie": 2, "dev_storie": 1,
                        "esame_storie": 1, "n_per_tipo": 1},
            "stadi": {
                1: {"tipi": ["posizione"], "soglia": 0.95, "storie_corte": True},
            },
            "modello": {"n_layer": 1, "n_head": 2, "d_model": 8, "d_ff": 16,
                        "dropout": 0.0},
            "training": {"batch": 1, "accumulo": 1, "lr": 3.0e-4, "beta1": 0.9,
                         "beta2": 0.95, "weight_decay": 0.1, "warmup_step": 1,
                         "grad_clip": 1.0, "max_step": 4,
                         "intervallo_valutazione": 2, "dev_campione": 1},
        }

    def _percorso_config(self, tmp_path, config):
        import yaml
        percorso = tmp_path / "config.yaml"
        percorso.write_text(yaml.safe_dump(config), encoding="utf-8")
        return percorso

    def _scrivi_dataset(self, config):
        from esami.genera import scrivi_dataset
        for split in ("train", "dev", "esame"):
            scrivi_dataset(1, split, config)

    def test_parziale_salvato_a_ogni_valutazione(self, tmp_path):
        import json
        from cervello.addestra import addestra_stadio, crea_ottimizzatore
        from cervello.dati import carica_esempi
        from cervello.modello import ConfigModello, Modello
        from esami.genera import percorso_dataset

        config = self._config(tmp_path)
        self._scrivi_dataset(config)
        vocab = carica_vocabolario()
        torch.manual_seed(0)
        cfg = ConfigModello(vocab_size=vocab.dimensione, ctx=1024, **config["modello"])
        modello = Modello(cfg)
        ottimizzatore = crea_ottimizzatore(modello, config)

        esempi = carica_esempi(percorso_dataset(1, "train", config))
        with open(percorso_dataset(1, "dev", config), encoding="utf-8") as f:
            dev_record = [json.loads(r) for r in f]
        percorso_parziale = tmp_path / "stadio1_parziale.pt"

        addestra_stadio(
            modello, ottimizzatore, config, 1, esempi, dev_record, vocab, "cpu",
            tmp_path / "log.jsonl", 0, percorso_parziale=percorso_parziale,
        )

        assert percorso_parziale.exists()
        stato = torch.load(percorso_parziale, map_location="cpu")
        assert stato["stadio"] == 1
        assert stato["step"] == config["training"]["max_step"]
        for chiave in ("epoca", "step_inizio_epoca", "passaggi_consecutivi",
                       "modello", "ottimizzatore", "rng_torch"):
            assert chiave in stato

    def test_parziale_di_stadio_sbagliato_solleva(self, tmp_path):
        from cervello.addestra import _salva_parziale, carica_parziale, crea_ottimizzatore
        from cervello.modello import ConfigModello, Modello

        config = self._config(tmp_path)
        vocab = carica_vocabolario()
        cfg = ConfigModello(vocab_size=vocab.dimensione, ctx=1024, **config["modello"])
        torch.manual_seed(0)
        modello = Modello(cfg)
        ottimizzatore = crea_ottimizzatore(modello, config)
        percorso = tmp_path / "stadio1_parziale.pt"
        _salva_parziale(percorso, modello, ottimizzatore, 2, 1, 0, 0, 0)

        with pytest.raises(ValueError, match="stadio 2"):
            carica_parziale(percorso, modello, ottimizzatore, 1, "cpu")

    def test_ripresa_riproduce_il_run_ininterrotto(self, tmp_path, monkeypatch):
        """Run A: 4 step senza interruzioni. Run B: interrotto alla seconda
        valutazione (step 4, dopo il parziale di step 2), poi ripreso dal
        parziale. I pesi finali devono coincidere byte per byte (su CPU)."""
        import cervello.addestra as addestra_mod
        from cervello.addestra import esegui_curriculum

        config = self._config(tmp_path)
        percorso_config = self._percorso_config(tmp_path, config)
        self._scrivi_dataset(config)
        dir_risultati = tmp_path / "risultati" / "run_test"

        # Run A: ininterrotto (l'esame fallisce, ma il checkpoint c'è).
        esegui_curriculum(config, percorso_config, solo_stadio=1)
        pesi_a = torch.load(dir_risultati / "stadio1.pt", map_location="cpu")["modello"]
        import shutil
        shutil.rmtree(dir_risultati)

        # Run B, parte 1: interruzione simulata alla seconda valutazione.
        valuta_originale = addestra_mod._valuta_dev
        chiamate = {"n": 0}

        def valuta_e_interrompi(*args, **kwargs):
            chiamate["n"] += 1
            if chiamate["n"] >= 2:
                raise RuntimeError("interruzione simulata")
            return valuta_originale(*args, **kwargs)

        monkeypatch.setattr(addestra_mod, "_valuta_dev", valuta_e_interrompi)
        with pytest.raises(RuntimeError, match="interruzione simulata"):
            esegui_curriculum(config, percorso_config, solo_stadio=1)
        monkeypatch.undo()

        percorso_parziale = dir_risultati / "stadio1_parziale.pt"
        assert percorso_parziale.exists()
        assert torch.load(percorso_parziale, map_location="cpu")["step"] == 2

        # Run B, parte 2: ripresa dal parziale fino a fine stadio.
        esegui_curriculum(config, percorso_config, solo_stadio=1)
        pesi_b = torch.load(dir_risultati / "stadio1.pt", map_location="cpu")["modello"]

        assert not percorso_parziale.exists()  # rimosso a stadio completato
        assert pesi_a.keys() == pesi_b.keys()
        for chiave in pesi_a:
            assert torch.equal(pesi_a[chiave], pesi_b[chiave]), chiave

    def test_ripresa_da_copia_di_sicurezza_dopo_perdita_del_locale(self, tmp_path, monkeypatch):
        """Scenario Colab: il runtime muore (filesystem locale PERSO, resta
        solo la copia su Drive), si riparte da un runtime nuovo. Il parziale
        va recuperato dalla copia di sicurezza e i pesi finali devono
        coincidere con il run ininterrotto."""
        import json
        import shutil
        import cervello.addestra as addestra_mod
        from cervello.addestra import esegui_curriculum

        config = self._config(tmp_path)
        percorso_config = self._percorso_config(tmp_path, config)
        self._scrivi_dataset(config)
        dir_risultati = tmp_path / "risultati" / "run_test"
        dir_copia = tmp_path / "drive"

        # Run A: ininterrotto, senza copia di sicurezza.
        esegui_curriculum(config, percorso_config, solo_stadio=1)
        pesi_a = torch.load(dir_risultati / "stadio1.pt", map_location="cpu")["modello"]
        shutil.rmtree(dir_risultati)

        # Run B: interrotto alla seconda valutazione, con copia su "Drive".
        valuta_originale = addestra_mod._valuta_dev
        chiamate = {"n": 0}

        def valuta_e_interrompi(*args, **kwargs):
            chiamate["n"] += 1
            if chiamate["n"] >= 2:
                raise RuntimeError("runtime morto")
            return valuta_originale(*args, **kwargs)

        monkeypatch.setattr(addestra_mod, "_valuta_dev", valuta_e_interrompi)
        with pytest.raises(RuntimeError, match="runtime morto"):
            esegui_curriculum(config, percorso_config, solo_stadio=1,
                              copia_sicurezza=dir_copia)
        monkeypatch.undo()

        # Il parziale è stato replicato sulla copia; il locale muore.
        assert (dir_copia / "stadio1_parziale.pt").exists()
        assert (dir_copia / "log.jsonl").exists()
        shutil.rmtree(dir_risultati)

        # Runtime nuovo: si recupera dalla copia e si riprende.
        esegui_curriculum(config, percorso_config, solo_stadio=1,
                          copia_sicurezza=dir_copia)
        pesi_b = torch.load(dir_risultati / "stadio1.pt", map_location="cpu")["modello"]

        for chiave in pesi_a:
            assert torch.equal(pesi_a[chiave], pesi_b[chiave]), chiave
        # A stadio completato: checkpoint finale ed esito replicati sulla
        # copia, parziale rimosso da entrambe le parti.
        assert (dir_copia / "stadio1.pt").exists()
        assert (dir_copia / "esame_stadio1.json").exists()
        assert not (dir_copia / "stadio1_parziale.pt").exists()
        assert not (dir_risultati / "stadio1_parziale.pt").exists()
        # Il log recuperato dalla copia mantiene la storia intera: la
        # valutazione di prima dell'interruzione (step 2) più quella dopo
        # la ripresa (step 4), senza buchi né ripartenze da zero.
        with open(dir_risultati / "log.jsonl", encoding="utf-8") as f:
            steps = [json.loads(r)["step"] for r in f]
        assert steps == [2, 4], steps
