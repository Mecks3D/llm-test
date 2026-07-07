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
