"""Test del modulo cervello/ (fasi/FASE2_PIANO.md)."""
from __future__ import annotations

import pytest

try:
    import torch
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
