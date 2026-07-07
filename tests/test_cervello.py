"""Test del modulo cervello/ (fasi/FASE2_PIANO.md)."""
from __future__ import annotations

import pytest

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
