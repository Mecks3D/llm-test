"""Test del modulo lingua/ (fasi/FASE1_PIANO.md §12)."""
from __future__ import annotations

import pytest

from mondo import dati_mondo as dm
from lingua.lessico import ORDINE_PRIM, RELAZIONI_PARENTELA, carica_lessico


# ---------------------------------------------------------------------------
# Gruppo 1: lessico
# ---------------------------------------------------------------------------

class TestLessico:
    def test_valida_passa(self):
        lessico = carica_lessico()
        lessico.valida()  # non deve sollevare

    def test_65_prim_in_testa_nellordine_normativo(self):
        lessico = carica_lessico()
        voci = lessico.voci()
        assert len(voci) >= 65
        for i, lemma_atteso in enumerate(ORDINE_PRIM):
            assert voci[i].categoria == "PRIM"
            assert voci[i].lemma == lemma_atteso

    def test_copertura_azioni(self):
        from mondo.azioni import AZIONI
        lessico = carica_lessico()
        for nome in list(AZIONI.keys()) + ["bruciare"]:
            assert nome in lessico

    def test_copertura_persone_luoghi_oggetti(self):
        lessico = carica_lessico()
        for p in dm.PERSONE:
            assert p.id in lessico
        for l in dm.LUOGHI:
            assert l.id in lessico
        for tipo in dm.OGGETTI_UNICI:
            assert tipo.lemma in lessico

    def test_copertura_risorse_e_fonti(self):
        lessico = carica_lessico()
        for fonte, info in dm.RISORSE.items():
            assert fonte in lessico
            assert info["lemma_unita"] in lessico

    def test_copertura_parentela(self):
        lessico = carica_lessico()
        for relazione in RELAZIONI_PARENTELA:
            assert relazione in lessico
        assert len(RELAZIONI_PARENTELA) == 19

    def test_generi_persone_coerenti(self):
        lessico = carica_lessico()
        for p in dm.PERSONE:
            assert lessico[p.id].tratti["genere"] == p.genere

    def test_lemma_duplicato_solleva_errore(self, tmp_path):
        percorso = tmp_path / "lessico_rotto.tsv"
        contenuto = "io\tPRIM\tpos=PRON\t-\nio\tPRIM\tpos=PRON\t-\n"
        percorso.write_text(contenuto, encoding="utf-8")
        lessico = carica_lessico(percorso)
        with pytest.raises(ValueError):
            lessico.valida()

    def test_ordine_prim_sbagliato_solleva_errore(self, tmp_path):
        percorso = tmp_path / "lessico_rotto.tsv"
        contenuto = "tu\tPRIM\tpos=PRON\t-\nio\tPRIM\tpos=PRON\t-\n"
        percorso.write_text(contenuto, encoding="utf-8")
        lessico = carica_lessico(percorso)
        with pytest.raises(ValueError):
            lessico.valida()
