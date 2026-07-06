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


# ---------------------------------------------------------------------------
# Gruppo 2: morfologia
# ---------------------------------------------------------------------------

from lingua import morfologia as mf


class TestArticoli:
    def test_articolo_det_maschile(self):
        assert mf.articolo_det("pane") == "il"
        assert mf.articolo_det("orto") == "l'"
        assert mf.articolo_det("secchio") == "il"
        assert mf.articolo_det("melo") == "il"

    def test_articolo_det_femminile(self):
        assert mf.articolo_det("acqua") == "l'"
        assert mf.articolo_det("legna") == "la"
        assert mf.articolo_det("palla") == "la"
        assert mf.articolo_det("scatola") == "la"

    def test_articolo_det_plurale(self):
        assert mf.articolo_det("oggetto", plurale=True) == "gli"
        assert mf.articolo_det("pane", plurale=True) == "i"
        assert mf.articolo_det("mela", plurale=True) == "le"

    def test_articolo_indet(self):
        assert mf.articolo_indet("mela") == "una"
        assert mf.articolo_indet("acqua") == "un'"
        assert mf.articolo_indet("pane") == "un"

    def test_partitivo(self):
        assert mf.partitivo("acqua") == "dell'acqua"
        assert mf.partitivo("legna") == "della legna"

    def test_prep_articolata_fusa(self):
        assert mf.prep_articolata("in", "cestino") == "nel"
        assert mf.prep_articolata("in", "scatola") == "nella"
        assert mf.prep_articolata("di", "acqua") == "dell'"

    def test_prep_lemma_contenitori(self):
        assert mf.prep_lemma("in", "cestino") == "nel cestino"
        assert mf.prep_lemma("in", "scatola") == "nella scatola"
        assert mf.prep_lemma("in", "secchio") == "nel secchio"
        assert mf.prep_lemma("in", "camino") == "nel camino"
        assert mf.prep_lemma("da", "cestino") == "dal cestino"
        assert mf.prep_lemma("da", "scatola") == "dalla scatola"


class TestLuoghi:
    def test_loc_in_loc_da_tabella_43(self):
        attesi = {
            "cucina": ("in cucina", "dalla cucina"),
            "salotto": ("in salotto", "dal salotto"),
            "giardino": ("in giardino", "dal giardino"),
            "camera": ("in camera", "dalla camera"),
            "orto": ("nell'orto", "dall'orto"),
            "bosco": ("nel bosco", "dal bosco"),
        }
        for luogo, (loc_in, loc_da) in attesi.items():
            assert mf.loc_in(luogo) == loc_in
            assert mf.loc_da(luogo) == loc_da

    def test_luogo_da_loc_inverso(self):
        for luogo in ("cucina", "salotto", "giardino", "camera", "orto", "bosco"):
            assert mf.luogo_da_loc_in(mf.loc_in(luogo)) == luogo
            assert mf.luogo_da_loc_da(mf.loc_da(luogo)) == luogo


class TestNumeri:
    def test_round_trip_0_70(self):
        for n in range(0, 71):
            assert mf.numero_da_lettere(mf.numero_in_lettere(n)) == n

    def test_ventitre_ha_accento_in_superficie(self):
        assert mf.numero_in_lettere(23) == "ventitré"
        assert mf.numero_da_lettere("ventitré") == 23

    def test_numero_fuori_dominio(self):
        with pytest.raises(ValueError):
            mf.numero_in_lettere(71)


class TestOrdinali:
    def test_round_trip_1_30(self):
        for n in range(1, 31):
            assert mf.ordinale_inverso(mf.ordinale(n, "m")) == n
            assert mf.ordinale_inverso(mf.ordinale(n, "f")) == n

    def test_femminile_regolare(self):
        assert mf.ordinale(1, "f") == "prima"
        assert mf.ordinale(2, "f") == "seconda"
        assert mf.ordinale(1, "m") == "primo"

    def test_ordinale_fuori_dominio(self):
        with pytest.raises(ValueError):
            mf.ordinale(31, "m")


class TestOre:
    def test_round_trip_1_24(self):
        for t in range(1, 25):
            assert mf.ora_da_lettere(mf.ora_in_lettere(t)) == t

    def test_valori_noti(self):
        assert mf.ora_in_lettere(1) == "all'una"
        assert mf.ora_in_lettere(9) == "alle nove"
        assert mf.ora_in_lettere(23) == "alle ventitré"


class TestPluraliEForme:
    def test_plurale(self):
        assert mf.plurale("mela") == "mele"
        assert mf.plurale("pane") == "pani"
        assert mf.plurale("oggetto") == "oggetti"

    def test_forma_verbale(self):
        assert mf.forma_verbale("andare", "pres3s") == "va"
        assert mf.forma_verbale("dare", "pres3s") == "dà"
        assert mf.forma_verbale("dare", "part") == "dato"
        assert mf.forma_verbale("mettere_dentro", "superficie_pres3s") == "mette"

    def test_aggettivo(self):
        assert mf.aggettivo("stanco", "m") == "stanco"
        assert mf.aggettivo("stanco", "f") == "stanca"


# ---------------------------------------------------------------------------
# Gruppo 3: golden eventi (FASE1_PIANO.md §12)
# ---------------------------------------------------------------------------

from mondo.grafo import evento_a_grafo
from mondo.tipi import Evento
from lingua.contesto import StatoDiscorso
from lingua.verbalizza import verbalizza_evento

SEQUENZA_A: list[tuple[Evento, str]] = [
    (Evento(t=9, azione="andare", agente="sara", luogo="giardino", luogo_origine="cucina"),
     "Alle nove Sara va dalla cucina in giardino."),
    (Evento(t=9, azione="prendere", agente="piero", oggetto="mela_1", argomento="melo", luogo="orto"),
     "Intanto Piero raccoglie una mela dal melo."),
    (Evento(t=10, azione="giocare", agente="sara", oggetto="palla", luogo="giardino"),
     "Alle dieci Sara gioca con la palla."),
    (Evento(t=10, azione="prendere", agente="anna", oggetto="acqua_1", argomento="pozzo", luogo="orto"),
     "Intanto Anna prende dell'acqua dal pozzo."),
    (Evento(t=11, azione="prendere", agente="piero", oggetto="mela_2", argomento="melo", luogo="orto"),
     "Alle undici Piero raccoglie una mela dal melo."),
    (Evento(t=11, azione="mettere_dentro", agente="piero", oggetto="mela_1", argomento="cestino", luogo="orto"),
     "Intanto Piero mette la prima mela nel cestino."),
    (Evento(t=12, azione="andare", agente="piero", luogo="giardino", luogo_origine="orto"),
     "Alle dodici Piero va in giardino."),
    (Evento(t=12, azione="dare", agente="piero", oggetto="mela_2", destinatario="sara", luogo="giardino"),
     "Intanto Piero dà la seconda mela a Sara."),
    (Evento(t=13, azione="mangiare", agente="sara", oggetto="mela_2", luogo="giardino"),
     "Alle tredici Sara mangia la seconda mela."),
    (Evento(t=14, azione="dormire", agente="luca", luogo="camera", argomento="stanchezza"),
     "Alle quattordici Luca si addormenta in camera perché è stanco."),
    (Evento(t=16, azione="svegliarsi", agente="luca", luogo="camera"),
     "Alle sedici Luca si sveglia."),
    (Evento(t=16, azione="cercare", agente="maria", oggetto="libro", luogo="salotto"),
     "Intanto Maria cerca il libro in salotto."),
    (Evento(t=17, azione="aprire", agente="maria", oggetto="scatola", luogo="salotto"),
     "Alle diciassette Maria apre la scatola."),
    (Evento(t=17, azione="dire", agente="maria", destinatario="marco", luogo="salotto"),
     "Intanto Maria dice qualcosa a Marco."),
    (Evento(t=18, azione="tirare_fuori", agente="marco", oggetto="pane", argomento="scatola", luogo="salotto"),
     "Alle diciotto Marco tira fuori il pane dalla scatola."),
    (Evento(t=18, azione="guardare", agente="sara", oggetto="palla", luogo="giardino"),
     "Intanto Sara guarda la palla."),
    (Evento(t=19, azione="chiudere", agente="maria", oggetto="scatola", luogo="salotto"),
     "Alle diciannove Maria chiude la scatola."),
    (Evento(t=19, azione="posare", agente="marco", oggetto="pane", luogo="salotto"),
     "Intanto Marco posa il pane."),
]

SEQUENZA_B: list[tuple[Evento, str]] = [
    (Evento(t=6, azione="prendere", agente="marco", oggetto="legna_1", argomento="bosco_legna", luogo="bosco"),
     "Alle sei Marco raccoglie della legna nel bosco."),
    (Evento(t=7, azione="andare", agente="marco", luogo="salotto", luogo_origine="bosco"),
     "Alle sette Marco va in salotto."),
    (Evento(t=8, azione="mettere_dentro", agente="marco", oggetto="legna_1", argomento="camino", luogo="salotto"),
     "Alle otto Marco mette la legna nel camino."),
    (Evento(t=8, azione="bruciare", agente="camino", oggetto="legna_1", luogo="salotto"),
     "Intanto il camino brucia la legna."),
]


class TestGoldenEventi:
    def _verifica_sequenza(self, sequenza):
        contesto = StatoDiscorso()
        for evento, frase_attesa in sequenza:
            frase = verbalizza_evento(evento_a_grafo(evento), contesto)
            assert frase == frase_attesa, f"t={evento.t} azione={evento.azione}: {frase!r} != {frase_attesa!r}"

    def test_sequenza_a(self):
        self._verifica_sequenza(SEQUENZA_A)

    def test_sequenza_b_contesto_nuovo(self):
        self._verifica_sequenza(SEQUENZA_B)


# ---------------------------------------------------------------------------
# Gruppo 5 (parziale): round-trip di massa sugli eventi, seed 0-299
# ---------------------------------------------------------------------------

from mondo.generatore import _lunghezza_storia
from mondo.simulatore import genera_storia
from lingua.analizza import analizza_evento, analizza_storia
from lingua.verbalizza import verbalizza_storia as _verbalizza_storia


class TestRoundTripEventi:
    def test_round_trip_eventi_seed_0_299(self):
        for seed in range(300):
            n_tick = _lunghezza_storia(seed)
            storia = genera_storia(seed=seed, n_tick=n_tick)
            grafi = [evento_a_grafo(e) for e in storia.eventi]
            frasi = _verbalizza_storia(grafi, StatoDiscorso())
            ricostruiti = analizza_storia(frasi, StatoDiscorso())
            assert ricostruiti == grafi, f"seed {seed}: round-trip fallito"

    def test_determinismo_verbalizza_storia(self):
        storia = genera_storia(seed=42, n_tick=20)
        grafi = [evento_a_grafo(e) for e in storia.eventi]
        frasi1 = _verbalizza_storia(grafi, StatoDiscorso())
        frasi2 = _verbalizza_storia(grafi, StatoDiscorso())
        assert frasi1 == frasi2

    def test_frase_malformata_solleva_valueerror(self):
        with pytest.raises(ValueError):
            analizza_evento("Questa non è una frase valida.", StatoDiscorso())
