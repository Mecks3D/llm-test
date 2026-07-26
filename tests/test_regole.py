"""Test del riferimento simbolico e delle sonde (FASE_MENTE.md §8-§9)."""
from __future__ import annotations

from mondo import dati_mondo as dm
from percezione.sintetica import storia_e_osservazioni
from percezione.tipi import Osservazione, Rilevazione
from regole.allineamento import allinea, purezza_binding
from regole.tracker import NON_LO_SO, TrackerRegole
from sonde.banco import CANALI_TUTTI, valuta_campione, valuta_storia

LUOGHI = tuple(l.id for l in dm.LUOGHI)
PERSONE = frozenset(p.id for p in dm.PERSONE)


def _tracker(**kwargs) -> TrackerRegole:
    return TrackerRegole(luoghi=LUOGHI, classi_persona=PERSONE, **kwargs)


def _oss(t, vista, classi, completa=True, verita=None):
    return Osservazione(
        t=t,
        vista=vista,
        rilevazioni=tuple(Rilevazione(c) for c in classi),
        completa=completa,
        verita=tuple(verita) if verita is not None else None,
    )


# -- persistenza, la proprietà centrale (§5.3) -------------------------------

def test_persistenza_senza_evidenza():
    """Senza nuova evidenza la credenza NON cambia: è la permanenza degli
    oggetti, cablata invece che sperata."""
    tr = _tracker(politica_assenza="ignora")
    tr.osserva(_oss(1, "cucina", ["mela"]))
    prima = tr.dove(tr.slot[0]).valore
    for t in range(2, 20):
        tr.osserva(_oss(t, "salotto", []))
    assert tr.dove(tr.slot[0]).valore == prima == "cucina"


def test_confidenza_decresce_col_tempo():
    tr = _tracker(politica_assenza="ignora")
    tr.osserva(_oss(1, "cucina", ["mela"]))
    presto = tr.dove(tr.slot[0]).confidenza
    for t in range(2, 15):
        tr.osserva(_oss(t, "salotto", []))
    assert tr.dove(tr.slot[0]).confidenza < presto


def test_evidenza_negativa_azzera_solo_se_richiesto():
    tr = _tracker(politica_assenza="azzera")
    tr.osserva(_oss(1, "cucina", ["mela"]))
    tr.osserva(_oss(2, "cucina", []))
    assert tr.dove(tr.slot[0]).valore == NON_LO_SO

    tr = _tracker(politica_assenza="dubita")
    tr.osserva(_oss(1, "cucina", ["mela"]))
    tr.osserva(_oss(2, "cucina", []))
    assert tr.dove(tr.slot[0]).valore == "cucina"


def test_smentire_il_luogo_non_smentisce_il_possesso():
    """La cucina vuota rifiuta "Anna è in cucina" — e quindi anche "la mela è
    in cucina". Ma la mela resta in mano ad Anna: le due relazioni sono
    indipendenti, ed è il motivo per cui il bersaglio di uno slot è un altro
    slot e non un luogo (§5.1)."""
    tr = _tracker(politica_assenza="azzera")
    tr.ascolta_evento("andare", agente="anna", luogo="cucina", t=1)
    tr.ascolta_evento("prendere", agente="anna", oggetto="mela_1", luogo="cucina", t=1)
    tr.osserva(_oss(2, "cucina", []))
    mela = tr._slot_per_nome("mela_1")
    assert tr.dove(mela).valore == NON_LO_SO
    assert tr.chi_ha(mela).valore == "anna"


# -- catena di contenimento (§5.2) ------------------------------------------

def test_catena_di_contenimento_transitiva():
    tr = _tracker()
    tr.ascolta_evento("andare", agente="anna", luogo="cucina", t=1)
    tr.ascolta_evento("prendere", agente="anna", oggetto="cestino", luogo="cucina", t=1)
    tr.ascolta_evento(
        "mettere_dentro", agente="anna", oggetto="mela_1", argomento="cestino", luogo="cucina", t=2
    )
    tr.ascolta_evento("andare", agente="anna", luogo="orto", t=3)

    mela = tr._slot_per_nome("mela_1")
    assert tr.dove(mela).valore == "orto"  # la mela segue il cestino che segue Anna
    assert tr.chi_ha(mela).valore == "anna"


def test_catena_ciclica_non_esplode():
    tr = _tracker()
    a, b = tr._nuovo("scatola", "lingua"), tr._nuovo("cestino", "lingua")
    a.rel_tipo, a.rel_valore = "slot", b.id
    b.rel_tipo, b.rel_valore = "slot", a.id
    assert tr.dove(a).valore == NON_LO_SO
    assert tr.chi_ha(a).valore == NON_LO_SO


def test_oggetto_distrutto():
    tr = _tracker()
    tr.ascolta_evento("andare", agente="anna", luogo="cucina", t=1)
    tr.ascolta_evento("prendere", agente="anna", oggetto="mela_1", luogo="cucina", t=1)
    tr.ascolta_evento("mangiare", agente="anna", oggetto="mela_1", luogo="cucina", t=2)
    assert tr.dove(tr._slot_per_nome("mela_1")).valore == NON_LO_SO


# -- associazione dati (§5.4) ------------------------------------------------

def test_due_istanze_restano_separate():
    """Due mele in due stanze: il sensore le chiama uguale, il tracker no."""
    tr = _tracker()
    tr.osserva(_oss(1, "cucina", ["mela"]))
    tr.osserva(_oss(1, "orto", ["mela"]))
    assert len([s for s in tr.slot if s.classe == "mela"]) == 2
    luoghi = {tr.dove(s).valore for s in tr.slot if s.classe == "mela"}
    assert luoghi == {"cucina", "orto"}


def test_rivedere_nella_stessa_vista_non_duplica():
    tr = _tracker()
    for t in range(1, 6):
        tr.osserva(_oss(t, "cucina", ["mela"]))
    assert len(tr.slot) == 1


def test_la_vista_conferma_e_non_sovrascrive_la_lingua():
    """Vedere la mela in cucina non deve cancellare "è in mano ad Anna":
    la vista non sa distinguere i due casi (§5.5)."""
    tr = _tracker()
    tr.ascolta_evento("andare", agente="anna", luogo="cucina", t=1)
    tr.ascolta_evento("prendere", agente="anna", oggetto="mela_1", luogo="cucina", t=1)
    tr.osserva(_oss(2, "cucina", ["anna", "mela"]))
    mela = tr._slot_per_nome("mela_1")
    assert tr.chi_ha(mela).valore == "anna"
    assert tr.dove(mela).valore == "cucina"


def test_fusione_canali_stesso_slot():
    """Il nome linguistico si lega allo slot che la vista ha già costruito,
    altrimenti i due canali tengono credenze separate e si sabotano."""
    tr = _tracker()
    tr.osserva(_oss(1, "cucina", ["mela"]))
    n_prima = len(tr.slot)
    tr.ascolta_evento("guardare", agente="anna", oggetto="mela_1", luogo="cucina", t=1)
    assert len([s for s in tr.slot if s.classe == "mela"]) == 1
    assert len(tr.slot) == n_prima + 1  # solo lo slot nuovo di anna


# -- la verità diagnostica non deve influenzare le decisioni -----------------

def test_verita_non_influenza():
    """`Osservazione.verita` serve solo all'allineamento post-hoc: togliendola
    il tracker deve comportarsi in modo identico."""
    _, flusso = storia_e_osservazioni(2000, 10)
    con = _tracker()
    senza = _tracker()
    for oss in flusso:
        con.osserva(oss)
        senza.osserva(oss.senza_verita())
    stato = lambda tr: [(s.classe, s.rel_tipo, s.rel_valore) for s in tr.slot]
    assert stato(con) == stato(senza)


def test_allineamento_e_purezza():
    _, flusso = storia_e_osservazioni(2000, 10)
    tr = _tracker()
    for oss in flusso:
        tr.osserva(oss)
    mappa = allinea(tr)
    assert mappa  # qualcosa è stato allineato
    assert all(0 <= slot_id < len(tr.slot) for slot_id in mappa.values())
    purezza, n = purezza_binding(tr)
    assert n > 0 and 0.0 <= purezza <= 1.0


# -- predizione (§6) ---------------------------------------------------------

def test_predici_elenca_le_classi_credute():
    tr = _tracker()
    tr.osserva(_oss(1, "cucina", ["mela", "anna"]))
    assert set(tr.predici("cucina")) == {"mela", "anna"}
    assert tr.predici("orto") == ()


# -- banco e sonde -----------------------------------------------------------

def test_determinismo_del_banco():
    a = valuta_storia(2000, 12)
    b = valuta_storia(2000, 12)
    assert a == b


def test_due_scale_sono_distinte():
    """L'oro narrativo e lo stato vero non coincidono: se coincidessero, la
    distinzione introdotta in §8 sarebbe inutile e il confronto fra canali
    tornerebbe a mentire."""
    esiti, _ = valuta_campione(2000, 10)
    assert any(e.oro != e.oro_vero for e in esiti)
    assert all(e.oro_vero != NON_LO_SO for e in esiti)


def test_fusione_batte_i_canali_singoli():
    """Il numero che giustifica l'intera architettura: due canali di evidenza
    su una credenza sola valgono più di ciascuno da solo."""
    soli_lingua, _ = valuta_campione(2000, 15, canali=("lingua",))
    sola_vista, _ = valuta_campione(2000, 15, canali=("visione",))
    fusi, _ = valuta_campione(2000, 15, canali=CANALI_TUTTI)
    acc = lambda e: sum(x.esatto_reale for x in e) / len(e)
    assert acc(fusi) > acc(soli_lingua)
    assert acc(fusi) > acc(sola_vista)


def test_binding_e_piu_difficile_sulle_istanze_ambigue():
    esiti, _ = valuta_campione(2000, 25)
    ambigue = [e for e in esiti if e.ambigua]
    uniche = [e for e in esiti if not e.ambigua]
    assert ambigue and uniche
    acc = lambda e: sum(x.esatto_reale for x in e) / len(e)
    assert acc(ambigue) < acc(uniche)


def test_interferenza_cast_cresce():
    """Criterio di accettazione mai misurato del piano JEPA: il degrado da
    cast 1 a cast pieno deve esserci ma restare piccolo."""
    from sonde.sonde import p3_interferenza

    righe = p3_interferenza(2000, 10)
    assert righe[0][2] > righe[-1][2]  # cast 1 meglio di cast 6
    assert righe[0][2] - righe[-1][2] < 0.30  # ma non il crollo di `v1` (0,98 -> 0,57)


def test_sonde_girano_tutte():
    """Fumo: le sette sonde producono righe non vuote e valori in [0,1]."""
    from sonde.sonde import (
        p1_permanenza,
        p2_binding,
        p3_interferenza,
        p4_calibrazione,
        p5_robustezza,
        p5b_politica_assenza,
        p6_propagazione,
        p7_sorpresa,
    )

    esiti, purezza = valuta_campione(2000, 6)
    tabelle = [
        p1_permanenza(2000, 6),
        p2_binding(esiti, purezza),
        p3_interferenza(2000, 3),
        p4_calibrazione(esiti),
        p5_robustezza(2000, 3),
        p5b_politica_assenza(2000, 3),
        p6_propagazione(esiti),
        p7_sorpresa(2000, 3),
    ]
    for righe in tabelle:
        assert righe
        for riga in righe:
            assert all(0.0 <= v <= 1.0 for v in riga[2:] if isinstance(v, float))


def test_adattatore_v1_regge_il_vocabolario_cresciuto():
    """Il vocabolario è cresciuto dopo i run di `v1` (`che-cosa`, `[STATO]`),
    ma solo in coda: gli id vecchi non si sono spostati. L'adattatore deve
    caricare il checkpoint alla SUA dimensione, non a quella attuale."""
    import json
    import subprocess

    attuale = json.load(open("cervello/vocabolario.json"))["token"]
    for commit in ("635afbd", "697451d"):
        vecchio = json.loads(
            subprocess.run(
                ["git", "show", f"{commit}:cervello/vocabolario.json"],
                capture_output=True, text=True, check=True,
            ).stdout
        )["token"]
        assert attuale[: len(vecchio)] == vecchio, (
            f"il vocabolario di {commit} non è più un prefisso di quello attuale: "
            "i checkpoint storici non sono più caricabili e sonde/adattatori.py va rivisto"
        )
