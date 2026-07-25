"""Test del canale percettivo (FASE_MENTE.md §4)."""
from __future__ import annotations

import random

import pytest

from mondo import dati_mondo as dm
from mondo.simulatore import genera_storia
from percezione.rumore import degrada
from percezione.sintetica import (
    _entita_in_vista,
    _oggetto_visibile,
    osserva_storia,
    stati_per_tick,
    vocabolario_classi,
)
from percezione.tipi import CLASSE_PERSONA_GENERICA, ConfigPercezione, Osservazione, Rilevazione

SEED = 2000
N_TICK = 10


def test_tipi_non_importa_il_progetto():
    """`tipi.py` è la cucitura sim-to-real: deve restare stdlib puro."""
    import ast
    import pathlib

    sorgente = pathlib.Path("percezione/tipi.py").read_text(encoding="utf-8")
    moduli = set()
    for nodo in ast.walk(ast.parse(sorgente)):
        if isinstance(nodo, ast.Import):
            moduli |= {a.name.split(".")[0] for a in nodo.names}
        elif isinstance(nodo, ast.ImportFrom) and nodo.level == 0 and nodo.module:
            moduli.add(nodo.module.split(".")[0])
    assert moduli <= {"dataclasses", "typing", "__future__"}, moduli


def test_determinismo():
    config = ConfigPercezione(p_mancata=0.3, p_falso_positivo=0.2, p_confusione=0.3)
    assert osserva_storia(SEED, N_TICK, config) == osserva_storia(SEED, N_TICK, config)


def test_stati_per_tick_e_prefisso():
    stati = stati_per_tick(SEED, 5)
    assert len(stati) == 5
    assert [s.t for s in stati] == [1, 2, 3, 4, 5]
    # una storia corta è un prefisso di una lunga
    assert genera_storia(seed=SEED, n_tick=3).eventi == genera_storia(seed=SEED, n_tick=5).eventi[
        : len(genera_storia(seed=SEED, n_tick=3).eventi)
    ]


def test_osservazione_pulita_corrisponde_allo_stato():
    """Senza rumore la vista elenca esattamente le entità presenti e visibili."""
    stati = stati_per_tick(SEED, N_TICK)
    flusso = osserva_storia(SEED, N_TICK)
    per_chiave = {(o.t, o.vista): o for o in flusso}

    for t, stato in enumerate(stati, start=1):
        for luogo in (l.id for l in dm.LUOGHI):
            attese = _entita_in_vista(stato, luogo)
            assert per_chiave[(t, luogo)].verita == tuple(attese)


def test_nessuna_rilevazione_porta_identita_di_istanza():
    """Il sensore emette CLASSI, mai istanze: `mela`, non `mela_3`."""
    classi_ammesse = set(vocabolario_classi(ConfigPercezione()))
    for oss in osserva_storia(SEED, N_TICK):
        for ril in oss.rilevazioni:
            assert ril.classe in classi_ammesse
            assert "_" not in ril.classe


def test_persona_addormentata_resta_visibile():
    """Una telecamera vede chi dorme; `testimoni_in` no, perché non testimonia."""
    trovato = False
    for seed in range(2000, 2040):
        for stato in stati_per_tick(seed, 12):
            dormienti = [p for p in stato.persone.values() if p.addormentato]
            for p in dormienti:
                trovato = True
                assert p.id in _entita_in_vista(stato, p.luogo)
                assert p.id not in stato.testimoni_in(p.luogo)
    assert trovato, "nessun dormiente nel campione: test non informativo"


def test_occlusione_contenitore_chiuso():
    """Un oggetto dentro un contenitore chiuso non si vede, ma c'è."""
    trovato = False
    for seed in range(2000, 2060):
        for stato in stati_per_tick(seed, 14):
            for oid, ogg in stato.oggetti.items():
                tipo, rif = ogg.posizione
                if tipo != "contenitore":
                    continue
                contenitore = stato.oggetti[rif]
                if contenitore.apribile and not contenitore.aperto:
                    trovato = True
                    assert not _oggetto_visibile(stato, oid)
                    luogo = stato.luogo_effettivo(oid)
                    assert oid not in _entita_in_vista(stato, luogo)
                else:
                    assert _oggetto_visibile(stato, oid)
    assert trovato, "nessun contenitore chiuso con contenuto nel campione"


def test_verita_allineata_e_rimovibile():
    for oss in osserva_storia(SEED, N_TICK, ConfigPercezione(p_falso_positivo=0.3)):
        assert len(oss.verita) == len(oss.rilevazioni)
        assert oss.senza_verita().verita is None
        assert oss.senza_verita().rilevazioni == oss.rilevazioni

    with pytest.raises(ValueError):
        Osservazione(t=1, vista="cucina", rilevazioni=(Rilevazione("mela"),), verita=())


def test_manopola_mancata_totale():
    config = ConfigPercezione(p_mancata=1.0)
    for oss in osserva_storia(SEED, N_TICK, config):
        assert all(v is None for v in oss.verita)  # restano solo i falsi positivi


def test_manopola_falso_positivo_totale():
    config = ConfigPercezione(p_falso_positivo=1.0)
    flusso = osserva_storia(SEED, N_TICK, config)
    assert any(v is None for oss in flusso for v in oss.verita)


def test_persone_non_identificate():
    config = ConfigPercezione(persone_identificate=False)
    persone = {p.id for p in dm.PERSONE}
    for oss in osserva_storia(SEED, N_TICK, config):
        assert not (set(oss.classi()) & persone)
    assert any(CLASSE_PERSONA_GENERICA in oss.classi() for oss in osserva_storia(SEED, N_TICK, config))


def test_confusione_cambia_solo_classi_plausibili():
    from percezione.tipi import CONFUSIONI_PLAUSIBILI

    config = ConfigPercezione(p_confusione=1.0, persone_identificate=True)
    for oss in osserva_storia(SEED, N_TICK, config):
        for ril, vero in zip(oss.rilevazioni, oss.verita):
            if vero is None:
                continue
            classe_vera = vero.split("_")[0]
            if ril.classe == classe_vera:
                continue  # nessuna alternativa plausibile per questa classe
            assert any(
                {ril.classe, classe_vera} <= gruppo for gruppo in CONFUSIONI_PLAUSIBILI
            ), (ril.classe, classe_vera)


def test_tracce_persistono_e_si_rompono():
    """Un id di traccia vive finché l'entità resta continuativamente in vista."""
    config = ConfigPercezione(id_traccia=True)
    flusso = osserva_storia(SEED, 12, config)
    storico: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for oss in flusso:
        for ril, vero in zip(oss.rilevazioni, oss.verita):
            storico.setdefault((oss.vista, vero), []).append((oss.t, ril.id_traccia))

    interrotto = False
    for (_, vero), voci in storico.items():
        for (t1, tr1), (t2, tr2) in zip(voci, voci[1:]):
            if t2 == t1 + 1:
                assert tr1 == tr2, (vero, t1, t2)
            else:
                interrotto = True
                assert tr1 != tr2
    assert interrotto, "nessuna traccia interrotta nel campione: test non informativo"


def test_completa_e_configurabile():
    assert all(o.completa for o in osserva_storia(SEED, 4, ConfigPercezione(completa=True)))
    assert not any(o.completa for o in osserva_storia(SEED, 4, ConfigPercezione(completa=False)))


def test_riquadro_solo_se_richiesto():
    assert all(
        r.riquadro is None for o in osserva_storia(SEED, 4) for r in o.rilevazioni
    )
    assert all(
        r.riquadro is not None
        for o in osserva_storia(SEED, 4, ConfigPercezione(riquadro=True))
        for r in o.rilevazioni
    )


def test_degrada_e_puro_rispetto_allrng():
    """Stesso RNG, stesso risultato: nessuno stato nascosto in `degrada`."""
    pulita = Osservazione(
        t=1, vista="cucina", rilevazioni=(Rilevazione("mela"), Rilevazione("palla")),
        verita=("mela_1", "palla"),
    )
    config = ConfigPercezione(p_mancata=0.5, p_confusione=0.5, p_falso_positivo=0.5)
    a = degrada(pulita, config, random.Random("x"), ("mela", "palla", "pane"))
    b = degrada(pulita, config, random.Random("x"), ("mela", "palla", "pane"))
    assert a == b
