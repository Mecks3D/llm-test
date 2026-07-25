"""Telecamera sintetica: osservazioni derivate dallo stato del micro-mondo.

Backend di `percezione/` che sostituisce il detector reale finché non c'è
hardware (decisione 4: si resta a lungo nel simulato). Emette, per ogni
tick e ogni vista, l'insieme delle entità visibili — poi `rumore.py` lo
degrada come farebbe un sensore vero.

Unico modulo di `percezione/` che importa `mondo/`: `mente/` non lo vedrà
mai. Usa solo l'API pubblica `genera_storia`.
"""
from __future__ import annotations

import random
from dataclasses import replace

from mondo import dati_mondo as dm
from mondo.simulatore import Storia, genera_storia
from mondo.tipi import StatoMondo

from .rumore import degrada
from .tipi import CLASSE_PERSONA_GENERICA, ConfigPercezione, Osservazione, Rilevazione


def stati_per_tick(
    seed: int, n_tick: int, persone: tuple[dm.Persona, ...] | None = None
) -> list[StatoMondo]:
    """Stato del mondo dopo ciascun tick 1..n_tick.

    Rigenera la storia per ogni lunghezza: una storia più corta è un prefisso
    esatto di una più lunga (stesso seed, stesso RNG), come già sfrutta
    `esami/diagnosi.py::_posizioni_per_tick`. Costa O(n_tick²) passi di
    simulazione — con n_tick ≤ 22 sono ~250 passi per storia, trascurabili,
    e in cambio non si duplica la logica del motore qui dentro.
    """
    return [
        genera_storia(seed=seed, n_tick=t, persone=persone).stato_finale
        for t in range(1, n_tick + 1)
    ]


def vocabolario_classi(config: ConfigPercezione) -> tuple[str, ...]:
    """Etichette che il sensore può emettere: serve ai falsi positivi."""
    oggetti = {o.lemma for o in dm.OGGETTI_UNICI} | {v["lemma_unita"] for v in dm.RISORSE.values()}
    persone = (
        {p.id for p in dm.PERSONE} if config.persone_identificate else {CLASSE_PERSONA_GENERICA}
    )
    return tuple(sorted(oggetti | persone))


def _oggetto_visibile(stato: StatoMondo, oid: str) -> bool:
    """Un oggetto è visibile se la catena che lo porta a un luogo fisico non
    attraversa un contenitore chiuso (occlusione vera, non simulata)."""
    visto: set[str] = set()
    corrente = oid
    while True:
        if corrente in visto:  # difesa: catena ciclica, non deve accadere
            return False
        visto.add(corrente)
        tipo, rif = stato.oggetti[corrente].posizione
        if tipo in ("luogo", "persona"):
            return True
        contenitore = stato.oggetti[rif]
        if contenitore.apribile and not contenitore.aperto:
            return False
        corrente = rif


def _classe(stato: StatoMondo, eid: str, config: ConfigPercezione) -> str:
    if eid in stato.persone:
        return eid if config.persone_identificate else CLASSE_PERSONA_GENERICA
    return stato.oggetti[eid].lemma


def _entita_in_vista(stato: StatoMondo, luogo: str) -> list[str]:
    """Chi e cosa una telecamera puntata su `luogo` inquadra, in ordine
    deterministico. Le persone addormentate si vedono eccome: `testimoni_in`
    le esclude perché non *testimoniano*, ma questa è un'altra cosa."""
    entita = [pid for pid, p in sorted(stato.persone.items()) if p.luogo == luogo]
    entita += [
        oid
        for oid, _ in sorted(stato.oggetti.items())
        if stato.luogo_effettivo(oid) == luogo and _oggetto_visibile(stato, oid)
    ]
    return entita


def osserva_stato(
    stato: StatoMondo, t: int, vista: str, config: ConfigPercezione, seme: str
) -> Osservazione:
    """Osservazione pulita di una vista, poi degradata secondo `config`."""
    entita = _entita_in_vista(stato, vista)
    if config.solo_viste_abitate and not entita:
        return Osservazione(t=t, vista=vista, rilevazioni=(), completa=config.completa, verita=())

    rilevazioni = tuple(Rilevazione(classe=_classe(stato, e, config)) for e in entita)
    pulita = Osservazione(
        t=t,
        vista=vista,
        rilevazioni=rilevazioni,
        completa=config.completa,
        verita=tuple(entita),
    )
    return degrada(
        pulita, config, random.Random(f"{seme}-{t}-{vista}"), vocabolario_classi(config)
    )


def _assegna_tracce(flusso: list[Osservazione]) -> list[Osservazione]:
    """Assegna id di traccia persistenti finché l'entità resta nella vista.

    Un tracker vero (tipo ByteTrack) mantiene un id mentre l'oggetto è
    continuamente inquadrato e ne apre uno nuovo quando lo ritrova dopo
    averlo perso: la continuità dentro una vista è data, l'identità fra viste
    e nel tempo no. Qui si riproduce esattamente quel contratto, usando
    `verita` — che resta materiale diagnostico e non finisce nel payload.
    """
    attive: dict[tuple[str, str], tuple[str, int]] = {}  # (vista, id_vero) -> (traccia, ultimo t)
    contatore = 0
    fuori: list[Osservazione] = []

    for oss in flusso:
        verita = oss.verita or (None,) * len(oss.rilevazioni)
        nuove: list[Rilevazione] = []
        for ril, vero in zip(oss.rilevazioni, verita):
            if vero is None:  # falso positivo: traccia effimera, mai riusata
                contatore += 1
                nuove.append(replace(ril, id_traccia=f"tr{contatore}"))
                continue
            chiave = (oss.vista, vero)
            precedente = attive.get(chiave)
            if precedente is not None and precedente[1] == oss.t - 1:
                traccia = precedente[0]
            else:
                contatore += 1
                traccia = f"tr{contatore}"
            attive[chiave] = (traccia, oss.t)
            nuove.append(replace(ril, id_traccia=traccia))
        fuori.append(replace(oss, rilevazioni=tuple(nuove)))
    return fuori


def osserva_storia(
    seed: int,
    n_tick: int,
    config: ConfigPercezione | None = None,
    persone: tuple[dm.Persona, ...] | None = None,
) -> tuple[Osservazione, ...]:
    """Flusso completo di osservazioni di una storia, in ordine (t, vista).

    Deterministico: stesso `seed` e stessa `config` -> stesso flusso.
    """
    config = config or ConfigPercezione()
    stati = stati_per_tick(seed, n_tick, persone)
    viste = config.viste if config.viste is not None else tuple(l.id for l in dm.LUOGHI)

    flusso: list[Osservazione] = []
    for t, stato in enumerate(stati, start=1):
        for vista in viste:
            flusso.append(osserva_stato(stato, t, vista, config, seme=f"perc-{seed}"))
    if config.id_traccia:
        flusso = _assegna_tracce(flusso)
    return tuple(flusso)


def storia_e_osservazioni(
    seed: int,
    n_tick: int,
    config: ConfigPercezione | None = None,
    persone: tuple[dm.Persona, ...] | None = None,
) -> tuple[Storia, tuple[Osservazione, ...]]:
    """Storia (per le domande e la verità) più il suo flusso percettivo."""
    storia = genera_storia(seed=seed, n_tick=n_tick, persone=persone)
    return storia, osserva_storia(seed, n_tick, config, persone)
