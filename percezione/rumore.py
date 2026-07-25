"""Degradazione di un'osservazione pulita, come farebbe un sensore vero.

Deterministico: l'RNG arriva sempre dal chiamante con un seme esplicito
(regola 2). Ogni manopola è indipendente dalle altre, così le sonde possono
muoverne una sola per volta (FASE_MENTE.md §4.2, §8 sonda P5).
"""
from __future__ import annotations

import random

from .tipi import CONFUSIONI_PLAUSIBILI, ConfigPercezione, Osservazione, Rilevazione

# Confidenze: un sensore onesto è mediamente meno sicuro quando sbaglia.
_BANDA_CORRETTA = (0.75, 0.99)
_BANDA_ERRATA = (0.40, 0.80)


def _alternative(classe: str) -> tuple[str, ...]:
    alt: set[str] = set()
    for gruppo in CONFUSIONI_PLAUSIBILI:
        if classe in gruppo:
            alt |= set(gruppo)
    alt.discard(classe)
    return tuple(sorted(alt))


def _confidenza(rng: random.Random, corretta: bool, informativa: bool) -> float:
    if not informativa:
        return 1.0
    lo, hi = _BANDA_CORRETTA if corretta else _BANDA_ERRATA
    return round(rng.uniform(lo, hi), 3)


def _riquadro(rng: random.Random) -> tuple[float, float, float, float]:
    """STUB: il micro-mondo non ha coordinate. Vedi ConfigPercezione.riquadro."""
    x, y = rng.uniform(0.0, 0.8), rng.uniform(0.0, 0.8)
    return (round(x, 3), round(y, 3), round(x + 0.2, 3), round(y + 0.2, 3))


def degrada(
    pulita: Osservazione,
    config: ConfigPercezione,
    rng: random.Random,
    vocabolario: tuple[str, ...] = (),
) -> Osservazione:
    """Applica mancate rilevazioni, confusione di classe e falsi positivi.

    `vocabolario` è l'insieme delle classi che il sensore può emettere: serve
    ai falsi positivi, che possono inventare cose mai presenti nella vista.
    L'assegnazione degli `id_traccia` NON avviene qui: richiede continuità
    fra tick e la fa `sintetica._assegna_tracce` sul flusso intero.
    """
    verita_in = pulita.verita if pulita.verita is not None else (None,) * len(pulita.rilevazioni)

    rilevazioni: list[Rilevazione] = []
    verita: list[str | None] = []

    for ril, vero in zip(pulita.rilevazioni, verita_in):
        if rng.random() < config.p_mancata:
            continue
        classe, corretta = ril.classe, True
        if rng.random() < config.p_confusione:
            alt = _alternative(ril.classe)
            if alt:
                classe, corretta = rng.choice(alt), False
        rilevazioni.append(
            Rilevazione(
                classe=classe,
                confidenza=_confidenza(rng, corretta, config.confidenza_informativa),
                riquadro=_riquadro(rng) if config.riquadro else None,
            )
        )
        verita.append(vero)

    if vocabolario and rng.random() < config.p_falso_positivo:
        rilevazioni.append(
            Rilevazione(
                classe=rng.choice(vocabolario),
                confidenza=_confidenza(rng, False, config.confidenza_informativa),
                riquadro=_riquadro(rng) if config.riquadro else None,
            )
        )
        verita.append(None)

    return Osservazione(
        t=pulita.t,
        vista=pulita.vista,
        rilevazioni=tuple(rilevazioni),
        completa=config.completa,
        verita=tuple(verita),
    )
