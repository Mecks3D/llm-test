"""Le sette sonde di FASE_MENTE.md §8.

Mai un numero solo: ogni sonda spezza l'accuratezza lungo l'asse che le
interessa, e le due scale (narrativa e reale, vedi `banco.Esito`) restano
sempre distinte.

Ogni funzione ritorna righe `(etichetta, n, valore, ...)` così che lo stesso
codice serva il riferimento simbolico oggi e `mente/` domani.
"""
from __future__ import annotations

from collections import defaultdict

from mondo import dati_mondo as dm
from percezione.tipi import ConfigPercezione
from regole.tracker import NON_LO_SO

from .banco import (
    CANALI_TUTTI,
    MAI_VISTA,
    Esito,
    EsitoPredittivo,
    lunghezza,
    valuta_campione,
    valuta_storia,
)

Riga = tuple


def _acc(esiti: list[Esito], reale: bool) -> float:
    if not esiti:
        return float("nan")
    return sum((e.esatto_reale if reale else e.esatto) for e in esiti) / len(esiti)


def _per_gruppo(esiti: list[Esito], chiave, reale: bool, ordine: list | None = None) -> list[Riga]:
    gruppi: dict = defaultdict(list)
    for e in esiti:
        gruppi[chiave(e)].append(e)
    chiavi = [k for k in ordine if k in gruppi] if ordine else sorted(gruppi)
    return [(k, len(gruppi[k]), _acc(gruppi[k], reale)) for k in chiavi]


# -- P1 permanenza ----------------------------------------------------------

FASCE_ETA = ((0, 0), (1, 2), (3, 5), (6, 10), (11, 10**6))


def _fascia(eta: int) -> str:
    if eta == MAI_VISTA:
        return "mai vista"
    for lo, hi in FASCE_ETA:
        if lo <= eta <= hi:
            return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**6 else f"{lo}+")
    return "?"


CAMERE_PARZIALI = ("cucina", "salotto")


def p1_permanenza(seed_base: int = 2000, n_storie: int = 20) -> list[Riga]:
    """Accuratezza in funzione dei tick trascorsi dall'ultima volta che la
    telecamera ha visto l'entità. Una credenza che persiste tiene la curva
    piatta; una che evapora la fa crollare.

    Con una telecamera in ogni stanza tutto è sempre in vista e la sonda non
    misura nulla (verificato: 299 casi su 303 a età zero). Serve la copertura
    parziale del deployment vero — due stanze su sei — e il solo canale
    visivo, perché la lingua rinfrescherebbe le credenze scavalcando la prova.
    """
    config = ConfigPercezione(viste=CAMERE_PARZIALI)
    esiti, _ = valuta_campione(seed_base, n_storie, config=config, canali=("visione",))
    ordine = [_fascia(lo) for lo, _ in FASCE_ETA] + ["mai vista"]
    return _per_gruppo(esiti, lambda e: _fascia(e.eta_evidenza), reale=True, ordine=ordine)


# -- P2 binding -------------------------------------------------------------

def p2_binding(esiti: list[Esito], purezza: float) -> list[Riga]:
    """Due istanze della stessa classe nella stessa storia: è il caso in cui
    il sistema deve tenere separati due individui che il sensore chiama con lo
    stesso nome. `purezza` è la frazione di rilevazioni finite nello slot giusto."""
    righe = _per_gruppo(esiti, lambda e: "ambigua" if e.ambigua else "unica", reale=True)
    return righe + [("purezza rilevazioni", 0, purezza)]


# -- P3 interferenza --------------------------------------------------------

def p3_interferenza(
    seed_base: int = 2000, n_storie: int = 20, **kwargs
) -> list[Riga]:
    """Accuratezza al crescere del cast da 1 a 6 persone.

    È il criterio di accettazione #3 del piano JEPA, che non era mai stato
    misurato. `v1` qui crollava (0,98 a cast 1, 0,57 a cast pieno): è la firma
    dell'interferenza fra entità.
    """
    righe: list[Riga] = []
    for k in range(1, len(dm.PERSONE) + 1):
        esiti, _ = valuta_campione(
            seed_base, n_storie, persone=tuple(dm.PERSONE[:k]), **kwargs
        )
        righe.append((f"cast {k}", len(esiti), _acc(esiti, reale=True)))
    return righe


# -- P4 calibrazione --------------------------------------------------------

def p4_calibrazione(esiti: list[Esito]) -> list[Riga]:
    """Curva astensione/accuratezza: se ci si astiene sulle risposte meno
    sicure, l'accuratezza sul resto deve salire. Se non sale, la confidenza
    non significa nulla."""
    risposte = sorted(
        (e for e in esiti if e.predetto != NON_LO_SO), key=lambda e: -e.confidenza
    )
    totale = len(esiti)
    righe: list[Riga] = []
    for quota in (0.0, 0.1, 0.25, 0.5):
        tenute = risposte[: int(len(risposte) * (1 - quota))]
        copertura = len(tenute) / totale if totale else 0.0
        righe.append((f"astensione {quota:.0%}", len(tenute), _acc(tenute, reale=True), copertura))

    astenuti = [e for e in esiti if e.predetto == NON_LO_SO]
    if astenuti:
        # quando si astiene, quanto spesso era davvero indeterminabile?
        giustificate = sum(1 for e in astenuti if not e.derivabile) / len(astenuti)
        righe.append(("astensioni giustificate", len(astenuti), giustificate))
    return righe


# -- P5 robustezza ----------------------------------------------------------

LIVELLI_RUMORE = (
    ("pulito", ConfigPercezione()),
    ("mancate 10%", ConfigPercezione(p_mancata=0.10)),
    ("mancate 30%", ConfigPercezione(p_mancata=0.30)),
    ("falsi pos. 20%", ConfigPercezione(p_falso_positivo=0.20)),
    ("confusione 20%", ConfigPercezione(p_confusione=0.20)),
    ("tutto insieme", ConfigPercezione(p_mancata=0.20, p_falso_positivo=0.15, p_confusione=0.15)),
    ("senza ev. negativa", ConfigPercezione(completa=False)),
)


def p5_robustezza(
    seed_base: int = 2000, n_storie: int = 20, canali: tuple[str, ...] = CANALI_TUTTI
) -> list[Riga]:
    """Il rumore del sensore è un asse sperimentale, non una costante."""
    righe: list[Riga] = []
    for nome, config in LIVELLI_RUMORE:
        dubbio = 1 if config.pulita() else 2
        esiti, purezza = valuta_campione(
            seed_base, n_storie, config=config, canali=canali, assenze_per_dubbio=dubbio
        )
        righe.append((nome, len(esiti), _acc(esiti, reale=True), purezza))
    return righe


def p5b_politica_assenza(
    seed_base: int = 2000, n_storie: int = 20, canali: tuple[str, ...] = CANALI_TUTTI
) -> list[Riga]:
    """Che farsene del "non lo vedo più": ablazione delle tre politiche.

    Domanda di buon senso, non di ingegneria: se la palla sparisce dalla
    vista, l'ho persa o è solo nascosta? Nel micro-mondo i contenitori si
    chiudono, quindi sparire è normale — e infatti azzerare la credenza costa.
    """
    righe: list[Riga] = []
    for politica in ("ignora", "dubita", "azzera"):
        esiti, _ = valuta_campione(
            seed_base, n_storie, canali=canali, politica_assenza=politica
        )
        righe.append((politica, len(esiti), _acc(esiti, reale=True)))
    return righe


# -- P6 propagazione --------------------------------------------------------

def p6_propagazione(esiti: list[Esito]) -> list[Riga]:
    """Profondità della catena di contenimento reale: 1 = appoggiato in un
    luogo, 2 = in mano o dentro un contenitore, 3 = dentro un contenitore
    tenuto da qualcuno. È la capacità che il `jepa/` non poteva avere, perché
    i contenitori non esistevano nel suo spazio dei bersagli."""
    return _per_gruppo(
        [e for e in esiti if e.tipo == "posizione"], lambda e: f"catena {e.profondita}", reale=True
    )


# -- P7 sorpresa ------------------------------------------------------------

def p7_sorpresa(
    seed_base: int = 2000, n_storie: int = 20, canali: tuple[str, ...] = CANALI_TUTTI, **kwargs
) -> list[Riga]:
    """Predizione della prossima osservazione, ristretta a ciò che CAMBIA.

    Senza restringere, la loss è banalmente soddisfatta prevedendo "non cambia
    nulla": il confronto con la baseline che copia il frame precedente rende
    la cosa visibile invece che opinabile (FASE_MENTE.md §6).
    """
    acc = EsitoPredittivo()
    for i in range(n_storie):
        seed = seed_base + i
        valuta_storia(seed, lunghezza(seed), canali=canali, predittivo=acc, **kwargs)
    return [
        ("tutte le classi", acc.totali, acc.corrette / max(1, acc.totali)),
        ("solo sorpresa", acc.totali_sorpresa, acc.corrette_sorpresa / max(1, acc.totali_sorpresa)),
        (
            "copia frame (0 atteso)",
            acc.totali_copia,
            acc.corrette_copia / max(1, acc.totali_copia),
        ),
    ]
