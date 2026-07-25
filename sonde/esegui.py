"""Esecuzione delle sette sonde sul riferimento simbolico.

    .venv/bin/python -m sonde.esegui [--storie N] [--seed-base S]

Stdlib puro, gira in locale su CPU in una manciata di secondi. Quando
esisterà `mente/`, questo stesso comando prende un `--sistema mente` e stampa
le due colonne affiancate: è così che la regola di §9 (mai un numero senza il
riferimento accanto) diventa impossibile da dimenticare.
"""
from __future__ import annotations

import argparse
import time

from .banco import CANALI_TUTTI, valuta_campione
from .sonde import (
    p1_permanenza,
    p2_binding,
    p3_interferenza,
    p4_calibrazione,
    p5_robustezza,
    p5b_politica_assenza,
    p6_propagazione,
    p7_sorpresa,
)

CANALI = (("lingua",), ("visione",), ("visione", "lingua"))


def _tabella(titolo: str, righe, intestazioni=("", "N", "esattezza")) -> None:
    print(f"\n{titolo}")
    larghezza = max([len(str(r[0])) for r in righe] + [len(intestazioni[0])])
    print("  " + intestazioni[0].ljust(larghezza) + "".join(f"{h:>12}" for h in intestazioni[1:]))
    for riga in righe:
        celle = "".join(
            f"{v:>12}" if isinstance(v, int) else f"{v:>12.3f}" for v in riga[1:]
        )
        print("  " + str(riga[0]).ljust(larghezza) + celle)


def principale(seed_base: int, n_storie: int, con_jepa: bool = False) -> None:
    avvio = time.time()
    print("=" * 74)
    print("SONDE — riferimento simbolico `regole/`  (FASE_MENTE.md §8-§9)")
    print(f"seed {seed_base}..{seed_base + n_storie - 1}, {n_storie} storie, cast pieno")
    print("=" * 74)

    print("\nQuadro d'insieme per canale d'evidenza")
    print("  %-18s %6s %11s %8s %9s" % ("canali", "N", "narrativa", "reale", "binding"))
    esiti_per_canale = {}
    for canali in CANALI:
        esiti, purezza = valuta_campione(seed_base, n_storie, canali=canali)
        esiti_per_canale[canali] = (esiti, purezza)
        n = len(esiti)
        print(
            "  %-18s %6d %11.3f %8.3f %9.3f"
            % (
                "+".join(canali),
                n,
                sum(e.esatto for e in esiti) / n,
                sum(e.esatto_reale for e in esiti) / n,
                purezza,
            )
        )
    print(
        "\n  narrativa = contro l'oro di mondo/domande.py (derivabile dal RACCONTO):\n"
        "              onesta solo per il canale lingua, penalizza chi ha visto di più.\n"
        "  reale     = contro lo stato vero del mondo: la scala del deployment."
    )

    esiti, purezza = esiti_per_canale[CANALI_TUTTI]

    _tabella(
        "P1 permanenza — tick dall'ultima vista (2 camere su 6, sola visione)",
        p1_permanenza(seed_base, n_storie),
        ("età", "N", "esatt."),
    )
    _tabella("P2 binding — istanze indistinguibili per il sensore", p2_binding(esiti, purezza))
    _tabella(
        "P3 interferenza — accuratezza al crescere del cast",
        p3_interferenza(seed_base, n_storie),
    )
    _tabella(
        "P4 calibrazione — astenersi sulle risposte meno sicure",
        p4_calibrazione(esiti),
        ("regime", "N", "esatt.", "copertura"),
    )
    _tabella(
        "P5 robustezza — degradazione del sensore",
        p5_robustezza(seed_base, n_storie),
        ("sensore", "N", "esatt.", "binding"),
    )
    _tabella(
        "P5b politica d'assenza — che farsene del \"non lo vedo più\"",
        p5b_politica_assenza(seed_base, n_storie),
    )
    _tabella("P6 propagazione — profondità della catena", p6_propagazione(esiti))
    _tabella("P7 sorpresa — predizione della prossima osservazione", p7_sorpresa(seed_base, n_storie))

    if con_jepa:
        _confronto_jepa(seed_base, n_storie, esiti_per_canale)

    print(f"\n{'=' * 74}\ntempo totale: {time.time() - avvio:.1f}s su CPU\n")


def _confronto_jepa(seed_base: int, n_storie: int, esiti_per_canale: dict) -> None:
    """M0.3: la storia del progetto su una scala unica (FASE_MENTE.md §11)."""
    from .adattatori import addestra_jepa_locale, esiti_jepa
    from .sonde import p2_binding, p6_propagazione

    print("\n" + "-" * 74)
    print("M0.3 — sistemi precedenti misurati con lo stesso strumento")
    print("-" * 74)
    modello = addestra_jepa_locale(seed_torch=0)
    jepa = esiti_jepa(seed_base, n_storie, modello)

    righe = [("jepa/ (lingua)", jepa)]
    righe += [(f"regole/ ({'+'.join(c)})", esiti_per_canale[c][0]) for c in CANALI]
    print("\n  %-26s %6s %11s %8s" % ("sistema", "N", "narrativa", "reale"))
    for nome, esiti in righe:
        print(
            "  %-26s %6d %11.3f %8.3f"
            % (
                nome,
                len(esiti),
                sum(e.esatto for e in esiti) / len(esiti),
                sum(e.esatto_reale for e in esiti) / len(esiti),
            )
        )
    _tabella("  P2 binding — jepa/", p2_binding(jepa, float("nan")))
    _tabella("  P6 propagazione — jepa/", p6_propagazione(jepa))
    print(
        "\n  `v1` non è misurabile qui: nessun checkpoint .pt in locale.\n"
        "  L'adattatore va scritto quando ce n'è uno (vedi FASE_MENTE.md §11)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storie", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument(
        "--con-jepa", action="store_true",
        help="addestra e misura anche jepa/ sulla stessa scala (~70s, richiede torch)",
    )
    argomenti = parser.parse_args()
    principale(argomenti.seed_base, argomenti.storie, argomenti.con_jepa)
