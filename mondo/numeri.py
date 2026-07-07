"""Numerali cardinali per il nodo NUM del tempo (grafo.py).

Il lessico (`lingua/lessico.tsv`) è l'unica fonte del vocabolario, anche per
`mondo/` (PROGETTO.md): questo modulo ne legge però solo il *file*, con un
parser minimo indipendente da `lingua/lessico.py`, per non violare la regola
"mondo/ non importa lingua/". Stessa selezione di righe di
`lingua.morfologia._costruisci_numeri`, ma la destinazione è il lemma
(es. "diciassette"), non la forma di superficie accentata usata nel testo
(es. "ventitré").
"""
from __future__ import annotations

from pathlib import Path

_PERCORSO_LESSICO = Path(__file__).resolve().parent.parent / "lingua" / "lessico.tsv"


def _analizza_tratti(campo: str) -> dict[str, str]:
    if campo == "-":
        return {}
    tratti: dict[str, str] = {}
    for coppia in campo.split(","):
        chiave, _, valore = coppia.partition("=")
        tratti[chiave] = valore
    return tratti


def _costruisci_valore_a_lemma() -> dict[int, str]:
    per_lemma: dict[str, tuple[str, dict[str, str]]] = {}
    with open(_PERCORSO_LESSICO, encoding="utf-8") as f:
        for riga in f:
            riga = riga.rstrip("\n")
            if not riga or riga.startswith("#"):
                continue
            campi = riga.split("\t")
            if len(campi) != 4:
                continue
            lemma, categoria, tratti_raw, _definizione = campi
            per_lemma[lemma] = (categoria, _analizza_tratti(tratti_raw))

    valore_a_lemma: dict[int, str] = {}
    for lemma in ("uno", "due"):
        _categoria, tratti = per_lemma[lemma]
        valore_a_lemma[int(tratti["valore"])] = lemma
    for lemma, (categoria, tratti) in per_lemma.items():
        if categoria == "NUM":
            valore_a_lemma[int(tratti["valore"])] = lemma
    return valore_a_lemma


VALORE_A_LEMMA: dict[int, str] = _costruisci_valore_a_lemma()


def lemma_numero(valore: int) -> str:
    """Lemma del lessico per il numerale cardinale `valore` (es. 17 -> "diciassette")."""
    if valore not in VALORE_A_LEMMA:
        raise ValueError(f"valore numerico fuori dal lessico: {valore}")
    return VALORE_A_LEMMA[valore]
