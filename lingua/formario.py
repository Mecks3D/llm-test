"""Il formario: l'insieme di tutte le superfici lecite secondo il lessico
(FASE1_PIANO.md §11) — usato per verificare che nessuna frase generata
contenga lemmi fuori dal lessico (criterio di accettazione, FASE1.md).
"""
from __future__ import annotations

from . import morfologia as mf
from .lessico import carica_lessico

_LESSICO = carica_lessico()

# Tratti puramente classificatori: non portano testo di superficie.
_CHIAVI_NON_SUPERFICIE: frozenset[str] = frozenset({"genere", "valore", "massa"})

_PREFISSI_ELISIONE: tuple[str, ...] = ("l'", "un'", "dell'", "nell'", "dall'", "all'")


def _spezza(frammento: str) -> list[str]:
    """Spazio, poi punteggiatura, poi elisione — la stessa scomposizione
    usata per tokenizzare le frasi generate (vedi `parole_fuori_formario`),
    così le due letture restano coerenti per costruzione."""
    pezzi: list[str] = []
    for grezzo in frammento.split(" "):
        parola = grezzo.strip(".?!,")
        if not parola:
            continue
        for prefisso in _PREFISSI_ELISIONE:
            if parola.lower().startswith(prefisso) and len(parola) > len(prefisso):
                pezzi.append(parola[:len(prefisso)])
                pezzi.append(parola[len(prefisso):])
                break
        else:
            pezzi.append(parola)
    return pezzi


def _costruisci_formario() -> frozenset[str]:
    formario: set[str] = set(_PREFISSI_ELISIONE)
    formario.update(mf.TUTTE_LE_PREPOSIZIONI_ARTICOLATE)
    for voce in _LESSICO.voci():
        if "_" not in voce.lemma:  # gli id strutturali (mettere_dentro...) non sono parole
            for pezzo in _spezza(voce.lemma):
                formario.add(pezzo.lower())
        for chiave, valore in voce.tratti.items():
            if chiave in _CHIAVI_NON_SUPERFICIE:
                continue
            for pezzo in _spezza(valore):
                formario.add(pezzo.lower())
        # Le forme femminili di ordinali e del participio di "raccogliere"
        # si calcolano a runtime (morfologia.ordinale, stampi._rendi_raccolta_risposta)
        # e non vivono come stringhe letterali nel lessico: la regola -o->-a
        # è la stessa in entrambi i punti, replicata qui per il formario.
        if voce.categoria == "ORD":
            formario.add(f"{voce.lemma[:-1]}a")
    part_raccogliere = _LESSICO["raccogliere"].tratti["part"]
    formario.add(f"{part_raccogliere[:-1]}a")
    formario.add(f"{part_raccogliere[:-1]}e")
    return frozenset(formario)


FORMARIO: frozenset[str] = _costruisci_formario()

NOMI_PROPRI: frozenset[str] = frozenset(v.lemma.capitalize() for v in _LESSICO.per_categoria("PROPRIO"))


def parole_fuori_formario(frase: str) -> list[str]:
    """Ritorna i token di `frase` che non appartengono al formario (lista
    vuota se la frase è pulita). Confronto case-insensitive, tranne per i
    nomi propri (FASE1_PIANO.md §11)."""
    fuori = []
    for token in _spezza(frase):
        if token in NOMI_PROPRI:
            continue
        if token.lower() in FORMARIO:
            continue
        fuori.append(token)
    return fuori
