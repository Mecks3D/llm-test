"""Morfologia italiana della grammatica controllata (FASE1_PIANO.md §8).

Tutto guidato dal lessico: le funzioni leggono i tratti di `lessico.tsv` (mai
inventati qui) e sono pure rispetto al lessico caricato una volta all'avvio
del modulo. Copre SOLO ciò che gli stampi del micro-mondo producono
(FASE1.md): niente morfologia italiana generale.
"""
from __future__ import annotations

from .lessico import carica_lessico

_LESSICO = carica_lessico()

_VOCALI = "aeiouAEIOU"


def _genere(lemma: str) -> str:
    genere = _LESSICO[lemma].tratti.get("genere")
    if genere is None:
        raise ValueError(f"lemma senza tratto genere: {lemma!r}")
    return genere


def _inizia_speciale_maschile(lemma: str) -> bool:
    """z, gn, ps, x, y, o s+consonante: la classe che vuole 'lo'/'gli'."""
    p = lemma.lower()
    if p.startswith(("z", "gn", "ps", "x", "y")):
        return True
    return p.startswith("s") and len(p) > 1 and p[1] not in _VOCALI


def articolo_det_per_genere(genere: str, parola: str, plurale: bool = False) -> str:
    """Come articolo_det, ma il genere è dato esplicitamente e la scelta
    lo/il/l'/gli/i dipende dalla lettera iniziale di `parola` (serve quando
    tra articolo e nome si interpone un ordinale: "l'ottava mela")."""
    vocale = parola[0].lower() in _VOCALI
    speciale = _inizia_speciale_maschile(parola)
    if genere == "m":
        if plurale:
            return "gli" if (vocale or speciale) else "i"
        if speciale:
            return "lo"
        return "l'" if vocale else "il"
    if plurale:
        return "le"
    return "l'" if vocale else "la"


def articolo_det(lemma: str, plurale: bool = False) -> str:
    return articolo_det_per_genere(_genere(lemma), lemma, plurale)


def articolo_indet(lemma: str) -> str:
    genere = _genere(lemma)
    if genere == "m":
        return "uno" if _inizia_speciale_maschile(lemma) else "un"
    return "un'" if lemma[0].lower() in _VOCALI else "una"


_PREP_ARTICOLATA = {
    ("a", "il"): "al", ("a", "la"): "alla", ("a", "l'"): "all'",
    ("da", "il"): "dal", ("da", "la"): "dalla", ("da", "l'"): "dall'",
    ("di", "il"): "del", ("di", "la"): "della", ("di", "l'"): "dell'",
    ("in", "il"): "nel", ("in", "la"): "nella", ("in", "l'"): "nell'",
}


def prep_articolata(prep: str, lemma: str) -> str:
    articolo = articolo_det(lemma)
    chiave = (prep, articolo)
    if chiave not in _PREP_ARTICOLATA:
        raise ValueError(f"combinazione preposizione-articolo non coperta: {prep!r} + {articolo!r} ({lemma!r})")
    return _PREP_ARTICOLATA[chiave]


# Tutte le forme fuse preposizione+articolo coperte (usato dal formario,
# lingua/formario.py: sono combinatorie, non legate a un lemma specifico).
TUTTE_LE_PREPOSIZIONI_ARTICOLATE: tuple[str, ...] = tuple(_PREP_ARTICOLATA.values())


def unisci(primo: str, secondo: str) -> str:
    """Accosta due pezzi di frase, senza spazio se il primo elide (finisce
    in apostrofo): unisci("l'", "acqua") -> "l'acqua", unisci("la", "legna")
    -> "la legna"."""
    if primo.endswith("'"):
        return f"{primo}{secondo}"
    return f"{primo} {secondo}"


def partitivo(lemma: str) -> str:
    return unisci(prep_articolata("di", lemma), lemma)


def prep_lemma(prep: str, lemma: str) -> str:
    """Preposizione articolata + lemma, es. "nel cestino", "dalla scatola"."""
    return unisci(prep_articolata(prep, lemma), lemma)


def loc_in(luogo_id: str) -> str:
    return _LESSICO[luogo_id].tratti["loc_in"]


def loc_da(luogo_id: str) -> str:
    return _LESSICO[luogo_id].tratti["loc_da"]


_LUOGO_DA_LOC_IN = {v.tratti["loc_in"]: v.lemma for v in _LESSICO.per_categoria("NOME") if "loc_in" in v.tratti}
_LUOGO_DA_LOC_DA = {v.tratti["loc_da"]: v.lemma for v in _LESSICO.per_categoria("NOME") if "loc_da" in v.tratti}


def luogo_da_loc_in(testo: str) -> str:
    if testo not in _LUOGO_DA_LOC_IN:
        raise ValueError(f"complemento di luogo sconosciuto: {testo!r}")
    return _LUOGO_DA_LOC_IN[testo]


def luogo_da_loc_da(testo: str) -> str:
    if testo not in _LUOGO_DA_LOC_DA:
        raise ValueError(f"complemento di provenienza sconosciuto: {testo!r}")
    return _LUOGO_DA_LOC_DA[testo]


def _costruisci_numeri() -> tuple[dict[int, str], dict[str, int]]:
    valore_a_superficie: dict[int, str] = {}
    for lemma in ("uno", "due"):
        voce = _LESSICO[lemma]
        valore_a_superficie[int(voce.tratti["valore"])] = voce.tratti.get("superficie", lemma)
    for voce in _LESSICO.per_categoria("NUM"):
        valore_a_superficie[int(voce.tratti["valore"])] = voce.tratti.get("superficie", voce.lemma)
    superficie_a_valore = {superficie: valore for valore, superficie in valore_a_superficie.items()}
    return valore_a_superficie, superficie_a_valore


_VALORE_A_NUMERO, _NUMERO_A_VALORE = _costruisci_numeri()


def numero_in_lettere(n: int) -> str:
    if n not in _VALORE_A_NUMERO:
        raise ValueError(f"numero fuori dal dominio coperto dal lessico: {n}")
    return _VALORE_A_NUMERO[n]


def numero_da_lettere(s: str) -> int:
    if s not in _NUMERO_A_VALORE:
        raise ValueError(f"parola-numero sconosciuta: {s!r}")
    return _NUMERO_A_VALORE[s]


def _costruisci_ordinali() -> tuple[dict[int, str], dict[str, int]]:
    valore_a_lemma: dict[int, str] = {}
    lemma_a_valore: dict[str, int] = {}
    for voce in _LESSICO.per_categoria("ORD"):
        valore = int(voce.tratti["valore"])
        valore_a_lemma[valore] = voce.lemma
        lemma_a_valore[voce.lemma] = valore
    return valore_a_lemma, lemma_a_valore


_VALORE_A_ORDINALE, _ORDINALE_A_VALORE = _costruisci_ordinali()


def ordinale(n: int, genere: str) -> str:
    if n not in _VALORE_A_ORDINALE:
        raise ValueError(f"ordinale fuori dal dominio coperto dal lessico: {n}")
    lemma = _VALORE_A_ORDINALE[n]
    return lemma[:-1] + "a" if genere == "f" else lemma


def ordinale_inverso(s: str) -> int:
    if s in _ORDINALE_A_VALORE:
        return _ORDINALE_A_VALORE[s]
    if s.endswith("a"):
        maschile = s[:-1] + "o"
        if maschile in _ORDINALE_A_VALORE:
            return _ORDINALE_A_VALORE[maschile]
    raise ValueError(f"ordinale sconosciuto: {s!r}")


def ora_in_lettere(t: int) -> str:
    if t == 1:
        return "all'una"
    if not (2 <= t <= 24):
        raise ValueError(f"ora fuori dal dominio 1-24: {t}")
    return f"alle {numero_in_lettere(t)}"


def ora_da_lettere(s: str) -> int:
    if s == "all'una":
        return 1
    if s.startswith("alle "):
        return numero_da_lettere(s[len("alle "):])
    raise ValueError(f"ora sconosciuta: {s!r}")


def plurale(lemma: str) -> str:
    voce = _LESSICO[lemma]
    if "plurale" not in voce.tratti:
        raise ValueError(f"lemma senza tratto plurale: {lemma!r}")
    return voce.tratti["plurale"]


def forma_verbale(lemma: str, forma: str) -> str:
    voce = _LESSICO[lemma]
    if forma not in voce.tratti:
        raise ValueError(f"verbo {lemma!r} senza forma {forma!r}")
    return voce.tratti[forma]


def aggettivo(lemma: str, genere: str) -> str:
    if genere == "f":
        return _LESSICO[lemma].tratti["femminile"]
    return lemma
