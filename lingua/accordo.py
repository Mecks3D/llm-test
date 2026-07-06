"""Controllo automatico di accordo genere/numero/elisione (FASE1_PIANO.md §11).

Indipendente dagli stampi: non chiama `morfologia.articolo_det` — ricalcola
da solo, dai tratti grezzi del lessico, quale articolo ci si aspetta prima
di un nome (o di un ordinale seguito da un nome), così un eventuale bug
negli stampi non si nasconderebbe dietro lo stesso codice nel checker.
"""
from __future__ import annotations

from .lessico import carica_lessico

_LESSICO = carica_lessico()
_VOCALI = "aeiouAEIOU"

_ARTICOLI_DETERMINATIVI = frozenset({"il", "lo", "la", "l'", "i", "gli", "le"})

# nome (singolare o plurale) -> genere, ricavato direttamente dai tratti
# NOME/PROPRIO del lessico (non dalle tabelle assemblate da morfologia.py).
_GENERE_SINGOLARE: dict[str, str] = {}
_GENERE_PLURALE: dict[str, str] = {}
for _voce in _LESSICO.voci():
    genere = _voce.tratti.get("genere")
    if genere is None:
        continue
    _GENERE_SINGOLARE[_voce.lemma] = genere
    plurale = _voce.tratti.get("plurale")
    if plurale is not None:
        _GENERE_PLURALE[plurale] = genere

_ORDINALI_M = frozenset(v.lemma for v in _LESSICO.per_categoria("ORD"))
_ORDINALI_F = frozenset(f"{v.lemma[:-1]}a" for v in _LESSICO.per_categoria("ORD"))


def _e_vocale(parola: str) -> bool:
    return bool(parola) and parola[0] in _VOCALI


def _e_esse_impura_o_speciale(parola: str) -> bool:
    p = parola.lower()
    if p.startswith(("z", "gn", "ps", "x", "y")):
        return True
    return p.startswith("s") and len(p) > 1 and p[1] not in _VOCALI


def _articoli_attesi(genere: str, parola_seguente: str, plurale: bool) -> frozenset[str]:
    vocale = _e_vocale(parola_seguente)
    speciale = _e_esse_impura_o_speciale(parola_seguente)
    if genere == "m":
        if plurale:
            return frozenset({"gli"}) if (vocale or speciale) else frozenset({"i"})
        if speciale:
            return frozenset({"lo"})
        return frozenset({"l'"}) if vocale else frozenset({"il"})
    if plurale:
        return frozenset({"le"})
    return frozenset({"l'"}) if vocale else frozenset({"la"})


_PREFISSI_ELISIONE = ("l'", "un'", "dell'", "nell'", "dall'", "all'")


def _tokenizza(frase: str) -> list[str]:
    token: list[str] = []
    for grezzo in frase.split(" "):
        parola = grezzo.strip(".?!,")
        if not parola:
            continue
        for prefisso in _PREFISSI_ELISIONE:
            if parola.lower().startswith(prefisso) and len(parola) > len(prefisso):
                token.append(parola[:len(prefisso)])
                token.append(parola[len(prefisso):])
                break
        else:
            token.append(parola)
    return token


def controlla_accordo(frase: str) -> list[str]:
    """Ritorna gli errori di accordo trovati in `frase` (lista vuota = ok).

    Cerca ogni occorrenza di un articolo determinativo seguito, dopo un
    ordinale opzionale, da un nome noto al lessico, e verifica che
    l'articolo osservato sia quello giusto per genere/numero/elisione."""
    errori: list[str] = []
    token = [t.lower() if t.lower() in _ARTICOLI_DETERMINATIVI else t for t in _tokenizza(frase)]
    n = len(token)
    for i, art in enumerate(token):
        if art.lower() not in _ARTICOLI_DETERMINATIVI:
            continue
        j = i + 1
        if j >= n:
            continue
        ordinale_genere = None
        if token[j] in _ORDINALI_M:
            ordinale_genere = "m"
        elif token[j] in _ORDINALI_F:
            ordinale_genere = "f"
        nome_idx = j + 1 if ordinale_genere is not None else j
        if nome_idx >= n:
            continue
        nome = token[nome_idx]
        plurale = nome in _GENERE_PLURALE
        genere = _GENERE_PLURALE.get(nome) if plurale else _GENERE_SINGOLARE.get(nome)
        if genere is None:
            continue
        if ordinale_genere is not None and ordinale_genere != genere:
            errori.append(f"'{frase}': ordinale {token[j]!r} non concorda in genere con {nome!r}")
            continue
        parola_seguente = token[j] if ordinale_genere is not None else nome
        attesi = _articoli_attesi(genere, parola_seguente, plurale)
        if art.lower() not in attesi:
            errori.append(
                f"'{frase}': articolo {art!r} prima di {parola_seguente!r} ({nome!r}, genere={genere}, "
                f"plurale={plurale}); atteso uno tra {sorted(attesi)}"
            )
    return errori
