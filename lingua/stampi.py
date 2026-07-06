"""Gli stampi dichiarativi della grammatica controllata (FASE1_PIANO.md §5).

Ogni costrutto vive qui una volta sola: `verbalizza.py` chiama le funzioni
`rendi_*`, `analizza.py` chiama le funzioni `riconosci_*`/`*_inversa`. Per gli
eventi, l'ordine di riconoscimento è quello della tabella §5 (FASE1_PIANO.md).
"""
from __future__ import annotations

from mondo import dati_mondo as dm
from mondo.tipi import Evento

from . import morfologia as mf
from .contesto import StatoDiscorso, estrai_lemma_istanza
from .lessico import carica_lessico

_LESSICO = carica_lessico()

_PERSONE_IDS: frozenset[str] = frozenset(p.id for p in dm.PERSONE)
_NOME_PERSONA_A_ID: dict[str, str] = {p.id.capitalize(): p.id for p in dm.PERSONE}
_LEMMI_RISORSA: frozenset[str] = frozenset(info["lemma_unita"] for info in dm.RISORSE.values())
_LEMMI_OGGETTI_UNICI: frozenset[str] = frozenset(t.lemma for t in dm.OGGETTI_UNICI)
_CONTENITORI: tuple[str, ...] = ("cestino", "scatola", "secchio", "camino")
_LUOGHI_LOC_IN: dict[str, str] = {l.id: mf.loc_in(l.id) for l in dm.LUOGHI}
_LUOGHI_LOC_DA: dict[str, str] = {l.id: mf.loc_da(l.id) for l in dm.LUOGHI}


def _genere(lemma_o_persona: str) -> str:
    return _LESSICO[lemma_o_persona].tratti["genere"]


# ---------------------------------------------------------------------------
# Sintagma nominale (FASE1_PIANO.md §4.2)
# ---------------------------------------------------------------------------

def sn(entita_id: str, contesto: StatoDiscorso) -> str:
    if entita_id in ("nessuno", "qualcuno"):
        return entita_id
    if entita_id in _PERSONE_IDS:
        return entita_id.capitalize()
    istanza = estrai_lemma_istanza(entita_id)
    if istanza is None:
        return mf.unisci(mf.articolo_det(entita_id), entita_id)
    lemma, indice = istanza
    max_i = contesto.max_indice.get(lemma, 0)
    if max_i <= 0:
        raise ValueError(f"istanza non ancora registrata nel contesto di discorso: {entita_id!r}")
    if max_i == 1:
        return mf.unisci(mf.articolo_det(lemma), lemma)
    genere = _genere(lemma)
    parola_ordinale = mf.ordinale(indice, genere)
    articolo = mf.articolo_det_per_genere(genere, parola_ordinale)
    return mf.unisci(articolo, f"{parola_ordinale} {lemma}")


_ARTICOLI_SPAZIO: tuple[str, ...] = ("gli ", "il ", "lo ", "la ", "le ", "i ")


def _stacca_articolo(testo: str) -> tuple[str, str]:
    if testo.startswith("l'"):
        return "l'", testo[2:]
    for art in _ARTICOLI_SPAZIO:
        if testo.startswith(art):
            return art[:-1], testo[len(art):]
    raise ValueError(f"sintagma nominale senza articolo riconoscibile: {testo!r}")


def sn_inversa(testo: str, contesto: StatoDiscorso) -> str:
    if testo in ("nessuno", "qualcuno"):
        return testo
    if testo and testo[0].isupper():
        id_persona = _NOME_PERSONA_A_ID.get(testo)
        if id_persona is None:
            raise ValueError(f"nome proprio sconosciuto: {testo!r}")
        return id_persona
    _articolo, resto = _stacca_articolo(testo)
    if " " in resto:
        parola_ordinale, lemma = resto.split(" ", 1)
    else:
        parola_ordinale, lemma = None, resto
    if lemma in _LEMMI_OGGETTI_UNICI:
        if parola_ordinale is not None:
            raise ValueError(f"sintagma nominale non valido per oggetto unico: {testo!r}")
        return lemma
    if lemma in _LEMMI_RISORSA:
        max_i = contesto.max_indice.get(lemma, 0)
        if parola_ordinale is None:
            if max_i != 1:
                raise ValueError(
                    f"riferimento definito semplice non valido per {lemma!r} (max_indice={max_i}): {testo!r}"
                )
            return f"{lemma}_1"
        indice = mf.ordinale_inverso(parola_ordinale)
        if indice > max_i:
            raise ValueError(f"ordinale oltre il massimo indice noto per {lemma!r}: {testo!r}")
        return f"{lemma}_{indice}"
    raise ValueError(f"sintagma nominale sconosciuto: {testo!r}")


# ---------------------------------------------------------------------------
# Complemento di luogo (FASE1_PIANO.md §3, "Regola di esplicitazione del luogo")
# ---------------------------------------------------------------------------

def clausola_luogo(evento: Evento, contesto: StatoDiscorso) -> str:
    if evento.luogo is None:
        return ""
    if evento.azione == "prendere" and evento.argomento in ("melo", "pozzo", "bosco_legna"):
        return ""
    if evento.azione == "mettere_dentro" and evento.argomento == "camino":
        return ""
    if evento.azione == "bruciare":
        return ""
    if contesto.posizione_persone.get(evento.agente) == evento.luogo:
        return ""
    return f" {mf.loc_in(evento.luogo)}"


def _stacca_clausola_luogo(testo: str) -> tuple[str, str | None]:
    for luogo_id, loc_in_testo in _LUOGHI_LOC_IN.items():
        suffisso = f" {loc_in_testo}"
        if testo.endswith(suffisso):
            return testo[: -len(suffisso)], luogo_id
    return testo, None


def _stacca_prep_lemma(prep: str, testo: str) -> tuple[str, str | None]:
    for contenitore in _CONTENITORI:
        suffisso = f" {mf.prep_lemma(prep, contenitore)}"
        if testo.endswith(suffisso):
            return testo[: -len(suffisso)], contenitore
    return testo, None


def _deduci_luogo(luogo_esplicito: str | None, agente: str, contesto: StatoDiscorso) -> str | None:
    if luogo_esplicito is not None:
        return luogo_esplicito
    return contesto.posizione_persone.get(agente)


# ---------------------------------------------------------------------------
# Prefisso di tempo (FASE1_PIANO.md §4.1)
# ---------------------------------------------------------------------------

def prefisso_tempo(evento: Evento, contesto: StatoDiscorso) -> str:
    if contesto.tick_corrente is not None and evento.t == contesto.tick_corrente:
        return "Intanto "
    ora = mf.ora_in_lettere(evento.t)
    return f"{ora[0].upper()}{ora[1:]} "


def stacca_prefisso_tempo(frase: str, contesto: StatoDiscorso) -> tuple[int, str]:
    if frase.startswith("Intanto "):
        if contesto.tick_corrente is None:
            raise ValueError("'Intanto' non può aprire la prima frase-evento della storia")
        return contesto.tick_corrente, frase[len("Intanto "):]
    if frase.startswith("All'una "):
        return 1, frase[len("All'una "):]
    if frase.startswith("Alle "):
        resto = frase[len("Alle "):]
        parola_numero, _, resto2 = resto.partition(" ")
        return mf.numero_da_lettere(parola_numero), resto2
    raise ValueError(f"prefisso di tempo non riconosciuto: {frase!r}")


# ---------------------------------------------------------------------------
# Stampi degli eventi: rendering (FASE1_PIANO.md §5)
# ---------------------------------------------------------------------------

def rendi_evento_corpo(evento: Evento, contesto: StatoDiscorso) -> str:
    azione = evento.azione
    ag = sn(evento.agente, contesto)

    if azione == "andare":
        dest = mf.loc_in(evento.luogo)
        if contesto.posizione_persone.get(evento.agente) is None:
            origine = mf.loc_da(evento.luogo_origine)
            return f"{ag} {mf.forma_verbale('andare', 'pres3s')} {origine} {dest}"
        return f"{ag} {mf.forma_verbale('andare', 'pres3s')} {dest}"

    if azione == "prendere":
        if evento.argomento == "melo":
            return f"{ag} {mf.forma_verbale('raccogliere', 'pres3s')} una mela dal melo"
        if evento.argomento == "pozzo":
            return f"{ag} {mf.forma_verbale('prendere', 'pres3s')} {mf.partitivo('acqua')} dal pozzo"
        if evento.argomento == "bosco_legna":
            return f"{ag} {mf.forma_verbale('raccogliere', 'pres3s')} {mf.partitivo('legna')} nel bosco"
        return (f"{ag} {mf.forma_verbale('prendere', 'pres3s')} {sn(evento.oggetto, contesto)}"
                f"{clausola_luogo(evento, contesto)}")

    if azione == "tirare_fuori":
        return (f"{ag} {mf.forma_verbale('tirare_fuori', 'superficie_pres3s')} {sn(evento.oggetto, contesto)} "
                f"{mf.prep_lemma('da', evento.argomento)}{clausola_luogo(evento, contesto)}")

    if azione == "posare":
        return f"{ag} {mf.forma_verbale('posare', 'pres3s')} {sn(evento.oggetto, contesto)}{clausola_luogo(evento, contesto)}"

    if azione == "mettere_dentro":
        return (f"{ag} {mf.forma_verbale('mettere_dentro', 'superficie_pres3s')} {sn(evento.oggetto, contesto)} "
                f"{mf.prep_lemma('in', evento.argomento)}{clausola_luogo(evento, contesto)}")

    if azione == "dare":
        return (f"{ag} {mf.forma_verbale('dare', 'pres3s')} {sn(evento.oggetto, contesto)} a "
                f"{sn(evento.destinatario, contesto)}{clausola_luogo(evento, contesto)}")

    if azione == "mangiare":
        return f"{ag} {mf.forma_verbale('mangiare', 'pres3s')} {sn(evento.oggetto, contesto)}{clausola_luogo(evento, contesto)}"

    if azione == "aprire":
        return f"{ag} {mf.forma_verbale('aprire', 'pres3s')} {sn(evento.oggetto, contesto)}{clausola_luogo(evento, contesto)}"

    if azione == "chiudere":
        return f"{ag} {mf.forma_verbale('chiudere', 'pres3s')} {sn(evento.oggetto, contesto)}{clausola_luogo(evento, contesto)}"

    if azione == "guardare":
        return f"{ag} {mf.forma_verbale('guardare', 'pres3s')} {sn(evento.oggetto, contesto)}{clausola_luogo(evento, contesto)}"

    if azione == "dire":
        return (f"{ag} {mf.forma_verbale('dire', 'pres3s')} qualcosa a {sn(evento.destinatario, contesto)}"
                f"{clausola_luogo(evento, contesto)}")

    if azione == "dormire":
        corpo = f"{ag} {mf.forma_verbale('addormentarsi', 'pres3s')}{clausola_luogo(evento, contesto)}"
        if evento.argomento == "stanchezza":
            genere = _genere(evento.agente)
            return f"{corpo} perché è {mf.aggettivo('stanco', genere)}"
        return corpo

    if azione == "svegliarsi":
        return f"{ag} {mf.forma_verbale('svegliarsi', 'pres3s')}{clausola_luogo(evento, contesto)}"

    if azione == "giocare":
        if evento.oggetto is not None:
            return (f"{ag} {mf.forma_verbale('giocare', 'pres3s')} con {sn(evento.oggetto, contesto)}"
                    f"{clausola_luogo(evento, contesto)}")
        return f"{ag} {mf.forma_verbale('giocare', 'pres3s')}{clausola_luogo(evento, contesto)}"

    if azione == "cercare":
        return f"{ag} {mf.forma_verbale('cercare', 'pres3s')} {sn(evento.oggetto, contesto)}{clausola_luogo(evento, contesto)}"

    if azione == "bruciare":
        return f"{ag} {mf.forma_verbale('bruciare', 'pres3s')} {sn(evento.oggetto, contesto)}"

    raise ValueError(f"nessuno stampo per l'azione {azione!r}")


# ---------------------------------------------------------------------------
# Stampi degli eventi: riconoscimento (ordine = tabella §5)
# ---------------------------------------------------------------------------

def _stacca_nome_persona(corpo: str) -> tuple[str | None, str]:
    token, spazio, resto = corpo.partition(" ")
    if spazio and token in _NOME_PERSONA_A_ID:
        return _NOME_PERSONA_A_ID[token], resto
    return None, corpo


def _prova_andare(corpo: str, contesto: StatoDiscorso, t: int) -> Evento | None:
    agente, resto = _stacca_nome_persona(corpo)
    if agente is None:
        return None
    pref = f"{mf.forma_verbale('andare', 'pres3s')} "
    if not resto.startswith(pref):
        return None
    resto = resto[len(pref):]
    for luogo_o, loc_da_testo in _LUOGHI_LOC_DA.items():
        pref_o = f"{loc_da_testo} "
        if resto.startswith(pref_o):
            resto_dest = resto[len(pref_o):]
            for luogo_d, loc_in_testo in _LUOGHI_LOC_IN.items():
                if resto_dest == loc_in_testo:
                    return Evento(t=t, azione="andare", agente=agente, luogo=luogo_d, luogo_origine=luogo_o)
            return None
    for luogo_d, loc_in_testo in _LUOGHI_LOC_IN.items():
        if resto == loc_in_testo:
            luogo_o = contesto.posizione_persone.get(agente)
            if luogo_o is None:
                return None
            return Evento(t=t, azione="andare", agente=agente, luogo=luogo_d, luogo_origine=luogo_o)
    return None


def _prova_prendere(corpo: str, contesto: StatoDiscorso, t: int) -> Evento | None:
    agente, resto = _stacca_nome_persona(corpo)
    if agente is None:
        return None
    v_racc = mf.forma_verbale("raccogliere", "pres3s")
    v_prend = mf.forma_verbale("prendere", "pres3s")
    if resto == f"{v_racc} una mela dal melo":
        indice = contesto.max_indice.get("mela", 0) + 1
        return Evento(t=t, azione="prendere", agente=agente, oggetto=f"mela_{indice}",
                      argomento="melo", luogo=dm.RISORSE["melo"]["luogo"])
    if resto == f"{v_prend} {mf.partitivo('acqua')} dal pozzo":
        indice = contesto.max_indice.get("acqua", 0) + 1
        return Evento(t=t, azione="prendere", agente=agente, oggetto=f"acqua_{indice}",
                      argomento="pozzo", luogo=dm.RISORSE["pozzo"]["luogo"])
    if resto == f"{v_racc} {mf.partitivo('legna')} nel bosco":
        indice = contesto.max_indice.get("legna", 0) + 1
        return Evento(t=t, azione="prendere", agente=agente, oggetto=f"legna_{indice}",
                      argomento="bosco_legna", luogo=dm.RISORSE["bosco_legna"]["luogo"])
    pref = f"{v_prend} "
    if not resto.startswith(pref):
        return None
    corpo_o, luogo_esplicito = _stacca_clausola_luogo(resto[len(pref):])
    oggetto = sn_inversa(corpo_o, contesto)
    luogo = _deduci_luogo(luogo_esplicito, agente, contesto)
    if luogo is None:
        return None
    return Evento(t=t, azione="prendere", agente=agente, oggetto=oggetto, luogo=luogo)


def _prova_tirare_fuori(corpo: str, contesto: StatoDiscorso, t: int) -> Evento | None:
    agente, resto = _stacca_nome_persona(corpo)
    if agente is None:
        return None
    pref = f"{mf.forma_verbale('tirare_fuori', 'superficie_pres3s')} "
    if not resto.startswith(pref):
        return None
    corpo_resto, luogo_esplicito = _stacca_clausola_luogo(resto[len(pref):])
    corpo_o, contenitore = _stacca_prep_lemma("da", corpo_resto)
    if contenitore is None:
        return None
    oggetto = sn_inversa(corpo_o, contesto)
    luogo = _deduci_luogo(luogo_esplicito, agente, contesto)
    if luogo is None:
        return None
    return Evento(t=t, azione="tirare_fuori", agente=agente, oggetto=oggetto, argomento=contenitore, luogo=luogo)


def _prova_semplice_oggetto(corpo: str, contesto: StatoDiscorso, t: int, azione: str, lemma_verbo: str) -> Evento | None:
    agente, resto = _stacca_nome_persona(corpo)
    if agente is None:
        return None
    pref = f"{mf.forma_verbale(lemma_verbo, 'pres3s')} "
    if not resto.startswith(pref):
        return None
    corpo_o, luogo_esplicito = _stacca_clausola_luogo(resto[len(pref):])
    oggetto = sn_inversa(corpo_o, contesto)
    luogo = _deduci_luogo(luogo_esplicito, agente, contesto)
    if luogo is None:
        return None
    return Evento(t=t, azione=azione, agente=agente, oggetto=oggetto, luogo=luogo)


def _prova_posare(corpo, contesto, t):
    return _prova_semplice_oggetto(corpo, contesto, t, "posare", "posare")


def _prova_mettere_dentro(corpo: str, contesto: StatoDiscorso, t: int) -> Evento | None:
    agente, resto = _stacca_nome_persona(corpo)
    if agente is None:
        return None
    pref = f"{mf.forma_verbale('mettere_dentro', 'superficie_pres3s')} "
    if not resto.startswith(pref):
        return None
    corpo_resto, luogo_esplicito = _stacca_clausola_luogo(resto[len(pref):])
    corpo_o, contenitore = _stacca_prep_lemma("in", corpo_resto)
    if contenitore is None:
        return None
    oggetto = sn_inversa(corpo_o, contesto)
    if contenitore == "camino":
        luogo = "salotto"
    else:
        luogo = _deduci_luogo(luogo_esplicito, agente, contesto)
    if luogo is None:
        return None
    return Evento(t=t, azione="mettere_dentro", agente=agente, oggetto=oggetto, argomento=contenitore, luogo=luogo)


def _prova_dare(corpo: str, contesto: StatoDiscorso, t: int) -> Evento | None:
    agente, resto = _stacca_nome_persona(corpo)
    if agente is None:
        return None
    pref = f"{mf.forma_verbale('dare', 'pres3s')} "
    if not resto.startswith(pref):
        return None
    corpo_resto, luogo_esplicito = _stacca_clausola_luogo(resto[len(pref):])
    idx = corpo_resto.rfind(" a ")
    if idx == -1:
        return None
    oggetto = sn_inversa(corpo_resto[:idx], contesto)
    destinatario = sn_inversa(corpo_resto[idx + len(" a "):], contesto)
    luogo = _deduci_luogo(luogo_esplicito, agente, contesto)
    if luogo is None:
        return None
    return Evento(t=t, azione="dare", agente=agente, oggetto=oggetto, destinatario=destinatario, luogo=luogo)


def _prova_mangiare(corpo, contesto, t):
    return _prova_semplice_oggetto(corpo, contesto, t, "mangiare", "mangiare")


def _prova_aprire(corpo, contesto, t):
    return _prova_semplice_oggetto(corpo, contesto, t, "aprire", "aprire")


def _prova_chiudere(corpo, contesto, t):
    return _prova_semplice_oggetto(corpo, contesto, t, "chiudere", "chiudere")


def _prova_guardare(corpo, contesto, t):
    return _prova_semplice_oggetto(corpo, contesto, t, "guardare", "guardare")


def _prova_dire(corpo: str, contesto: StatoDiscorso, t: int) -> Evento | None:
    agente, resto = _stacca_nome_persona(corpo)
    if agente is None:
        return None
    pref = f"{mf.forma_verbale('dire', 'pres3s')} qualcosa a "
    if not resto.startswith(pref):
        return None
    corpo_d, luogo_esplicito = _stacca_clausola_luogo(resto[len(pref):])
    destinatario = sn_inversa(corpo_d, contesto)
    luogo = _deduci_luogo(luogo_esplicito, agente, contesto)
    if luogo is None:
        return None
    return Evento(t=t, azione="dire", agente=agente, destinatario=destinatario, luogo=luogo)


def _prova_dormire(corpo: str, contesto: StatoDiscorso, t: int) -> Evento | None:
    agente, resto = _stacca_nome_persona(corpo)
    if agente is None:
        return None
    verbo = mf.forma_verbale("addormentarsi", "pres3s")
    if not resto.startswith(verbo):
        return None
    resto = resto[len(verbo):]
    argomento = None
    for genere_prova in ("m", "f"):
        suffisso = f" perché è {mf.aggettivo('stanco', genere_prova)}"
        if resto.endswith(suffisso):
            resto = resto[: -len(suffisso)]
            argomento = "stanchezza"
            break
    corpo_resto, luogo_esplicito = _stacca_clausola_luogo(resto)
    if corpo_resto != "":
        return None
    luogo = _deduci_luogo(luogo_esplicito, agente, contesto)
    if luogo is None:
        return None
    return Evento(t=t, azione="dormire", agente=agente, argomento=argomento, luogo=luogo)


def _prova_svegliarsi(corpo: str, contesto: StatoDiscorso, t: int) -> Evento | None:
    agente, resto = _stacca_nome_persona(corpo)
    if agente is None:
        return None
    verbo = mf.forma_verbale("svegliarsi", "pres3s")
    if not resto.startswith(verbo):
        return None
    corpo_resto, luogo_esplicito = _stacca_clausola_luogo(resto[len(verbo):])
    if corpo_resto != "":
        return None
    luogo = _deduci_luogo(luogo_esplicito, agente, contesto)
    if luogo is None:
        return None
    return Evento(t=t, azione="svegliarsi", agente=agente, luogo=luogo)


def _prova_giocare(corpo: str, contesto: StatoDiscorso, t: int) -> Evento | None:
    agente, resto = _stacca_nome_persona(corpo)
    if agente is None:
        return None
    verbo = mf.forma_verbale("giocare", "pres3s")
    if not resto.startswith(verbo):
        return None
    resto = resto[len(verbo):]
    if resto.startswith(" con "):
        corpo_o, luogo_esplicito = _stacca_clausola_luogo(resto[len(" con "):])
        oggetto = sn_inversa(corpo_o, contesto)
        luogo = _deduci_luogo(luogo_esplicito, agente, contesto)
        if luogo is None:
            return None
        return Evento(t=t, azione="giocare", agente=agente, oggetto=oggetto, luogo=luogo)
    corpo_resto, luogo_esplicito = _stacca_clausola_luogo(resto)
    if corpo_resto != "":
        return None
    luogo = _deduci_luogo(luogo_esplicito, agente, contesto)
    if luogo is None:
        return None
    return Evento(t=t, azione="giocare", agente=agente, oggetto=None, luogo=luogo)


def _prova_cercare(corpo, contesto, t):
    return _prova_semplice_oggetto(corpo, contesto, t, "cercare", "cercare")


def _prova_bruciare(corpo: str, contesto: StatoDiscorso, t: int) -> Evento | None:
    pref = f"il camino {mf.forma_verbale('bruciare', 'pres3s')} "
    if not corpo.startswith(pref):
        return None
    oggetto = sn_inversa(corpo[len(pref):], contesto)
    return Evento(t=t, azione="bruciare", agente="camino", oggetto=oggetto, luogo="salotto")


# Ordine = tabella FASE1_PIANO.md §5.
_PROVE_EVENTO = (
    _prova_andare,
    _prova_prendere,
    _prova_tirare_fuori,
    _prova_posare,
    _prova_mettere_dentro,
    _prova_dare,
    _prova_mangiare,
    _prova_aprire,
    _prova_chiudere,
    _prova_guardare,
    _prova_dire,
    _prova_dormire,
    _prova_svegliarsi,
    _prova_giocare,
    _prova_cercare,
    _prova_bruciare,
)


def riconosci_evento_corpo(corpo: str, contesto: StatoDiscorso, t: int) -> Evento:
    for prova in _PROVE_EVENTO:
        evento = prova(corpo, contesto, t)
        if evento is not None:
            return evento
    raise ValueError(f"nessuno stampo riconosce la frase-evento: {corpo!r}")
