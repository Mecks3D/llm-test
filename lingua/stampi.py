"""Gli stampi dichiarativi della grammatica controllata (FASE1_PIANO.md §5).

Ogni costrutto vive qui una volta sola: `verbalizza.py` chiama le funzioni
`rendi_*`, `analizza.py` chiama le funzioni `riconosci_*`/`*_inversa`. Per gli
eventi, l'ordine di riconoscimento è quello della tabella §5 (FASE1_PIANO.md).
"""
from __future__ import annotations

from mondo import dati_mondo as dm
from mondo.grafo import NON_LO_SO, Grafo, grafo_fatto
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


def _capitalizza(s: str) -> str:
    return f"{s[0].upper()}{s[1:]}" if s else s


def _decapitalizza(s: str) -> str:
    return f"{s[0].lower()}{s[1:]}" if s else s


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
    # Le risposte capitalizzano la prima parola della frase anche quando è
    # un sintagma comune ("La prima mela è..."): un nome proprio noto vince,
    # altrimenti si tratta di maiuscola di inizio-frase, non di persona.
    if testo not in _NOME_PERSONA_A_ID and testo and testo[0].isupper():
        testo = _decapitalizza(testo)
    if testo in ("nessuno", "qualcuno"):
        return testo
    if testo in _NOME_PERSONA_A_ID:
        return _NOME_PERSONA_A_ID[testo]
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


# ---------------------------------------------------------------------------
# Domande e risposte (FASE1_PIANO.md §6)
# ---------------------------------------------------------------------------

def _rendi_nome_massa(lemma: str) -> str:
    """Riferimento generico (non un'istanza) a un nome massa: "l'acqua",
    "la legna" — usato solo dallo stampo "raccolta"."""
    return mf.unisci(mf.articolo_det(lemma), lemma)


def _analizza_nome_massa(testo: str) -> str:
    _articolo, lemma = _stacca_articolo(testo)
    if lemma not in _LEMMI_RISORSA:
        raise ValueError(f"nome massa sconosciuto: {testo!r}")
    return lemma


def _testo_luogo_o_contenitore(entita_id: str) -> str:
    if entita_id in _LUOGHI_LOC_IN:
        return mf.loc_in(entita_id)
    if entita_id in _CONTENITORI:
        return mf.prep_lemma("in", entita_id)
    raise ValueError(f"né luogo né contenitore: {entita_id!r}")


def _luogo_o_contenitore_da_testo(testo: str) -> str:
    for luogo_id, loc_in_testo in _LUOGHI_LOC_IN.items():
        if testo == loc_in_testo:
            return luogo_id
    for contenitore in _CONTENITORI:
        if testo == mf.prep_lemma("in", contenitore):
            return contenitore
    raise ValueError(f"luogo/contenitore sconosciuto: {testo!r}")


def _costruisci_mappa_parentela_inversa() -> dict[str, str]:
    mappa: dict[str, str] = {}
    for voce in _LESSICO.per_categoria("REL"):
        superfici = (
            [voce.tratti["superficie"]] if "superficie" in voce.tratti
            else [voce.tratti["superficie_m"], voce.tratti["superficie_f"]]
        )
        for superficie in superfici:
            _articolo, sostantivo = superficie.split(" ", 1)
            mappa[sostantivo] = voce.lemma
    return mappa


_PARENTELA_SOSTANTIVO_A_RELAZIONE: dict[str, str] = _costruisci_mappa_parentela_inversa()


def _sn_parentela(relazione: str, persona_a: str) -> str:
    voce = _LESSICO[relazione]
    if "superficie" in voce.tratti:
        return voce.tratti["superficie"]
    return voce.tratti[f"superficie_{_genere(persona_a)}"]


def _ha_dato() -> str:
    return f"{mf.forma_verbale('avere', 'pres3s')} {mf.forma_verbale('dare', 'part')}"


# --- rendering -------------------------------------------------------------

def rendi_domanda(grafo: Grafo, contesto: StatoDiscorso) -> str:
    radice = grafo.nodi[0].lemma
    rel = {a.relazione: grafo.nodi[a.dipendente].lemma for a in grafo.archi}
    quesito = rel.get("quesito")

    if radice == "trovarsi" and quesito == "dove" and "nsubj" in rel:
        return f"Dove si trova {sn(rel['nsubj'], contesto)}?"
    if radice == "trovarsi" and quesito == "dove":
        agente_testo = sn(rel["nmod:agente"], contesto)
        return (f"Dove si trova {sn(rel['nmod:oggetto'], contesto)} che {agente_testo} "
                f"{_ha_dato()} a {sn(rel['nmod:destinatario'], contesto)}?")
    if radice == "avere" and quesito == "chi":
        return f"Chi ha {sn(rel['obj'], contesto)}?"
    if radice == "portare" and quesito == "quanti":
        return f"Quanti oggetti porta {sn(rel['nsubj'], contesto)}?"
    if radice == "esserci" and quesito == "quanti":
        return f"Quanti oggetti ci sono {_testo_luogo_o_contenitore(rel['obl:luogo'])}?"
    if radice == "dare" and quesito == "chi":
        return f"Chi {_ha_dato()} {sn(rel['obj'], contesto)} a {sn(rel['iobj'], contesto)}?"
    if radice == "essere" and quesito == "che-parente":
        return f"Che parente è {sn(rel['nsubj'], contesto)} di {sn(rel['nmod:relativo'], contesto)}?"
    if radice == "dormire" and quesito == "perche":
        return f"Perché {sn(rel['nsubj'], contesto)} dorme?"
    if radice == "raccogliere" and quesito == "quante":
        lemma_u = rel["obj"]
        if lemma_u == "mela":
            return "Quante mele sono state raccolte?"
        return f"Quante volte è stata raccolta {_rendi_nome_massa(lemma_u)}?"
    raise ValueError(f"nessuno stampo di domanda per radice={radice!r} quesito={quesito!r}")


def _rendi_raccolta_risposta(lemma_u: str, n: int) -> str:
    if lemma_u == "mela":
        if n == 0:
            return "Non è stata raccolta nessuna mela."
        if n == 1:
            return "È stata raccolta una mela."
        return f"Sono state raccolte {mf.numero_in_lettere(n)} mele."
    soggetto = _capitalizza(_rendi_nome_massa(lemma_u))
    if n == 0:
        return f"{soggetto} non è mai stata raccolta."
    if n == 1:
        return f"{soggetto} è stata raccolta una volta."
    return f"{soggetto} è stata raccolta {mf.numero_in_lettere(n)} volte."


def rendi_risposta(grafo: Grafo, contesto: StatoDiscorso) -> str:
    if grafo == NON_LO_SO:
        return "Non lo so."
    radice = grafo.nodi[0].lemma
    rel = {a.relazione: grafo.nodi[a.dipendente].lemma for a in grafo.archi}

    # Le risposte non hanno prefisso: la prima lettera va sempre capitalizzata
    # (anche quando il primo elemento è un sintagma comune, non un nome
    # proprio), quindi si costruisce la frase in minuscolo e si capitalizza
    # una sola volta, alla fine.
    if radice == "essere" and "nmod:parentela" in rel:
        a, b, relazione = rel["nsubj"], rel["nmod:relativo"], rel["nmod:parentela"]
        testo = f"{sn(a, contesto)} è {_sn_parentela(relazione, a)} di {sn(b, contesto)}."
    elif radice == "essere":
        e, luogo = rel["nsubj"], rel["obl:luogo"]
        testo = f"{sn(e, contesto)} è {mf.loc_in(luogo)}."
    elif radice == "avere":
        p, o = rel["nsubj"], rel["obj"]
        testo = f"{sn(p, contesto)} ha {sn(o, contesto)}."
    elif radice == "portare":
        p, n = rel["nsubj"], int(rel["obl:quantita"])
        soggetto = sn(p, contesto)
        if n == 0:
            testo = f"{soggetto} non porta nessun oggetto."
        elif n == 1:
            testo = f"{soggetto} porta un oggetto."
        else:
            testo = f"{soggetto} porta {mf.numero_in_lettere(n)} oggetti."
    elif radice == "esserci":
        b, n = rel["obl:luogo"], int(rel["obl:quantita"])
        testo_b = _testo_luogo_o_contenitore(b)
        if n == 0:
            testo = f"{testo_b} non c'è nessun oggetto."
        elif n == 1:
            testo = f"{testo_b} c'è un oggetto."
        else:
            testo = f"{testo_b} ci sono {mf.numero_in_lettere(n)} oggetti."
    elif radice == "dare":
        a, o, d = rel["nsubj"], rel["obj"], rel["iobj"]
        testo = f"{sn(a, contesto)} {_ha_dato()} {sn(o, contesto)} a {sn(d, contesto)}."
    elif radice == "dormire":
        p = rel["nsubj"]
        testo = f"{sn(p, contesto)} {mf.forma_verbale('dormire', 'pres3s')} perché è {mf.aggettivo('stanco', _genere(p))}."
    elif radice == "raccogliere":
        testo = _rendi_raccolta_risposta(rel["obj"], int(rel["obl:quantita"]))
    else:
        raise ValueError(f"nessuno stampo di risposta per radice={radice!r}")
    return _capitalizza(testo)


# --- riconoscimento ----------------------------------------------------

def analizza_domanda(frase: str, contesto: StatoDiscorso) -> Grafo:
    if not frase.endswith("?"):
        raise ValueError(f"domanda senza punto interrogativo finale: {frase!r}")
    corpo = frase[:-1]

    if corpo.startswith("Dove si trova "):
        resto = corpo[len("Dove si trova "):]
        idx_che = resto.find(" che ")
        if idx_che != -1:
            o_testo = resto[:idx_che]
            resto2 = resto[idx_che + len(" che "):]
            token_a, spazio, resto3 = resto2.partition(" ")
            if not spazio:
                raise ValueError(f"domanda di deduzione malformata: {frase!r}")
            pref2 = f"{_ha_dato()} a "
            if not resto3.startswith(pref2):
                raise ValueError(f"domanda di deduzione malformata: {frase!r}")
            d_testo = resto3[len(pref2):]
            return grafo_fatto("trovarsi", **{
                "nmod:agente": sn_inversa(token_a, contesto),
                "nmod:oggetto": sn_inversa(o_testo, contesto),
                "nmod:destinatario": sn_inversa(d_testo, contesto),
                "quesito": "dove",
            })
        return grafo_fatto("trovarsi", nsubj=sn_inversa(resto, contesto), quesito="dove")

    if corpo.startswith("Chi ha "):
        resto = corpo[len("Chi ha "):]
        pref_dato = f"{mf.forma_verbale('dare', 'part')} "
        if resto.startswith(pref_dato):
            resto2 = resto[len(pref_dato):]
            idx = resto2.rfind(" a ")
            if idx == -1:
                raise ValueError(f"domanda di transfer malformata: {frase!r}")
            o_id = sn_inversa(resto2[:idx], contesto)
            d_id = sn_inversa(resto2[idx + len(" a "):], contesto)
            return grafo_fatto("dare", obj=o_id, iobj=d_id, quesito="chi")
        return grafo_fatto("avere", obj=sn_inversa(resto, contesto), quesito="chi")

    if corpo.startswith("Quanti oggetti porta "):
        p_testo = corpo[len("Quanti oggetti porta "):]
        return grafo_fatto("portare", nsubj=sn_inversa(p_testo, contesto), quesito="quanti")

    if corpo.startswith("Quanti oggetti ci sono "):
        b_testo = corpo[len("Quanti oggetti ci sono "):]
        b_id = _luogo_o_contenitore_da_testo(b_testo)
        return grafo_fatto("esserci", **{"obl:luogo": b_id, "quesito": "quanti"})

    if corpo.startswith("Che parente è "):
        resto = corpo[len("Che parente è "):]
        idx = resto.find(" di ")
        if idx == -1:
            raise ValueError(f"domanda di parentela malformata: {frase!r}")
        a_id = sn_inversa(resto[:idx], contesto)
        b_id = sn_inversa(resto[idx + len(" di "):], contesto)
        return grafo_fatto("essere", nsubj=a_id, **{"nmod:relativo": b_id, "quesito": "che-parente"})

    if corpo.startswith("Perché ") and corpo.endswith(" dorme"):
        p_testo = corpo[len("Perché "):-len(" dorme")]
        return grafo_fatto("dormire", nsubj=sn_inversa(p_testo, contesto), quesito="perche")

    if corpo == "Quante mele sono state raccolte":
        return grafo_fatto("raccogliere", obj="mela", quesito="quante")

    pref_volte = "Quante volte è stata raccolta "
    if corpo.startswith(pref_volte):
        lemma_u = _analizza_nome_massa(corpo[len(pref_volte):])
        return grafo_fatto("raccogliere", obj=lemma_u, quesito="quante")

    raise ValueError(f"nessuno stampo di domanda riconosce la frase: {frase!r}")


def _prova_risposta_raccolta(corpo: str, contesto: StatoDiscorso) -> Grafo | None:
    if corpo == "Non è stata raccolta nessuna mela":
        return grafo_fatto("raccogliere", obj="mela", **{"obl:quantita": "0"})
    if corpo == "È stata raccolta una mela":
        return grafo_fatto("raccogliere", obj="mela", **{"obl:quantita": "1"})
    pref, suff = "Sono state raccolte ", " mele"
    if corpo.startswith(pref) and corpo.endswith(suff):
        n = mf.numero_da_lettere(corpo[len(pref):-len(suff)])
        return grafo_fatto("raccogliere", obj="mela", **{"obl:quantita": str(n)})
    for lemma_u in ("acqua", "legna"):
        soggetto = _capitalizza(_rendi_nome_massa(lemma_u))
        if corpo == f"{soggetto} non è mai stata raccolta":
            return grafo_fatto("raccogliere", obj=lemma_u, **{"obl:quantita": "0"})
        if corpo == f"{soggetto} è stata raccolta una volta":
            return grafo_fatto("raccogliere", obj=lemma_u, **{"obl:quantita": "1"})
        pref2, suff2 = f"{soggetto} è stata raccolta ", " volte"
        if corpo.startswith(pref2) and corpo.endswith(suff2):
            n = mf.numero_da_lettere(corpo[len(pref2):-len(suff2)])
            return grafo_fatto("raccogliere", obj=lemma_u, **{"obl:quantita": str(n)})
    return None


def _prova_risposta_transfer(corpo: str, contesto: StatoDiscorso) -> Grafo | None:
    marcatore = f" {_ha_dato()} "
    idx = corpo.find(marcatore)
    if idx == -1:
        return None
    a_testo = corpo[:idx]
    resto = corpo[idx + len(marcatore):]
    idx2 = resto.rfind(" a ")
    if idx2 == -1:
        return None
    o_id = sn_inversa(resto[:idx2], contesto)
    d_id = sn_inversa(resto[idx2 + len(" a "):], contesto)
    a_id = sn_inversa(a_testo, contesto)
    return grafo_fatto("dare", nsubj=a_id, obj=o_id, iobj=d_id)


def _prova_risposta_parentela(corpo: str, contesto: StatoDiscorso) -> Grafo | None:
    idx = corpo.find(" è ")
    if idx == -1:
        return None
    a_testo = corpo[:idx]
    resto = corpo[idx + len(" è "):]
    idx2 = resto.find(" di ")
    if idx2 == -1:
        return None
    relazione_testo, b_testo = resto[:idx2], resto[idx2 + len(" di "):]
    if " " not in relazione_testo:
        return None
    _articolo, sostantivo = relazione_testo.split(" ", 1)
    relazione = _PARENTELA_SOSTANTIVO_A_RELAZIONE.get(sostantivo)
    if relazione is None:
        return None
    a_id = sn_inversa(a_testo, contesto)
    b_id = sn_inversa(b_testo, contesto)
    return grafo_fatto("essere", nsubj=a_id, **{"nmod:parentela": relazione, "nmod:relativo": b_id})


def _prova_risposta_posizione(corpo: str, contesto: StatoDiscorso) -> Grafo | None:
    idx = corpo.find(" è ")
    if idx == -1:
        return None
    e_testo = corpo[:idx]
    l_testo = corpo[idx + len(" è "):]
    try:
        luogo_id = mf.luogo_da_loc_in(l_testo)
    except ValueError:
        return None
    e_id = sn_inversa(e_testo, contesto)
    return grafo_fatto("essere", nsubj=e_id, **{"obl:luogo": luogo_id})


def _prova_risposta_possesso(corpo: str, contesto: StatoDiscorso) -> Grafo | None:
    marcatore = f" {mf.forma_verbale('avere', 'pres3s')} "
    idx = corpo.find(marcatore)
    if idx == -1:
        return None
    p_testo = corpo[:idx]
    o_testo = corpo[idx + len(marcatore):]
    p_id = "nessuno" if p_testo == "Nessuno" else sn_inversa(p_testo, contesto)
    o_id = sn_inversa(o_testo, contesto)
    return grafo_fatto("avere", nsubj=p_id, obj=o_id)


def _prova_risposta_conteggio_persona(corpo: str, contesto: StatoDiscorso) -> Grafo | None:
    if corpo.endswith(" non porta nessun oggetto"):
        p_id = sn_inversa(corpo[:-len(" non porta nessun oggetto")], contesto)
        return grafo_fatto("portare", nsubj=p_id, **{"obl:quantita": "0"})
    if corpo.endswith(" porta un oggetto"):
        p_id = sn_inversa(corpo[:-len(" porta un oggetto")], contesto)
        return grafo_fatto("portare", nsubj=p_id, **{"obl:quantita": "1"})
    idx = corpo.find(" porta ")
    if idx != -1 and corpo.endswith(" oggetti"):
        n_testo = corpo[idx + len(" porta "):-len(" oggetti")]
        try:
            n = mf.numero_da_lettere(n_testo)
        except ValueError:
            return None
        p_id = sn_inversa(corpo[:idx], contesto)
        return grafo_fatto("portare", nsubj=p_id, **{"obl:quantita": str(n)})
    return None


def _prova_risposta_conteggio_posto(corpo: str, contesto: StatoDiscorso) -> Grafo | None:
    if corpo.endswith(" non c'è nessun oggetto"):
        b_id = _luogo_o_contenitore_da_testo(_decapitalizza(corpo[:-len(" non c'è nessun oggetto")]))
        return grafo_fatto("esserci", **{"obl:luogo": b_id, "obl:quantita": "0"})
    if corpo.endswith(" c'è un oggetto"):
        b_id = _luogo_o_contenitore_da_testo(_decapitalizza(corpo[:-len(" c'è un oggetto")]))
        return grafo_fatto("esserci", **{"obl:luogo": b_id, "obl:quantita": "1"})
    idx = corpo.find(" ci sono ")
    if idx != -1 and corpo.endswith(" oggetti"):
        n_testo = corpo[idx + len(" ci sono "):-len(" oggetti")]
        try:
            n = mf.numero_da_lettere(n_testo)
        except ValueError:
            return None
        b_id = _luogo_o_contenitore_da_testo(_decapitalizza(corpo[:idx]))
        return grafo_fatto("esserci", **{"obl:luogo": b_id, "obl:quantita": str(n)})
    return None


def _prova_risposta_causa(corpo: str, contesto: StatoDiscorso) -> Grafo | None:
    marcatore = " dorme perché è "
    idx = corpo.find(marcatore)
    if idx == -1:
        return None
    agg_testo = corpo[idx + len(marcatore):]
    if agg_testo not in (mf.aggettivo("stanco", "m"), mf.aggettivo("stanco", "f")):
        return None
    p_id = sn_inversa(corpo[:idx], contesto)
    return grafo_fatto("dormire", nsubj=p_id, **{"advcl:causa": "stanchezza"})


_PROVE_RISPOSTA = (
    _prova_risposta_raccolta,
    _prova_risposta_transfer,
    _prova_risposta_parentela,
    _prova_risposta_posizione,
    _prova_risposta_possesso,
    _prova_risposta_conteggio_persona,
    _prova_risposta_conteggio_posto,
    _prova_risposta_causa,
)


def analizza_risposta(frase: str, contesto: StatoDiscorso) -> Grafo:
    if frase == "Non lo so.":
        return NON_LO_SO
    if not frase.endswith("."):
        raise ValueError(f"risposta senza punto finale: {frase!r}")
    corpo = frase[:-1]
    for prova in _PROVE_RISPOSTA:
        grafo = prova(corpo, contesto)
        if grafo is not None:
            return grafo
    raise ValueError(f"nessuno stampo di risposta riconosce la frase: {frase!r}")
