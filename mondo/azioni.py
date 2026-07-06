"""Le azioni del micro-mondo, dichiarate come dati (stile STRIPS).

Ogni azione è un `Azione`: una tripletta (candidati, precondizioni, effetti).
Aggiungere un'azione = aggiungere una voce a `AZIONI` in questo file, nessun
altro modulo va toccato.

- `genera_candidati(stato, agente)` enumera, per un agente SVEGLIO (è la
  politica a garantirlo: chi dorme non sceglie azioni), le istanziazioni
  VALIDE dei parametri: ogni candidato emesso soddisfa le precondizioni
  (proprietà pretesa dal motore, che nel percorso caldo non riverifica, e
  fissata da tests/test_mondo.py::test_ogni_candidato_soddisfa_le_precondizioni).
- `precondizioni(stato, parametri)` verifica che l'istanza sia davvero
  eseguibile: è la parte "STRIPS" in senso stretto, separata dagli effetti
  così da poter essere testata in isolamento e da fare da rete di sicurezza
  per parametri costruiti a mano.
- `effetti(stato, parametri, t)` muta `stato` e ritorna l'`Evento` generato.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import dati_mondo as dm
from .tipi import Evento, StatoMondo, StatoOggetto


@dataclass(frozen=True)
class Azione:
    nome: str
    genera_candidati: Callable[[StatoMondo, str], list[dict]]
    precondizioni: Callable[[StatoMondo, dict], bool]
    effetti: Callable[[StatoMondo, dict, int], Evento]


def istanze_valide(azione: Azione, stato: StatoMondo, agente: str) -> list[dict]:
    return [p for p in azione.genera_candidati(stato, agente) if azione.precondizioni(stato, p)]


def _raggiungibile(stato: StatoMondo, entita_id: str, agente: str) -> bool:
    """Vero se l'oggetto è nello stesso luogo dell'agente (per terra, portato
    da lui, o dentro un contenitore aperto lì presente)."""
    return stato.luogo_effettivo(entita_id) == stato.persone[agente].luogo


def _oggetti_raggiungibili(stato: StatoMondo, agente: str, filtro=None) -> list[str]:
    luogo = stato.persone[agente].luogo
    risultato = []
    luogo_contenitore: dict[str, str] = {}  # memo: il luogo di ogni contenitore, risolto una volta
    for oid, o in stato.oggetti.items():
        if filtro is not None and not filtro(o):
            continue
        tipo, rif = o.posizione
        # scorciatoia per i due casi comuni (niente ricorsione su
        # luogo_effettivo): un oggetto appoggiato o in mano si risolve in
        # un solo dizionario, non serve la catena generica.
        if tipo == "luogo":
            if rif == luogo:
                risultato.append(oid)
            continue
        if tipo == "persona":
            if stato.persone[rif].luogo == luogo:
                risultato.append(oid)
            continue
        contenitore = stato.oggetti[rif]
        if contenitore.apribile and not contenitore.aperto:
            continue
        if rif not in luogo_contenitore:
            luogo_contenitore[rif] = stato.luogo_effettivo(rif)
        if luogo_contenitore[rif] == luogo:
            risultato.append(oid)
    return risultato


# ---------------------------------------------------------------------------
# andare
# ---------------------------------------------------------------------------

def _andare_candidati(stato: StatoMondo, agente: str) -> list[dict]:
    origine = stato.persone[agente].luogo
    return [{"agente": agente, "luogo_destinazione": dest} for dest in sorted(stato.collegamenti[origine])]


def _andare_precondizioni(stato: StatoMondo, p: dict) -> bool:
    persona = stato.persone[p["agente"]]
    if persona.addormentato:
        return False
    origine = persona.luogo
    return p["luogo_destinazione"] in stato.collegamenti[origine]


def _andare_effetti(stato: StatoMondo, p: dict, t: int) -> Evento:
    agente = p["agente"]
    origine = stato.persone[agente].luogo
    destinazione = p["luogo_destinazione"]
    testimoni = stato.testimoni_in(origine)
    stato.persone[agente].luogo = destinazione
    return Evento(t=t, azione="andare", agente=agente, luogo=destinazione,
                  luogo_origine=origine, testimoni=testimoni)


AZIONE_ANDARE = Azione("andare", _andare_candidati, _andare_precondizioni, _andare_effetti)


# ---------------------------------------------------------------------------
# prendere (oggetto libero, o unità da una risorsa finita)
# ---------------------------------------------------------------------------

def _prendere_candidati(stato: StatoMondo, agente: str) -> list[dict]:
    luogo = stato.persone[agente].luogo
    candidati = []
    for oid in stato.oggetti_in_luogo(luogo):
        if stato.oggetti[oid].fisso:
            continue
        candidati.append({"agente": agente, "oggetto": oid, "fonte": None})
    for fonte, info in dm.RISORSE.items():
        if info["luogo"] != luogo or stato.risorse.get(fonte, 0) <= 0:
            continue
        attrezzo = dm.ATTREZZO_RICHIESTO.get(fonte)
        if attrezzo is not None and attrezzo not in stato.oggetti_portati_da(agente):
            continue
        candidati.append({"agente": agente, "oggetto": None, "fonte": fonte})
    return candidati


def _prendere_precondizioni(stato: StatoMondo, p: dict) -> bool:
    if stato.persone[p["agente"]].addormentato:
        return False
    if p["fonte"] is not None:
        fonte = p["fonte"]
        if stato.risorse.get(fonte, 0) <= 0:
            return False
        attrezzo = dm.ATTREZZO_RICHIESTO.get(fonte)
        if attrezzo is not None and attrezzo not in stato.oggetti_portati_da(p["agente"]):
            return False
        return dm.RISORSE[fonte]["luogo"] == stato.persone[p["agente"]].luogo
    oid = p["oggetto"]
    if oid not in stato.oggetti:
        return False
    o = stato.oggetti[oid]
    if o.fisso:
        return False
    return o.posizione == ("luogo", stato.persone[p["agente"]].luogo)


def _prendere_effetti(stato: StatoMondo, p: dict, t: int) -> Evento:
    agente = p["agente"]
    luogo = stato.persone[agente].luogo
    testimoni = stato.testimoni_in(luogo)
    fonte = p["fonte"]
    if fonte is not None:
        info = dm.RISORSE[fonte]
        stato.risorse[fonte] -= 1
        nuovo_id = stato.nuovo_id(info["lemma_unita"])
        stato.oggetti[nuovo_id] = StatoOggetto(
            id=nuovo_id, lemma=info["lemma_unita"], commestibile=info["commestibile"],
            posizione=("persona", agente),
        )
        oggetto_id = nuovo_id
    else:
        oggetto_id = p["oggetto"]
        stato.oggetti[oggetto_id].posizione = ("persona", agente)
    # argomento = fonte da cui è stata raccolta l'unità (None se l'oggetto
    # era già libero nel mondo): distingue "raccogliere una mela dal melo"
    # da "prendere la mela [che era per terra]", utile alla Fase 1 per
    # scegliere il verbo e a questo modulo per verificare la conservazione.
    return Evento(t=t, azione="prendere", agente=agente, oggetto=oggetto_id, luogo=luogo,
                  argomento=fonte, testimoni=testimoni)


AZIONE_PRENDERE = Azione("prendere", _prendere_candidati, _prendere_precondizioni, _prendere_effetti)


# ---------------------------------------------------------------------------
# posare
# ---------------------------------------------------------------------------

def _posare_candidati(stato: StatoMondo, agente: str) -> list[dict]:
    return [{"agente": agente, "oggetto": oid} for oid in stato.oggetti_portati_da(agente)]


def _posare_precondizioni(stato: StatoMondo, p: dict) -> bool:
    o = stato.oggetti.get(p["oggetto"])
    return o is not None and o.posizione == ("persona", p["agente"])


def _posare_effetti(stato: StatoMondo, p: dict, t: int) -> Evento:
    agente = p["agente"]
    luogo = stato.persone[agente].luogo
    stato.oggetti[p["oggetto"]].posizione = ("luogo", luogo)
    return Evento(t=t, azione="posare", agente=agente, oggetto=p["oggetto"], luogo=luogo,
                  testimoni=stato.testimoni_in(luogo))


AZIONE_POSARE = Azione("posare", _posare_candidati, _posare_precondizioni, _posare_effetti)


# ---------------------------------------------------------------------------
# mettere_dentro
# ---------------------------------------------------------------------------

def _contenitori_raggiungibili_aperti(stato: StatoMondo, agente: str) -> list[str]:
    luogo = stato.persone[agente].luogo
    risultato = []
    for oid, o in stato.oggetti.items():
        if not o.contenitore:
            continue
        if o.apribile and not o.aperto:
            continue
        if stato.luogo_effettivo(oid) == luogo:
            risultato.append(oid)
    return risultato


def _mettere_dentro_candidati(stato: StatoMondo, agente: str) -> list[dict]:
    # Un contenitore non può mai finire dentro un altro contenitore: evita
    # cicli di contenimento (A dentro B dentro A) senza bisogno di
    # rilevarli a runtime.
    oggetti = [oid for oid in stato.oggetti_portati_da(agente) if not stato.oggetti[oid].contenitore]
    contenitori = _contenitori_raggiungibili_aperti(stato, agente)
    return [
        {"agente": agente, "oggetto": oid, "contenitore": cid}
        for oid in oggetti for cid in contenitori if oid != cid
    ]


def _mettere_dentro_precondizioni(stato: StatoMondo, p: dict) -> bool:
    o = stato.oggetti.get(p["oggetto"])
    c = stato.oggetti.get(p["contenitore"])
    if o is None or c is None or not c.contenitore or o.contenitore:
        return False
    if o.posizione != ("persona", p["agente"]):
        return False
    if c.apribile and not c.aperto:
        return False
    return stato.luogo_effettivo(p["contenitore"]) == stato.persone[p["agente"]].luogo


def _mettere_dentro_effetti(stato: StatoMondo, p: dict, t: int) -> Evento:
    agente = p["agente"]
    stato.oggetti[p["oggetto"]].posizione = ("contenitore", p["contenitore"])
    luogo = stato.persone[agente].luogo
    return Evento(t=t, azione="mettere_dentro", agente=agente, oggetto=p["oggetto"],
                  argomento=p["contenitore"], luogo=luogo, testimoni=stato.testimoni_in(luogo))


AZIONE_METTERE_DENTRO = Azione("mettere_dentro", _mettere_dentro_candidati,
                                _mettere_dentro_precondizioni, _mettere_dentro_effetti)


# ---------------------------------------------------------------------------
# tirare_fuori
# ---------------------------------------------------------------------------

def _tirare_fuori_candidati(stato: StatoMondo, agente: str) -> list[dict]:
    luogo = stato.persone[agente].luogo
    candidati = []
    for cid in _contenitori_raggiungibili_aperti(stato, agente):
        for oid in stato.oggetti_dentro(cid):
            candidati.append({"agente": agente, "oggetto": oid, "contenitore": cid})
    return candidati


def _tirare_fuori_precondizioni(stato: StatoMondo, p: dict) -> bool:
    o = stato.oggetti.get(p["oggetto"])
    c = stato.oggetti.get(p["contenitore"])
    if o is None or c is None:
        return False
    if o.posizione != ("contenitore", p["contenitore"]):
        return False
    if c.apribile and not c.aperto:
        return False
    return stato.luogo_effettivo(p["contenitore"]) == stato.persone[p["agente"]].luogo


def _tirare_fuori_effetti(stato: StatoMondo, p: dict, t: int) -> Evento:
    agente = p["agente"]
    stato.oggetti[p["oggetto"]].posizione = ("persona", agente)
    luogo = stato.persone[agente].luogo
    return Evento(t=t, azione="tirare_fuori", agente=agente, oggetto=p["oggetto"],
                  argomento=p["contenitore"], luogo=luogo, testimoni=stato.testimoni_in(luogo))


AZIONE_TIRARE_FUORI = Azione("tirare_fuori", _tirare_fuori_candidati,
                              _tirare_fuori_precondizioni, _tirare_fuori_effetti)


# ---------------------------------------------------------------------------
# dare
# ---------------------------------------------------------------------------

def _dare_candidati(stato: StatoMondo, agente: str) -> list[dict]:
    luogo = stato.persone[agente].luogo
    oggetti = stato.oggetti_portati_da(agente)
    destinatari = [pid for pid, pp in stato.persone.items()
                   if pp.luogo == luogo and pid != agente and not pp.addormentato]
    return [{"agente": agente, "oggetto": oid, "destinatario": did} for oid in oggetti for did in destinatari]


def _dare_precondizioni(stato: StatoMondo, p: dict) -> bool:
    o = stato.oggetti.get(p["oggetto"])
    dest = stato.persone.get(p["destinatario"])
    if o is None or dest is None or o.posizione != ("persona", p["agente"]):
        return False
    if dest.addormentato:
        return False
    return dest.luogo == stato.persone[p["agente"]].luogo


def _dare_effetti(stato: StatoMondo, p: dict, t: int) -> Evento:
    agente = p["agente"]
    stato.oggetti[p["oggetto"]].posizione = ("persona", p["destinatario"])
    luogo = stato.persone[agente].luogo
    return Evento(t=t, azione="dare", agente=agente, oggetto=p["oggetto"], destinatario=p["destinatario"],
                  luogo=luogo, testimoni=stato.testimoni_in(luogo))


AZIONE_DARE = Azione("dare", _dare_candidati, _dare_precondizioni, _dare_effetti)


# ---------------------------------------------------------------------------
# mangiare
# ---------------------------------------------------------------------------

def _mangiare_candidati(stato: StatoMondo, agente: str) -> list[dict]:
    return [{"agente": agente, "oggetto": oid}
            for oid in _oggetti_raggiungibili(stato, agente, lambda o: o.commestibile)]


def _mangiare_precondizioni(stato: StatoMondo, p: dict) -> bool:
    o = stato.oggetti.get(p["oggetto"])
    if o is None or not o.commestibile:
        return False
    if stato.persone[p["agente"]].addormentato:
        return False
    return _raggiungibile(stato, p["oggetto"], p["agente"])


def _mangiare_effetti(stato: StatoMondo, p: dict, t: int) -> Evento:
    agente = p["agente"]
    luogo = stato.persone[agente].luogo
    testimoni = stato.testimoni_in(luogo)
    del stato.oggetti[p["oggetto"]]
    persona = stato.persone[agente]
    persona.fame = max(0, persona.fame - dm.RISTORO_FAME_MANGIARE)
    return Evento(t=t, azione="mangiare", agente=agente, oggetto=p["oggetto"], luogo=luogo, testimoni=testimoni)


AZIONE_MANGIARE = Azione("mangiare", _mangiare_candidati, _mangiare_precondizioni, _mangiare_effetti)


# ---------------------------------------------------------------------------
# aprire / chiudere
# ---------------------------------------------------------------------------

def _contenitori_apribili_raggiungibili(stato: StatoMondo, agente: str, aperto: bool) -> list[str]:
    luogo = stato.persone[agente].luogo
    risultato = []
    for oid, o in stato.oggetti.items():
        if not (o.contenitore and o.apribile) or o.aperto != aperto:
            continue
        tipo, rif = o.posizione
        if tipo == "persona" and rif == agente:
            risultato.append(oid)
        elif stato.luogo_effettivo(oid) == luogo:
            risultato.append(oid)
    return risultato


def _aprire_candidati(stato: StatoMondo, agente: str) -> list[dict]:
    return [{"agente": agente, "oggetto": oid}
            for oid in _contenitori_apribili_raggiungibili(stato, agente, aperto=False)]


def _aprire_precondizioni(stato: StatoMondo, p: dict) -> bool:
    o = stato.oggetti.get(p["oggetto"])
    return o is not None and o.apribile and not o.aperto


def _aprire_effetti(stato: StatoMondo, p: dict, t: int) -> Evento:
    agente = p["agente"]
    stato.oggetti[p["oggetto"]].aperto = True
    luogo = stato.persone[agente].luogo
    return Evento(t=t, azione="aprire", agente=agente, oggetto=p["oggetto"], luogo=luogo,
                  testimoni=stato.testimoni_in(luogo))


AZIONE_APRIRE = Azione("aprire", _aprire_candidati, _aprire_precondizioni, _aprire_effetti)


def _chiudere_candidati(stato: StatoMondo, agente: str) -> list[dict]:
    return [{"agente": agente, "oggetto": oid}
            for oid in _contenitori_apribili_raggiungibili(stato, agente, aperto=True)]


def _chiudere_precondizioni(stato: StatoMondo, p: dict) -> bool:
    o = stato.oggetti.get(p["oggetto"])
    return o is not None and o.apribile and o.aperto


def _chiudere_effetti(stato: StatoMondo, p: dict, t: int) -> Evento:
    agente = p["agente"]
    stato.oggetti[p["oggetto"]].aperto = False
    luogo = stato.persone[agente].luogo
    return Evento(t=t, azione="chiudere", agente=agente, oggetto=p["oggetto"], luogo=luogo,
                  testimoni=stato.testimoni_in(luogo))


AZIONE_CHIUDERE = Azione("chiudere", _chiudere_candidati, _chiudere_precondizioni, _chiudere_effetti)


# ---------------------------------------------------------------------------
# guardare
# ---------------------------------------------------------------------------

def _guardare_candidati(stato: StatoMondo, agente: str) -> list[dict]:
    luogo = stato.persone[agente].luogo
    bersagli = _oggetti_raggiungibili(stato, agente)
    bersagli += [pid for pid, pp in stato.persone.items() if pp.luogo == luogo and pid != agente]
    return [{"agente": agente, "oggetto": b} for b in bersagli]


def _guardare_precondizioni(stato: StatoMondo, p: dict) -> bool:
    return _raggiungibile(stato, p["oggetto"], p["agente"])


def _guardare_effetti(stato: StatoMondo, p: dict, t: int) -> Evento:
    agente = p["agente"]
    luogo = stato.persone[agente].luogo
    return Evento(t=t, azione="guardare", agente=agente, oggetto=p["oggetto"], luogo=luogo,
                  testimoni=stato.testimoni_in(luogo))


AZIONE_GUARDARE = Azione("guardare", _guardare_candidati, _guardare_precondizioni, _guardare_effetti)


# ---------------------------------------------------------------------------
# dire
# ---------------------------------------------------------------------------

def _dire_candidati(stato: StatoMondo, agente: str) -> list[dict]:
    luogo = stato.persone[agente].luogo
    return [{"agente": agente, "destinatario": pid}
            for pid, pp in stato.persone.items()
            if pp.luogo == luogo and pid != agente and not pp.addormentato]


def _dire_precondizioni(stato: StatoMondo, p: dict) -> bool:
    dest = stato.persone.get(p["destinatario"])
    return dest is not None and dest.luogo == stato.persone[p["agente"]].luogo and not dest.addormentato


def _dire_effetti(stato: StatoMondo, p: dict, t: int) -> Evento:
    agente = p["agente"]
    luogo = stato.persone[agente].luogo
    return Evento(t=t, azione="dire", agente=agente, destinatario=p["destinatario"], luogo=luogo,
                  testimoni=stato.testimoni_in(luogo))


AZIONE_DIRE = Azione("dire", _dire_candidati, _dire_precondizioni, _dire_effetti)


# ---------------------------------------------------------------------------
# dormire / svegliarsi
# ---------------------------------------------------------------------------

def _dormire_candidati(stato: StatoMondo, agente: str) -> list[dict]:
    if _dormire_precondizioni(stato, {"agente": agente}):
        return [{"agente": agente}]
    return []


def _dormire_precondizioni(stato: StatoMondo, p: dict) -> bool:
    persona = stato.persone[p["agente"]]
    # Niente pisolini da riposati: sotto SOGLIA_PISOLINO non ci si addormenta.
    return not persona.addormentato and persona.stanchezza >= dm.SOGLIA_PISOLINO


def _dormire_effetti(stato: StatoMondo, p: dict, t: int) -> Evento:
    agente = p["agente"]
    persona = stato.persone[agente]
    luogo = persona.luogo
    # Testimoni PRIMA di addormentarsi: chi si addormenta vede sé stesso farlo.
    testimoni = stato.testimoni_in(luogo)
    # La causa è un fatto del mondo, registrato nell'evento: "stanchezza" solo
    # se il sonno è dettato dall'esaustione; un pisolino volontario non ha una
    # causa determinabile (la domanda "perché dorme?" avrà oro "non-lo-so").
    # La stanchezza NON si azzera qui: si recupera tick per tick dormendo
    # (motore._aggiorna_fisiologia), il sonno dura più tick.
    causa = "stanchezza" if persona.stanchezza >= dm.SOGLIA_ESAUSTO_PER_ETA[persona.eta] else None
    persona.addormentato = True
    return Evento(t=t, azione="dormire", agente=agente, luogo=luogo,
                  argomento=causa, testimoni=testimoni)


AZIONE_DORMIRE = Azione("dormire", _dormire_candidati, _dormire_precondizioni, _dormire_effetti)


def _svegliarsi_candidati(stato: StatoMondo, agente: str) -> list[dict]:
    if stato.persone[agente].addormentato:
        return [{"agente": agente}]
    return []


def _svegliarsi_precondizioni(stato: StatoMondo, p: dict) -> bool:
    return stato.persone[p["agente"]].addormentato


def _svegliarsi_effetti(stato: StatoMondo, p: dict, t: int) -> Evento:
    agente = p["agente"]
    luogo = stato.persone[agente].luogo
    stato.persone[agente].addormentato = False
    return Evento(t=t, azione="svegliarsi", agente=agente, luogo=luogo, testimoni=stato.testimoni_in(luogo))


AZIONE_SVEGLIARSI = Azione("svegliarsi", _svegliarsi_candidati, _svegliarsi_precondizioni, _svegliarsi_effetti)


# ---------------------------------------------------------------------------
# giocare
# ---------------------------------------------------------------------------

def _giocare_candidati(stato: StatoMondo, agente: str) -> list[dict]:
    giocattoli = _oggetti_raggiungibili(stato, agente, lambda o: o.lemma == "palla")
    candidati = [{"agente": agente, "oggetto": g} for g in giocattoli]
    candidati.append({"agente": agente, "oggetto": None})
    return candidati


def _giocare_precondizioni(stato: StatoMondo, p: dict) -> bool:
    persona = stato.persone[p["agente"]]
    if persona.addormentato:
        return False
    if p["oggetto"] is not None:
        return _raggiungibile(stato, p["oggetto"], p["agente"])
    return True


def _giocare_effetti(stato: StatoMondo, p: dict, t: int) -> Evento:
    agente = p["agente"]
    luogo = stato.persone[agente].luogo
    persona = stato.persone[agente]
    persona.stanchezza = min(dm.SOGLIA_MASSIMA, persona.stanchezza + 1)
    return Evento(t=t, azione="giocare", agente=agente, oggetto=p["oggetto"], luogo=luogo,
                  testimoni=stato.testimoni_in(luogo))


AZIONE_GIOCARE = Azione("giocare", _giocare_candidati, _giocare_precondizioni, _giocare_effetti)


# ---------------------------------------------------------------------------
# cercare
# ---------------------------------------------------------------------------

def _cercare_candidati(stato: StatoMondo, agente: str) -> list[dict]:
    luogo = stato.persone[agente].luogo
    candidati = []
    luogo_contenitore: dict[str, str] = {}  # memo, come in _oggetti_raggiungibili
    for oid, o in stato.oggetti.items():
        tipo, rif = o.posizione
        if tipo == "luogo":
            altrove = rif != luogo
        elif tipo == "persona":
            altrove = stato.persone[rif].luogo != luogo
        else:
            if rif not in luogo_contenitore:
                luogo_contenitore[rif] = stato.luogo_effettivo(rif)
            altrove = luogo_contenitore[rif] != luogo
        if altrove:
            candidati.append({"agente": agente, "oggetto": oid})
    return candidati


def _cercare_precondizioni(stato: StatoMondo, p: dict) -> bool:
    if p["oggetto"] not in stato.oggetti:
        return False
    return stato.luogo_effettivo(p["oggetto"]) != stato.persone[p["agente"]].luogo


def _cercare_effetti(stato: StatoMondo, p: dict, t: int) -> Evento:
    agente = p["agente"]
    luogo = stato.persone[agente].luogo
    return Evento(t=t, azione="cercare", agente=agente, oggetto=p["oggetto"], luogo=luogo,
                  testimoni=stato.testimoni_in(luogo))


AZIONE_CERCARE = Azione("cercare", _cercare_candidati, _cercare_precondizioni, _cercare_effetti)


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------

AZIONI: dict[str, Azione] = {
    a.nome: a for a in (
        AZIONE_ANDARE, AZIONE_PRENDERE, AZIONE_POSARE, AZIONE_METTERE_DENTRO,
        AZIONE_TIRARE_FUORI, AZIONE_DARE, AZIONE_MANGIARE, AZIONE_APRIRE,
        AZIONE_CHIUDERE, AZIONE_GUARDARE, AZIONE_DIRE, AZIONE_DORMIRE,
        AZIONE_SVEGLIARSI, AZIONE_GIOCARE, AZIONE_CERCARE,
    )
}
