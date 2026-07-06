"""Generatore di domande con risposta esatta, calcolata dallo stato/eventi
della storia — mai scritta a mano, mai ambigua (FASE0.md).

Convenzione di lemma: i nodi dei grafi portano gli ID delle entità (gli
stessi usati in `Evento`: "sara", "mela_3", "cucina", ...), non il nome di
visualizzazione capitalizzato — evento_a_grafo funziona allo stesso modo,
ed è necessario perché la valutazione è grafo-vs-grafo (regola non
negoziabile #4): stessa entità, stesso nodo, sempre.

Ogni tipo di domanda mescola istanze derivabili e istanze la cui risposta
d'oro è "non lo so": la non-derivabilità è verificata formalmente (si
controlla l'assenza del fatto negli eventi/relazioni), non indovinata.

Nota su "parentela": con una sola famiglia chiusa di 6 persone, il calcolo
in parentela.py copre TUTTE le coppie (nessuna richiede più di 2 passi) —
quindi questo tipo, con questo mondo, non produce istanze "non lo so" per
costruzione, non per una svista. Le altre 6 categorie compensano: la quota
media resta nella fascia 15-20% richiesta da FASE0.md punto 8 (verificato
in statistiche.py).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from . import dati_mondo as dm
from . import parentela
from .grafo import NON_LO_SO, Grafo, grafo_a_dict, grafo_fatto
from .simulatore import Storia

# Quota di "non lo so" richiesta per tipo (~15-20% indicato da FASE0.md,
# punto 8). Non tutti i tipi hanno lo stesso margine in un mondo piccolo e
# molto interattivo come questo: posizione/possesso/conteggio/causa hanno un
# "bacino" naturale di fatti-non-menzionati piuttosto piccolo (pochi oggetti,
# pochi personaggi), quindi si richiede una quota più alta per avvicinarsi
# comunque al target; transfer/deduzione hanno invece un bacino di coppie
# non-avvenute enorme (quasi ogni combinazione oggetto-destinatario non è mai
# stata data), quindi una quota più bassa evita di sforare molto oltre il
# 20%. parentela resta a 0% per costruzione con questa famiglia chiusa (vedi
# nota nel docstring del modulo) e non ha una propria quota da tarare.
QUOTA_NON_LO_SO_PER_TIPO = {
    "posizione": 0.35,
    "possesso": 0.35,
    "conteggio": 0.35,
    "transfer": 0.12,
    "parentela": 0.18,
    "deduzione": 0.12,
    "causa": 0.35,
}


@dataclass(frozen=True)
class Domanda:
    tipo: str
    grafo_domanda: Grafo
    grafo_risposta: Grafo

    def to_dict(self) -> dict:
        return {
            "tipo": self.tipo,
            "grafo_domanda": grafo_a_dict(self.grafo_domanda),
            "grafo_risposta": grafo_a_dict(self.grafo_risposta),
        }


def _mescola(rng: random.Random, tipo: str, derivabili: list, non_derivabili: list, n: int) -> list:
    quota = QUOTA_NON_LO_SO_PER_TIPO[tipo]
    n_non = min(round(n * quota), len(non_derivabili))
    n_der = min(n - n_non, len(derivabili))
    scelti = rng.sample(derivabili, n_der) + rng.sample(non_derivabili, n_non)
    rng.shuffle(scelti)
    return scelti


def _oggetti_con_posizione_nota(storia: Storia) -> set[str]:
    """Oggetti la cui posizione è stabilita da almeno un evento della storia.

    "cercare" è escluso apposta: cercare X dice solo che X NON è lì, non
    rivela dove X si trovi davvero — non conta come "posizione nota".
    """
    noti: set[str] = set()
    for e in storia.eventi:
        if e.azione == "cercare":
            continue
        if e.oggetto is not None:
            noti.add(e.oggetto)
        if e.argomento is not None:
            noti.add(e.argomento)
    return noti


def _oggetti_mai_localizzati(storia: Storia) -> set[str]:
    return set(storia.stato_finale.oggetti.keys()) - _oggetti_con_posizione_nota(storia)


# ---------------------------------------------------------------------------
# 1. posizione
# ---------------------------------------------------------------------------

def _genera_posizione(storia: Storia, rng: random.Random, n: int) -> list[Domanda]:
    stato = storia.stato_finale
    mai_localizzati = _oggetti_mai_localizzati(storia)
    entita = list(stato.persone.keys()) + list(stato.oggetti.keys())

    derivabili = [e for e in entita if e not in mai_localizzati]
    non_derivabili = list(mai_localizzati)

    domande = []
    for entita_id in _mescola(rng, "posizione", derivabili, non_derivabili, n):
        grafo_domanda = grafo_fatto("trovarsi", nsubj=entita_id, quesito="dove")
        if entita_id in mai_localizzati:
            risposta = NON_LO_SO
        else:
            luogo = stato.luogo_effettivo(entita_id)
            risposta = grafo_fatto("essere", nsubj=entita_id, **{"obl:luogo": luogo})
        domande.append(Domanda("posizione", grafo_domanda, risposta))
    return domande


# ---------------------------------------------------------------------------
# 2. possesso
# ---------------------------------------------------------------------------

def _genera_possesso(storia: Storia, rng: random.Random, n: int) -> list[Domanda]:
    stato = storia.stato_finale
    mai_localizzati = _oggetti_mai_localizzati(storia)
    oggetti = list(stato.oggetti.keys())

    # Solo la parte dinamica può generare "non lo so" (il possesso statico è
    # un fatto sempre noto): le si lascia più spazio apposta.
    n_statico = max(1, n // 4)
    n_dinamico = n - n_statico

    domande: list[Domanda] = []

    # "Di chi è X?" — possesso statico, sempre un fatto noto (anche "di nessuno").
    for oid in rng.sample(oggetti, min(n_statico, len(oggetti))):
        o = stato.oggetti[oid]
        grafo_domanda = grafo_fatto("essere", nsubj=oid, quesito="di-chi")
        proprietario = o.proprietario if o.proprietario is not None else "nessuno"
        risposta = grafo_fatto("essere", nsubj=oid, **{"nmod:possesso": proprietario})
        domande.append(Domanda("possesso", grafo_domanda, risposta))

    # "Chi ha X adesso?" — dinamico: non derivabile se X non è mai stato localizzato.
    derivabili = [oid for oid in oggetti if oid not in mai_localizzati]
    non_derivabili = list(mai_localizzati)
    for oid in _mescola(rng, "possesso", derivabili, non_derivabili, n_dinamico):
        grafo_domanda = grafo_fatto("avere", obj=oid, quesito="chi")
        if oid in mai_localizzati:
            risposta = NON_LO_SO
        else:
            tipo, rif = stato.oggetti[oid].posizione
            portatore = rif if tipo == "persona" else "nessuno"
            risposta = grafo_fatto("avere", nsubj=portatore, obj=oid)
        domande.append(Domanda("possesso", grafo_domanda, risposta))

    return domande


# ---------------------------------------------------------------------------
# 3. conteggio (oggetti in un luogo o in un contenitore)
# ---------------------------------------------------------------------------

def _genera_conteggio(storia: Storia, rng: random.Random, n: int) -> list[Domanda]:
    stato = storia.stato_finale
    mai_localizzati = _oggetti_mai_localizzati(storia)
    contenitori = [oid for oid, o in stato.oggetti.items() if o.contenitore]
    luoghi = list(stato.luoghi.keys())

    # Solo i contenitori mai toccati possono dare "non lo so" (il conteggio
    # in un luogo è sempre un fatto noto): si dà loro più spazio apposta.
    n_luogo = max(1, n // 4)
    n_contenitore = n - n_luogo

    scelte = [("luogo", lid) for lid in rng.sample(luoghi, min(n_luogo, len(luoghi)))]

    bersagli_contenitore_derivabili = [("contenitore", cid) for cid in contenitori if cid not in mai_localizzati]
    bersagli_contenitore_non_derivabili = [("contenitore", cid) for cid in contenitori if cid in mai_localizzati]
    scelte += _mescola(rng, "conteggio", bersagli_contenitore_derivabili,
                        bersagli_contenitore_non_derivabili, n_contenitore)

    domande = []
    for tipo_bersaglio, bid in scelte:
        grafo_domanda = grafo_fatto("esserci", **{"obl:luogo": bid, "quesito": "quanti"})
        if tipo_bersaglio == "contenitore" and bid in mai_localizzati:
            risposta = NON_LO_SO
        else:
            quantita = (len(stato.oggetti_in_luogo(bid)) if tipo_bersaglio == "luogo"
                        else len(stato.oggetti_dentro(bid)))
            risposta = grafo_fatto("esserci", **{"obl:luogo": bid, "obl:quantita": str(quantita)})
        domande.append(Domanda("conteggio", grafo_domanda, risposta))
    return domande


# ---------------------------------------------------------------------------
# 4. transfer: "Chi ha dato X a Y?"
# ---------------------------------------------------------------------------

def _coppie_dare_avvenute(storia: Storia) -> dict[tuple[str, str], str]:
    """(oggetto, destinatario) -> agente, per ogni evento "dare" della storia."""
    return {(e.oggetto, e.destinatario): e.agente for e in storia.eventi if e.azione == "dare"}


def _genera_transfer(storia: Storia, rng: random.Random, n: int) -> list[Domanda]:
    stato = storia.stato_finale
    avvenute = _coppie_dare_avvenute(storia)

    oggetti_visti = sorted({e.oggetto for e in storia.eventi if e.oggetto is not None})
    persone = list(stato.persone.keys())

    tutte_le_coppie = [(oid, did) for oid in oggetti_visti for did in persone]
    derivabili = [c for c in tutte_le_coppie if c in avvenute]
    non_derivabili = [c for c in tutte_le_coppie if c not in avvenute]

    domande = []
    for oggetto_id, destinatario_id in _mescola(rng, "transfer", derivabili, non_derivabili, n):
        grafo_domanda = grafo_fatto("dare", obj=oggetto_id, iobj=destinatario_id, quesito="chi")
        agente_id = avvenute.get((oggetto_id, destinatario_id))
        if agente_id is not None:
            risposta = grafo_fatto("dare", nsubj=agente_id, obj=oggetto_id, iobj=destinatario_id)
        else:
            risposta = NON_LO_SO
        domande.append(Domanda("transfer", grafo_domanda, risposta))
    return domande


# ---------------------------------------------------------------------------
# 5. parentela (catene di 1-4 passi)
# ---------------------------------------------------------------------------

def _genera_parentela(storia: Storia, rng: random.Random, n: int) -> list[Domanda]:
    coppie = parentela.tutte_le_coppie()
    derivabili = [(a, b) for a, b in coppie if parentela.relazione_di(a, b) is not None]
    non_derivabili = [(a, b) for a, b in coppie if parentela.relazione_di(a, b) is None]

    domande = []
    for a, b in _mescola(rng, "parentela", derivabili, non_derivabili, n):
        grafo_domanda = grafo_fatto("essere", nsubj=a, **{"nmod:relativo": b, "quesito": "che-parente"})
        relazione = parentela.relazione_di(a, b)
        if relazione is None:
            risposta = NON_LO_SO
        else:
            risposta = grafo_fatto("essere", nsubj=a, **{"nmod:parentela": relazione, "nmod:relativo": b})
        domande.append(Domanda("parentela", grafo_domanda, risposta))
    return domande


# ---------------------------------------------------------------------------
# 6. deduzione multi-hop: "Dove si trova l'oggetto che X ha dato a Y?"
# ---------------------------------------------------------------------------

def _genera_deduzione(storia: Storia, rng: random.Random, n: int) -> list[Domanda]:
    stato = storia.stato_finale
    avvenute = _coppie_dare_avvenute(storia)  # (oggetto, destinatario) -> agente
    oggetti_visti = sorted({e.oggetto for e in storia.eventi if e.oggetto is not None})
    persone = list(stato.persone.keys())

    tutte_le_coppie = [(oid, did) for oid in oggetti_visti for did in persone]
    derivabili = [c for c in tutte_le_coppie if c in avvenute and c[0] in stato.oggetti]
    non_derivabili = [c for c in tutte_le_coppie if c not in avvenute]

    domande = []
    for oggetto_id, destinatario_id in _mescola(rng, "deduzione", derivabili, non_derivabili, n):
        agente_id = avvenute.get((oggetto_id, destinatario_id), "qualcuno")
        grafo_domanda = grafo_fatto(
            "trovarsi", **{"nmod:agente": agente_id, "nmod:oggetto": oggetto_id,
                           "nmod:destinatario": destinatario_id, "quesito": "dove"},
        )
        if (oggetto_id, destinatario_id) in avvenute and oggetto_id in stato.oggetti:
            luogo = stato.luogo_effettivo(oggetto_id)
            risposta = grafo_fatto("essere", nsubj=oggetto_id, **{"obl:luogo": luogo})
        else:
            risposta = NON_LO_SO
        domande.append(Domanda("deduzione", grafo_domanda, risposta))
    return domande


# ---------------------------------------------------------------------------
# 7. causa/energia: "Perché X dorme?" e "Quante mele restano?"
# ---------------------------------------------------------------------------

def _genera_causa(storia: Storia, rng: random.Random, n: int) -> list[Domanda]:
    stato = storia.stato_finale
    persone_addormentate = sorted({e.agente for e in storia.eventi if e.azione == "dormire"})
    persone_mai_addormentate = [pid for pid in stato.persone if pid not in persone_addormentate]

    # Solo la parte "perché dorme" può dare "non lo so" (il conteggio delle
    # risorse è sempre calcolabile dagli eventi): le si lascia più spazio.
    n_causa = max(1, (3 * n) // 4)
    n_risorsa = n - n_causa

    domande = []
    for pid in _mescola(rng, "causa", persone_addormentate, persone_mai_addormentate, n_causa):
        grafo_domanda = grafo_fatto("dormire", nsubj=pid, quesito="perche")
        if pid in persone_addormentate:
            risposta = grafo_fatto("dormire", nsubj=pid, **{"advcl:causa": "stanchezza"})
        else:
            risposta = NON_LO_SO
        domande.append(Domanda("causa", grafo_domanda, risposta))

    fonti = list(dm.RISORSE.keys())
    for fonte in rng.sample(fonti, min(n_risorsa, len(fonti))) if fonti else []:
        info = dm.RISORSE[fonte]
        grafo_domanda = grafo_fatto("restare", nsubj=info["lemma_unita"], quesito="quante")
        quantita = stato.risorse[fonte]
        risposta = grafo_fatto("restare", nsubj=info["lemma_unita"], **{"obl:quantita": str(quantita)})
        domande.append(Domanda("causa", grafo_domanda, risposta))

    return domande


# ---------------------------------------------------------------------------
# Punto di ingresso
# ---------------------------------------------------------------------------

_GENERATORI = (
    _genera_posizione,
    _genera_possesso,
    _genera_conteggio,
    _genera_transfer,
    _genera_parentela,
    _genera_deduzione,
    _genera_causa,
)


def genera_domande(storia: Storia, rng: random.Random, n_per_tipo: int = 6) -> list[Domanda]:
    domande: list[Domanda] = []
    for generatore in _GENERATORI:
        domande.extend(generatore(storia, rng, n_per_tipo))
    return domande
