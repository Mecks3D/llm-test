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

Epistemica (FASE0.md): lo stato INIZIALE è contingente, estratto per seed e
mai rivelato al lettore; un fatto è conoscibile solo se stabilito dagli
eventi. Le regole strutturali del mondo (mappa, famiglia, arredi, "le mani
iniziano vuote") sono invece conoscenza di sfondo, identica in ogni storia.

Nota su "parentela": la famiglia è struttura fissa (conoscenza di sfondo) e
con 6 persone il calcolo in parentela.py copre TUTTE le coppie — quindi
questo tipo non produce istanze "non lo so" per costruzione, non per una
svista (deviazione da FASE0.md punto 8 accettata e documentata lì).
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from . import dati_mondo as dm
from . import parentela
from .grafo import NON_LO_SO, Grafo, grafo_a_dict, grafo_fatto
from .numeri import lemma_numero
from .simulatore import Storia

# Quota di "non lo so" richiesta per tipo (~15-20% indicato da FASE0.md,
# punto 8). Non tutti i tipi hanno lo stesso margine: transfer/deduzione
# hanno un bacino di coppie non-avvenute enorme (quasi ogni combinazione
# oggetto-destinatario non è mai stata data), quindi una quota bassa evita
# di sforare molto oltre il 20%; gli altri tipi hanno bacini più piccoli e
# chiedono una quota più alta per avvicinarsi al target. parentela resta a
# 0% per costruzione (vedi nota nel docstring del modulo) e non ha una
# propria quota da tarare. Le quote effettive si verificano in statistiche.py.
QUOTA_NON_LO_SO_PER_TIPO = {
    "posizione": 0.30,
    "possesso": 0.30,
    "conteggio": 0.30,
    "transfer": 0.12,
    "parentela": 0.18,
    "deduzione": 0.12,
    "causa": 0.30,
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

    # Solo "Chi ha X adesso?" (possesso dinamico). Il possesso statico
    # ("Di chi è X?") non esiste più: con lo stato iniziale estratto per seed
    # la proprietà non è mai rivelata dagli eventi, quindi non sarebbe MAI
    # derivabile — tornerà quando ci sarà un meccanismo di rivelazione
    # (FASE0.md, "stato iniziale ignoto").
    domande: list[Domanda] = []
    derivabili = [oid for oid in oggetti if oid not in mai_localizzati]
    non_derivabili = list(mai_localizzati)
    for oid in _mescola(rng, "possesso", derivabili, non_derivabili, n):
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
    # Conoscenza completa = ogni oggetto è stato localizzato da almeno un
    # evento: solo allora i conteggi "in un posto" sono derivabili, perché
    # un oggetto mai menzionato potrebbe trovarsi proprio lì.
    conoscenza_completa = not mai_localizzati

    # "Quanti oggetti porta X?" — derivabile SEMPRE: le mani iniziano vuote
    # (regola strutturale del mondo) e ogni prendere/dare/posare/mangiare è
    # un evento visibile, quindi il carico di X si ricostruisce per intero.
    bersagli_persona = [("persona", pid) for pid in stato.persone]
    # "Quanti oggetti ci sono in Y?" — luogo o contenitore.
    bersagli_posto = ([("luogo", lid) for lid in stato.luoghi]
                      + [("contenitore", cid) for cid, o in stato.oggetti.items() if o.contenitore])

    derivabili = bersagli_persona + (bersagli_posto if conoscenza_completa else [])
    non_derivabili = [] if conoscenza_completa else bersagli_posto

    domande = []
    for tipo_bersaglio, bid in _mescola(rng, "conteggio", derivabili, non_derivabili, n):
        if tipo_bersaglio == "persona":
            grafo_domanda = grafo_fatto("portare", nsubj=bid, quesito="quanti")
            quantita = len(stato.oggetti_portati_da(bid))
            risposta = grafo_fatto("portare", nsubj=bid, **{"obl:quantita": lemma_numero(quantita)})
        else:
            grafo_domanda = grafo_fatto("esserci", **{"obl:luogo": bid, "quesito": "quanti"})
            if not conoscenza_completa:
                risposta = NON_LO_SO
            else:
                quantita = (len(stato.oggetti_in_luogo(bid)) if tipo_bersaglio == "luogo"
                            else len(stato.oggetti_dentro(bid)))
                risposta = grafo_fatto("esserci", **{"obl:luogo": bid, "obl:quantita": lemma_numero(quantita)})
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
# 7. causa/energia: "Perché X dorme?" e "Quante mele sono state raccolte?"
# ---------------------------------------------------------------------------

def _genera_causa(storia: Storia, rng: random.Random, n: int) -> list[Domanda]:
    # Le cause dei sonni, come registrate negli eventi: "stanchezza" se il
    # sonno era dettato dall'esaustione, None per i pisolini volontari (la
    # cui causa non è un fatto del mondo). Si domanda solo di chi ha dormito,
    # e solo se TUTTI i suoi sonni hanno la stessa causa: la domanda non ha
    # ancora un ancoraggio temporale, con cause miste sarebbe ambigua — e le
    # risposte d'oro non devono mai essere ambigue.
    cause_sonni: dict[str, list] = {}
    for e in storia.eventi:
        if e.azione == "dormire":
            cause_sonni.setdefault(e.agente, []).append(e.argomento)

    derivabili = sorted(pid for pid, cause in cause_sonni.items()
                        if all(c == "stanchezza" for c in cause))
    non_derivabili = sorted(pid for pid, cause in cause_sonni.items()
                            if all(c is None for c in cause))

    n_causa = max(1, (3 * n) // 4)
    n_risorsa = n - n_causa

    domande = []
    for pid in _mescola(rng, "causa", derivabili, non_derivabili, n_causa):
        grafo_domanda = grafo_fatto("dormire", nsubj=pid, quesito="perche")
        if pid in derivabili:
            risposta = grafo_fatto("dormire", nsubj=pid, **{"advcl:causa": "stanchezza"})
        else:
            risposta = NON_LO_SO
        domande.append(Domanda("causa", grafo_domanda, risposta))

    # "Quante X sono state raccolte?" — derivabile per puro conteggio di
    # eventi. ("Quante restano?" non è più una domanda lecita: la quantità
    # iniziale della fonte è un fatto contingente mai rivelato dagli eventi.)
    raccolte = Counter(e.argomento for e in storia.eventi
                       if e.azione == "prendere" and e.argomento is not None)
    fonti = sorted(dm.RISORSE.keys())
    for fonte in rng.sample(fonti, min(n_risorsa, len(fonti))):
        info = dm.RISORSE[fonte]
        grafo_domanda = grafo_fatto("raccogliere", obj=info["lemma_unita"], quesito="quante")
        risposta = grafo_fatto("raccogliere", obj=info["lemma_unita"],
                               **{"obl:quantita": lemma_numero(raccolte.get(fonte, 0))})
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
