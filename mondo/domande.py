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
from .grafo import NON_LO_SO, Grafo, evento_a_grafo, grafo_a_dict, grafo_fatto
from .numeri import lemma_numero
from .simulatore import Storia
from .tipi import Evento

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


# ---------------------------------------------------------------------------
# Esperimento "tempo" (fasi/FASE2_PIANO_TEMPO.md §2): un solo personaggio per
# storia, domande condizionate nel tempo o nel luogo. Estensione additiva:
# non tocca nulla di quanto sopra (genera_domande, _GENERATORI,
# QUOTA_NON_LO_SO_PER_TIPO restano byte-identici, decisione 10 del piano).
#
# Nota epistemica (come per "parentela"): con cast 1 ogni tick da sveglio del
# protagonista è narrato, quindi "posizione_tempo"/"azione_tempo" producono
# non-lo-so solo nel raro caso di inizio-storia-nel-sonno, e "azione_luogo"
# non lo produce mai per costruzione (vedi _genera_azione_luogo). Le domande
# non-lo-so del mix v1_tempo arrivano dal tipo "posizione" esistente.
#
# Nota su "azione_tempo"/"azione_luogo" e i prelievi da risorsa (melo/pozzo/
# bosco_legna): mai candidati (vedi _e_prelievo_risorsa) — vero bivio non
# previsto dal piano, deciso con Andrea il 2026-07-11 durante T2 (round-trip
# lingua/ altrimenti non invertibile per questi eventi, cfr. commit).
# ---------------------------------------------------------------------------

def _protagonista(storia: Storia) -> str:
    persone = list(storia.stato_finale.persone.keys())
    assert len(persone) == 1, f"genera_domande_tempo richiede un cast di una sola persona, trovate {len(persone)}"
    return persone[0]


def _posizione_al_tick(storia: Storia, pid: str, t: int) -> str | None:
    """Luogo dell'ultimo evento di `pid` con `e.t <= t` (None solo a inizio
    storia, prima di qualunque evento localizzante — cfr. §2.1 del piano)."""
    ultimo: str | None = None
    for e in storia.eventi:
        if e.agente == pid and e.t <= t and e.luogo is not None:
            ultimo = e.luogo
    return ultimo


def _evento_al_tick(storia: Storia, pid: str, t: int) -> Evento | None:
    for e in storia.eventi:
        if e.agente == pid and e.t == t:
            return e
    return None


def _grafo_evento_senza_tempo(evento: Evento) -> Grafo:
    """Stesso grafo di `evento_a_grafo`, senza il nodo/arco `obl:tempo`
    finale (non modifica `evento_a_grafo`: `obl:tempo` è sempre l'ultimo
    nodo/arco che aggiunge, quindi troncarli produce lo stesso ordine)."""
    g = evento_a_grafo(evento)
    return Grafo(nodi=g.nodi[:-1], archi=g.archi[:-1])


def _e_prelievo_risorsa(evento: Evento) -> bool:
    """Vero se `evento` è un "prendere" da una fonte finita (melo/pozzo/
    bosco_legna). Questi eventi non sono candidati per "azione_tempo"/
    "azione_luogo" (decisione di Andrea, 2026-07-11, vero bivio non previsto
    dal piano): lo stampo di superficie per questi prelievi (lingua/stampi.py,
    "raccoglie una mela dal melo") è deliberatamente indefinito — non
    menziona MAI l'istanza raccolta (mela_1 vs mela_2...) perché nella
    narrazione ordinaria l'indice si recupera dall'ordine di lettura. Una
    risposta isolata e fuori ordine come queste due non porta quell'ordine:
    il testo sarebbe identico per istanze diverse, quindi non invertibile."""
    return evento.azione == "prendere" and evento.argomento in dm.RISORSE


def _genera_posizione_tempo(storia: Storia, rng: random.Random, n: int, n_tick: int) -> list[Domanda]:
    pid = _protagonista(storia)
    candidati = list(range(1, n_tick + 1))
    scelti = rng.sample(candidati, min(n, len(candidati)))
    domande = []
    for t in scelti:
        grafo_domanda = grafo_fatto("trovarsi", nsubj=pid, **{"obl:tempo": lemma_numero(t)}, quesito="dove")
        luogo = _posizione_al_tick(storia, pid, t)
        if luogo is None:
            risposta = NON_LO_SO
        else:
            risposta = grafo_fatto("essere", nsubj=pid, **{"obl:luogo": luogo, "obl:tempo": lemma_numero(t)})
        domande.append(Domanda("posizione_tempo", grafo_domanda, risposta))
    return domande


def _genera_azione_tempo(storia: Storia, rng: random.Random, n: int, n_tick: int) -> list[Domanda]:
    pid = _protagonista(storia)
    candidati = []
    for t in range(1, n_tick + 1):
        evento_t = _evento_al_tick(storia, pid, t)
        if evento_t is not None and _e_prelievo_risorsa(evento_t):
            continue  # ambiguo in lingua/: vedi nota di _e_prelievo_risorsa
        candidati.append(t)
    scelti = rng.sample(candidati, min(n, len(candidati)))
    domande = []
    for t in scelti:
        grafo_domanda = grafo_fatto("fare", nsubj=pid, **{"obl:tempo": lemma_numero(t)}, quesito="che-cosa")
        evento = _evento_al_tick(storia, pid, t)
        if evento is not None:
            risposta = evento_a_grafo(evento)
        else:
            precedenti = [e for e in storia.eventi if e.agente == pid and e.t < t]
            if not precedenti:
                risposta = NON_LO_SO
            elif precedenti[-1].azione == "dormire":
                risposta = grafo_fatto("dormire", nsubj=pid, **{"obl:tempo": lemma_numero(t)})
            else:
                raise ValueError(
                    f"tick {t} senza evento per {pid!r} ma l'ultimo evento precedente non è 'dormire': "
                    f"{precedenti[-1]!r}"
                )
        domande.append(Domanda("azione_tempo", grafo_domanda, risposta))
    return domande


def _genera_azione_luogo(storia: Storia, rng: random.Random, n: int) -> list[Domanda]:
    pid = _protagonista(storia)
    per_luogo: dict[str, list[Evento]] = {}
    for e in storia.eventi:
        if e.agente == pid and e.luogo is not None:
            per_luogo.setdefault(e.luogo, []).append(e)

    candidati: list[str] = []
    grafo_per_luogo: dict[str, Grafo] = {}
    for luogo, eventi_luogo in per_luogo.items():
        if _e_prelievo_risorsa(eventi_luogo[0]):
            continue  # ambiguo in lingua/: vedi nota di _e_prelievo_risorsa
        grafi = [_grafo_evento_senza_tempo(e) for e in eventi_luogo]
        if all(g == grafi[0] for g in grafi):
            candidati.append(luogo)
            grafo_per_luogo[luogo] = grafi[0]

    domande = []
    for luogo in rng.sample(candidati, min(n, len(candidati))):
        grafo_domanda = grafo_fatto("fare", nsubj=pid, **{"obl:luogo": luogo}, quesito="che-cosa")
        domande.append(Domanda("azione_luogo", grafo_domanda, grafo_per_luogo[luogo]))
    return domande


def genera_domande_tempo(storia: Storia, rng: random.Random, n_per_tipo: int, n_tick: int) -> list[Domanda]:
    domande: list[Domanda] = []
    domande.extend(_genera_posizione_tempo(storia, rng, n_per_tipo, n_tick))
    domande.extend(_genera_azione_tempo(storia, rng, n_per_tipo, n_tick))
    domande.extend(_genera_azione_luogo(storia, rng, n_per_tipo))
    return domande
