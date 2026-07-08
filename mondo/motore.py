"""Motore di simulazione: stato iniziale, politica dei personaggi, fisica del
tick (fame/stanchezza, camino). Nessun testo italiano oltre ai lemmi già
usati come identificatori in dati_mondo.py.
"""
from __future__ import annotations

import random
from typing import Optional

from . import dati_mondo as dm
from .azioni import AZIONI, Azione, istanze_valide
from .tipi import Evento, StatoLuogo, StatoMondo, StatoOggetto, StatoPersona


def _prossimo_salto_verso(destinazione: str) -> dict[str, str]:
    """BFS sul grafo dei luoghi: per ogni luogo, il vicino da cui passare
    per raggiungere `destinazione` per il cammino più breve."""
    collegamenti = dm.costruisci_collegamenti()
    prossimo: dict[str, str] = {}
    visitati = {destinazione}
    frontiera = [destinazione]
    while frontiera:
        nuova_frontiera = []
        for luogo in frontiera:
            for vicino in collegamenti[luogo]:
                if vicino not in visitati:
                    visitati.add(vicino)
                    prossimo[vicino] = luogo
                    nuova_frontiera.append(vicino)
        frontiera = nuova_frontiera
    return prossimo


# La legna raccolta nel bosco serve a qualcosa solo se arriva fino al
# camino: senza una spinta esplicita verso il salotto, un personaggio che
# vaga a caso quasi non ce la porta mai, e "bruciare" (fuoco che consuma
# legna e scalda) non si osserverebbe quasi mai — è la prima catena
# causa-effetto del curriculum (PROGETTO.md), vale la pena renderla
# raggiungibile.
_PROSSIMO_SALTO_VERSO_SALOTTO = _prossimo_salto_verso("salotto")


def costruisci_stato_iniziale(
    rng: random.Random, persone_cast: tuple[dm.Persona, ...] = dm.PERSONE,
) -> StatoMondo:
    """Stato iniziale estratto per seed (FASE0.md, "stato iniziale ignoto"):
    posizioni di persone e oggetti, contenimento, aperto/chiuso e quantità
    delle risorse sono fatti contingenti, diversi da storia a storia. Solo
    la struttura (luoghi, famiglia, arredi, regole) è fissa. L'ordine delle
    estrazioni segue l'ordine dei dati in dati_mondo.py -> determinismo.

    `persone_cast` di default è il cast pieno (`dm.PERSONE`): un cast ridotto
    (curriculum a difficoltà crescente) è un sottoinsieme esplicito, mai
    scelto qui dentro."""
    luoghi = {l.id: StatoLuogo(id=l.id, lemma=l.lemma) for l in dm.LUOGHI}
    collegamenti = dm.costruisci_collegamenti()
    ids_luoghi = [l.id for l in dm.LUOGHI]

    persone = {
        p.id: StatoPersona(
            id=p.id, lemma=p.lemma, genere=p.genere, eta=p.eta,
            luogo_preferito=p.luogo_preferito, luogo=rng.choice(ids_luoghi),
        )
        for p in persone_cast
    }

    oggetti: dict[str, StatoOggetto] = {}
    # Prima i contenitori e gli arredi (devono esistere per poterci mettere
    # dentro gli altri oggetti), poi il resto.
    per_tipo = sorted(dm.OGGETTI_UNICI, key=lambda tipo: not (tipo.contenitore or tipo.fisso))
    for tipo in per_tipo:
        oid = tipo.lemma
        if tipo.fisso:
            posizione = ("luogo", dm.LUOGO_ARREDO[oid])
        elif not tipo.contenitore and rng.random() < dm.PROB_INIZIO_IN_CONTENITORE:
            # dentro un contenitore mobile (mai negli arredi: il camino
            # brucerebbe subito il contenuto); mai contenitore-in-contenitore.
            mobili = sorted(c for c, o in oggetti.items() if o.contenitore and not o.fisso)
            posizione = ("contenitore", rng.choice(mobili))
        else:
            posizione = ("luogo", rng.choice(ids_luoghi))
        oggetti[oid] = StatoOggetto(
            id=oid, lemma=tipo.lemma, commestibile=tipo.commestibile,
            contenitore=tipo.contenitore, apribile=tipo.apribile, fisso=tipo.fisso,
            aperto=rng.random() < 0.5 if tipo.apribile else True,
            posizione=posizione,
        )

    risorse = {
        fonte: rng.randint(info["quantita_min"], info["quantita_max"])
        for fonte, info in dm.RISORSE.items()
    }

    return StatoMondo(t=0, luoghi=luoghi, collegamenti=collegamenti, persone=persone,
                       oggetti=oggetti, risorse=risorse, risorse_iniziali=dict(risorse))


#  "cercare" e "guardare" hanno tipicamente molte più istanze candidate di
#  qualsiasi altra azione (quasi ogni oggetto non nella stanza è un bersaglio
#  papabile), quindi senza un contrappeso finirebbero per dominare la scelta
#  pesata e affossare la copertura delle azioni di interazione (dare,
#  aprire/chiudere, mettere_dentro/tirare_fuori), che hanno invece poche
#  istanze valide alla volta. I pesi qui sotto correggono lo squilibrio.
_PESO_PER_AZIONE = {
    "cercare": 0.08,
    "guardare": 0.25,
    "dire": 0.4,
    "dormire": 0.1,  # il pisolino volontario resta possibile ma raro (verificato
                     # empiricamente: ~1 sonno volontario ogni 10 forzati su
                     # 2000 semi): la maggior parte dei sonni deve avere causa
                     # determinata (soglia di esaustione)
    "dare": 6.0,
    "aprire": 6.0,
    "chiudere": 6.0,
    "mettere_dentro": 9.0,
    "tirare_fuori": 15.0,
}


def _porta_legna(stato: StatoMondo, agente: str) -> bool:
    return any(stato.oggetti[oid].lemma == "legna" for oid in stato.oggetti_portati_da(agente))


def scegli_azione(stato: StatoMondo, agente: str, rng: random.Random) -> Optional[tuple[Azione, dict]]:
    """Sceglie l'azione del tick per `agente`; None = continua a dormire
    (nessun evento: il sonno dura finché la stanchezza non è recuperata)."""
    persona = stato.persone[agente]

    if persona.addormentato:
        if persona.stanchezza <= 0:
            return AZIONI["svegliarsi"], {"agente": agente}
        return None

    soglia_esausto = dm.SOGLIA_ESAUSTO_PER_ETA[persona.eta]
    if persona.stanchezza >= soglia_esausto:
        candidati = istanze_valide(AZIONI["dormire"], stato, agente)
        if candidati:
            return AZIONI["dormire"], candidati[0]

    if persona.fame >= soglia_esausto:
        candidati = istanze_valide(AZIONI["mangiare"], stato, agente)
        if candidati:
            return AZIONI["mangiare"], rng.choice(candidati)

    # Contesto per i bias, calcolato una volta per agente e non per candidato
    # (è il punto caldo della generazione: vedi criterio di prestazione).
    porta_legna = _porta_legna(stato, agente)
    salto_salotto = _PROSSIMO_SALTO_VERSO_SALOTTO.get(persona.luogo)

    pool: list[tuple[Azione, dict]] = []
    pesi: list[float] = []
    for azione in AZIONI.values():
        nome = azione.nome
        peso_base = _PESO_PER_AZIONE.get(nome, 1.0)
        # niente istanze_valide qui: i candidati sono validi per contratto
        # (vedi azioni.py), riverificare le precondizioni per ognuno è il
        # punto caldo della generazione
        for parametri in azione.genera_candidati(stato, agente):
            peso = peso_base
            if nome == "andare":
                if porta_legna:
                    peso *= 25.0 if parametri["luogo_destinazione"] == salto_salotto else 0.15
                elif persona.luogo_preferito is not None:
                    peso *= 4.0 if parametri["luogo_destinazione"] == persona.luogo_preferito else 0.7
            elif nome == "mettere_dentro" and parametri["contenitore"] == "camino":
                peso *= 8.0
            elif nome == "prendere" and parametri.get("fonte") == "bosco_legna":
                peso *= 3.0
            elif nome == "dormire":
                peso *= 1 + persona.stanchezza / 3
            elif nome == "mangiare":
                peso *= 1 + persona.fame / 3
            pool.append((azione, parametri))
            pesi.append(peso)

    azione, parametri = rng.choices(pool, weights=pesi, k=1)[0]
    return azione, parametri


def _aggiorna_fisiologia(stato: StatoMondo) -> None:
    for persona in stato.persone.values():
        persona.fame = min(dm.SOGLIA_MASSIMA, persona.fame + 1)
        if persona.addormentato:
            persona.stanchezza = max(0, persona.stanchezza - dm.RECUPERO_STANCHEZZA_PER_TICK)
        else:
            persona.stanchezza = min(dm.SOGLIA_MASSIMA, persona.stanchezza + 1)


def _aggiorna_camino(stato: StatoMondo, t: int) -> Optional[Evento]:
    camino = stato.oggetti["camino"]
    legna_dentro = sorted(
        oid for oid in stato.oggetti_dentro(camino.id) if stato.oggetti[oid].lemma == "legna"
    )
    salotto = stato.luoghi["salotto"]
    if legna_dentro:
        bruciata = legna_dentro[0]
        del stato.oggetti[bruciata]
        salotto.calore = min(dm.SOGLIA_MASSIMA, salotto.calore + 1)
        return Evento(t=t, azione="bruciare", agente="camino", oggetto=bruciata,
                      luogo="salotto", testimoni=stato.testimoni_in("salotto"))
    salotto.calore = max(0, salotto.calore - 1)
    return None


def avanza_tick(
    stato: StatoMondo, rng: random.Random, t: int,
    persone_cast: tuple[dm.Persona, ...] = dm.PERSONE,
) -> list[Evento]:
    """`persone_cast` di default è il cast pieno; deve combaciare con quello
    usato in `costruisci_stato_iniziale` per la stessa storia (altrimenti
    `stato.persone` non conterrebbe l'agente)."""
    eventi: list[Evento] = []
    for persona in persone_cast:  # ordine fisso -> riproducibilità
        scelta = scegli_azione(stato, persona.id, rng)
        if scelta is None:  # sta dormendo: nessun evento questo tick
            continue
        azione, parametri = scelta
        eventi.append(azione.effetti(stato, parametri, t))

    evento_camino = _aggiorna_camino(stato, t)
    if evento_camino is not None:
        eventi.append(evento_camino)

    _aggiorna_fisiologia(stato)
    stato.t = t
    return eventi
