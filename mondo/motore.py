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


def costruisci_stato_iniziale() -> StatoMondo:
    luoghi = {l.id: StatoLuogo(id=l.id, lemma=l.lemma) for l in dm.LUOGHI}
    collegamenti = dm.costruisci_collegamenti()

    persone = {
        p.id: StatoPersona(
            id=p.id, lemma=p.lemma, genere=p.genere, eta=p.eta,
            luogo_preferito=p.luogo_preferito, luogo=dm.LUOGO_INIZIALE[p.id],
        )
        for p in dm.PERSONE
    }

    oggetti: dict[str, StatoOggetto] = {}
    for tipo in dm.OGGETTI_UNICI:
        oid = tipo.lemma
        oggetti[oid] = StatoOggetto(
            id=oid, lemma=tipo.lemma, commestibile=tipo.commestibile,
            contenitore=tipo.contenitore, apribile=tipo.apribile, fisso=tipo.fisso,
            aperto=dm.APERTO_INIZIALE.get(oid, True),
            posizione=("luogo", dm.LUOGO_INIZIALE_OGGETTO[oid]),
            proprietario=dm.PROPRIETARIO_INIZIALE.get(oid),
        )

    risorse = {fonte: info["quantita_iniziale"] for fonte, info in dm.RISORSE.items()}

    return StatoMondo(t=0, luoghi=luoghi, collegamenti=collegamenti, persone=persone,
                       oggetti=oggetti, risorse=risorse)


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
    "dare": 6.0,
    "aprire": 6.0,
    "chiudere": 6.0,
    "mettere_dentro": 9.0,
    "tirare_fuori": 15.0,
}


def _porta_legna(stato: StatoMondo, agente: str) -> bool:
    return any(stato.oggetti[oid].lemma == "legna" for oid in stato.oggetti_portati_da(agente))


def _peso_bias(stato: StatoMondo, agente: str, azione: Azione, parametri: dict) -> float:
    persona = stato.persone[agente]
    peso = _PESO_PER_AZIONE.get(azione.nome, 1.0)
    if azione.nome == "andare" and _porta_legna(stato, agente):
        origine = persona.luogo
        salto = _PROSSIMO_SALTO_VERSO_SALOTTO.get(origine)
        peso *= 25.0 if parametri["luogo_destinazione"] == salto else 0.15
    elif azione.nome == "mettere_dentro" and parametri["contenitore"] == "camino":
        peso *= 8.0
    elif azione.nome == "prendere" and parametri.get("fonte") == "bosco_legna":
        peso *= 3.0
    elif azione.nome == "andare" and persona.luogo_preferito is not None:
        peso *= 4.0 if parametri["luogo_destinazione"] == persona.luogo_preferito else 0.7
    elif azione.nome == "dormire":
        peso *= 1 + persona.stanchezza / 3
    elif azione.nome == "mangiare":
        peso *= 1 + persona.fame / 3
    return peso


def scegli_azione(stato: StatoMondo, agente: str, rng: random.Random) -> tuple[Azione, dict]:
    persona = stato.persone[agente]

    if persona.addormentato:
        return AZIONI["svegliarsi"], {"agente": agente}

    soglia_esausto = dm.SOGLIA_ESAUSTO_PER_ETA[persona.eta]
    if persona.stanchezza >= soglia_esausto:
        candidati = istanze_valide(AZIONI["dormire"], stato, agente)
        if candidati:
            return AZIONI["dormire"], candidati[0]

    if persona.fame >= soglia_esausto:
        candidati = istanze_valide(AZIONI["mangiare"], stato, agente)
        if candidati:
            return AZIONI["mangiare"], rng.choice(candidati)

    pool: list[tuple[Azione, dict]] = []
    pesi: list[float] = []
    for azione in AZIONI.values():
        for parametri in istanze_valide(azione, stato, agente):
            pool.append((azione, parametri))
            pesi.append(_peso_bias(stato, agente, azione, parametri))

    azione, parametri = rng.choices(pool, weights=pesi, k=1)[0]
    return azione, parametri


def _aggiorna_fisiologia(stato: StatoMondo) -> None:
    for persona in stato.persone.values():
        persona.fame = min(dm.SOGLIA_MASSIMA, persona.fame + 1)
        if not persona.addormentato:
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


def avanza_tick(stato: StatoMondo, rng: random.Random, t: int) -> list[Evento]:
    eventi: list[Evento] = []
    for persona in dm.PERSONE:  # ordine fisso -> riproducibilità
        azione, parametri = scegli_azione(stato, persona.id, rng)
        eventi.append(azione.effetti(stato, parametri, t))

    evento_camino = _aggiorna_camino(stato, t)
    if evento_camino is not None:
        eventi.append(evento_camino)

    _aggiorna_fisiologia(stato)
    stato.t = t
    return eventi
