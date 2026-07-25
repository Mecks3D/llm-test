"""Banco di prova: costruisce lo scenario, fa girare un tracker, raccoglie gli esiti.

Un solo punto in cui si mettono insieme storia, flusso percettivo, domande e
risposte d'oro, così le sette sonde leggono tutte gli stessi `Esito` e i
numeri stanno sulla stessa scala.

Nota sull'onestà della valutazione: le risposte d'oro parlano di `mela_3`, il
tracker conosce solo "quella mela lì". Per interrogarlo si usa
l'allineamento post-hoc di `regole/allineamento.py` — la pratica standard
nella valutazione del tracking. Se il tracker ha fuso due mele in uno slot,
l'allineamento ne assegna uno solo e l'altra resta senza: l'errore di binding
viene penalizzato, non nascosto.
"""
from __future__ import annotations

from dataclasses import dataclass

from mondo import dati_mondo as dm
from mondo.domande import genera_domande
from mondo.simulatore import Storia
from mondo.tipi import StatoMondo
from percezione.sintetica import storia_e_osservazioni
from percezione.tipi import ConfigPercezione, Osservazione
from regole.allineamento import allinea, purezza_binding
from regole.tracker import NESSUNO, NON_LO_SO, TrackerRegole

import random

LUOGHI = tuple(l.id for l in dm.LUOGHI)
PERSONE = frozenset(p.id for p in dm.PERSONE)
CANALI_TUTTI = ("visione", "lingua")


@dataclass(frozen=True)
class Esito:
    """Una domanda, la sua risposta d'oro e quella del sistema, più i tratti
    che servono a spezzare l'accuratezza per sonda."""

    seed: int
    tipo: str  # "posizione" | "possesso"
    entita: str  # id d'istanza reale (mela_3)
    oro: str  # oro NARRATIVO: derivabile dagli eventi raccontati (mondo/domande.py)
    oro_vero: str  # stato reale del mondo, sempre definito
    predetto: str
    confidenza: float
    eta_evidenza: int  # tick dall'ultima volta che la telecamera l'ha visto (P1)
    ambigua: bool  # esistono ≥2 istanze della stessa classe (P2)
    profondita: int  # profondità della catena di contenimento reale (P6)
    cast: int

    @property
    def esatto(self) -> bool:
        """Contro l'oro narrativo. È la scala storica del progetto (`v1`,
        `jepa/`), ma è onesta SOLO per il canale linguistico: quell'oro dice
        "non lo so" quando il fatto non è derivabile dagli eventi raccontati,
        e una telecamera che l'ha visto viene contata in errore."""
        return self.predetto == self.oro

    @property
    def esatto_reale(self) -> bool:
        """Contro lo stato vero del mondo. È la scala giusta per il
        deployment: un sistema che sa di più prende di più, e astenersi conta
        come sbagliare (a premiare l'astensione ben piazzata pensa la sonda P4)."""
        return self.predetto == self.oro_vero

    @property
    def derivabile(self) -> bool:
        return self.oro != NON_LO_SO


@dataclass
class EsitoPredittivo:
    """Sonda P7: quante classi il sistema si aspetta di vedere, e quante ne vede."""

    corrette: int = 0
    totali: int = 0
    corrette_sorpresa: int = 0
    totali_sorpresa: int = 0
    corrette_copia: int = 0  # baseline "non cambia nulla", sul sottoinsieme sorpresa
    totali_copia: int = 0


def _oro(domanda, tipo: str) -> str:
    for n in domanda.grafo_risposta.nodi:
        if n.lemma == "non-lo-so":
            return NON_LO_SO
        if n.lemma == NESSUNO:
            return NESSUNO
        if tipo == "posizione" and n.lemma in LUOGHI:
            return n.lemma
        if tipo == "possesso" and n.lemma in PERSONE:
            return n.lemma
    return NON_LO_SO


def _entita_della_domanda(domanda, stato: StatoMondo) -> str | None:
    for n in domanda.grafo_domanda.nodi:
        if n.lemma in stato.oggetti or n.lemma in stato.persone:
            return n.lemma
    return None


def _oro_vero(stato: StatoMondo, eid: str, tipo: str) -> str:
    """La verità nuda dello stato finale, indipendente da che cosa sia stato
    raccontato o visto."""
    if tipo == "posizione":
        return stato.luogo_effettivo(eid)
    if eid in stato.persone:
        return NESSUNO
    corrente = eid
    for _ in range(9):
        posizione, riferimento = stato.oggetti[corrente].posizione
        if posizione == "persona":
            return riferimento
        if posizione == "luogo":
            return NESSUNO
        corrente = riferimento
    return NESSUNO


def _profondita(stato: StatoMondo, eid: str) -> int:
    if eid in stato.persone:
        return 0
    profondita, corrente = 0, eid
    while True:
        tipo, rif = stato.oggetti[corrente].posizione
        profondita += 1
        if tipo in ("luogo", "persona"):
            return profondita
        corrente = rif


MAI_VISTA = -1


def _eta_evidenza(flusso: tuple[Osservazione, ...], eid: str, t_finale: int) -> int:
    """Tick trascorsi dall'ultima volta che una telecamera ha inquadrato
    l'entità, o `MAI_VISTA` se non è mai comparsa: sono due condizioni
    diverse e la sonda P1 le tiene separate."""
    ultimo = -1
    for oss in flusso:
        if oss.verita and eid in oss.verita:
            ultimo = max(ultimo, oss.t)
    return MAI_VISTA if ultimo < 0 else t_finale - ultimo


def costruisci_tracker(
    storia: Storia,
    flusso: tuple[Osservazione, ...],
    canali: tuple[str, ...] = CANALI_TUTTI,
    assenze_per_dubbio: int = 1,
    politica_assenza: str = "dubita",
    predittivo: EsitoPredittivo | None = None,
) -> TrackerRegole:
    """Fa girare il tracker sui canali richiesti, in ordine temporale.

    Visione e lingua sono interlacciate per tick: al tick t prima si guarda,
    poi si ascolta ciò che è stato raccontato di quel tick.
    """
    tracker = TrackerRegole(
        luoghi=LUOGHI,
        assenze_per_dubbio=assenze_per_dubbio,
        politica_assenza=politica_assenza,
        classi_persona=frozenset(PERSONE),
    )
    per_tick: dict[int, list[Osservazione]] = {}
    for oss in flusso:
        per_tick.setdefault(oss.t, []).append(oss)
    eventi_per_tick: dict[int, list] = {}
    for ev in storia.eventi:
        eventi_per_tick.setdefault(ev.t, []).append(ev)

    precedente: dict[str, tuple[str, ...]] = {}
    t_max = max([*per_tick, *eventi_per_tick], default=0)

    for t in range(1, t_max + 1):
        if "visione" in canali and predittivo is not None:
            # Prima si predicono TUTTE le viste del tick, poi si guarda: se si
            # alternasse, la predizione della cucina userebbe ciò che si è
            # appena visto in salotto allo stesso istante — le telecamere sono
            # simultanee, l'informazione non può viaggiare indietro.
            for oss in per_tick.get(t, []):
                _misura_predizione(tracker, oss, precedente, predittivo)
            for oss in per_tick.get(t, []):
                precedente[oss.vista] = tuple(sorted(oss.classi()))
        if "visione" in canali:
            for oss in per_tick.get(t, []):
                # `oss` porta ancora `verita`: il tracker la registra solo per
                # l'allineamento post-hoc e non la legge mai per decidere
                # (garantito da tests/test_regole.py::test_verita_non_influenza).
                tracker.osserva(oss)
        if "lingua" in canali:
            for ev in eventi_per_tick.get(t, []):
                tracker.ascolta_evento(
                    azione=ev.azione,
                    agente=ev.agente,
                    oggetto=ev.oggetto,
                    luogo=ev.luogo or ev.luogo_origine,
                    destinatario=ev.destinatario,
                    argomento=ev.argomento,
                    testimoni=ev.testimoni,
                    t=t,
                )
    return tracker


def _misura_predizione(
    tracker: TrackerRegole,
    oss: Osservazione,
    precedente: dict[str, tuple[str, ...]],
    acc: EsitoPredittivo,
) -> None:
    """Confronta ciò che il tracker si aspetta con ciò che la vista mostra.

    Il sottoinsieme "sorpresa" sono le classi che cambiano rispetto all'ultima
    volta che si è guardata questa vista: senza restringere lì, la loss è
    banalmente soddisfatta prevedendo "non cambia nulla" (FASE_MENTE.md §6).
    """
    atteso = set(tracker.predici(oss.vista))
    reale = set(oss.classi())
    prima = set(precedente.get(oss.vista, ()))

    universo = atteso | reale
    acc.totali += len(universo) or 1
    acc.corrette += len(atteso & reale) + (0 if universo else 1)

    sorpresa = reale ^ prima
    if sorpresa:
        acc.totali_sorpresa += len(sorpresa)
        acc.corrette_sorpresa += len({c for c in sorpresa if (c in atteso) == (c in reale)})
        acc.totali_copia += len(sorpresa)
        acc.corrette_copia += len({c for c in sorpresa if (c in prima) == (c in reale)})


def valuta_storia(
    seed: int,
    n_tick: int,
    config: ConfigPercezione | None = None,
    canali: tuple[str, ...] = CANALI_TUTTI,
    persone: tuple[dm.Persona, ...] | None = None,
    n_per_tipo: int = 8,
    assenze_per_dubbio: int = 1,
    politica_assenza: str = "dubita",
    predittivo: EsitoPredittivo | None = None,
) -> tuple[list[Esito], float, int]:
    """Esiti di una storia, più purezza di binding e numero di rilevazioni."""
    storia, flusso = storia_e_osservazioni(seed, n_tick, config, persone)
    tracker = costruisci_tracker(
        storia, flusso, canali, assenze_per_dubbio, politica_assenza, predittivo=predittivo
    )
    mappa = allinea(tracker)
    stato = storia.stato_finale
    cast = len(stato.persone)

    conta_classi: dict[str, int] = {}
    for oid, ogg in stato.oggetti.items():
        conta_classi[ogg.lemma] = conta_classi.get(ogg.lemma, 0) + 1

    domande = genera_domande(storia, random.Random(f"domande-{seed}"), n_per_tipo=n_per_tipo)
    esiti: list[Esito] = []
    for d in domande:
        if d.tipo not in ("posizione", "possesso"):
            continue
        eid = _entita_della_domanda(d, stato)
        if eid is None:
            continue
        slot_id = mappa.get(eid)
        slot = tracker.slot[slot_id] if slot_id is not None else None
        risposta = tracker.dove(slot) if d.tipo == "posizione" else tracker.chi_ha(slot)
        classe = eid if eid in stato.persone else stato.oggetti[eid].lemma
        esiti.append(
            Esito(
                seed=seed,
                tipo=d.tipo,
                entita=eid,
                oro=_oro(d, d.tipo),
                oro_vero=_oro_vero(stato, eid, d.tipo),
                predetto=risposta.valore,
                confidenza=risposta.confidenza,
                eta_evidenza=_eta_evidenza(flusso, eid, n_tick),
                ambigua=conta_classi.get(classe, 0) > 1,
                profondita=_profondita(stato, eid),
                cast=cast,
            )
        )
    purezza, n_ril = purezza_binding(tracker)
    return esiti, purezza, n_ril


def lunghezza(seed: int) -> int:
    """Stessa convenzione usata dagli esperimenti precedenti, per confrontabilità."""
    return random.Random(f"lunghezza-{seed}").randint(8, 22)


def valuta_campione(
    seed_base: int = 2000,
    n_storie: int = 40,
    **kwargs,
) -> tuple[list[Esito], float]:
    esiti: list[Esito] = []
    purezze: list[tuple[float, int]] = []
    for i in range(n_storie):
        seed = seed_base + i
        e, purezza, n = valuta_storia(seed, lunghezza(seed), **kwargs)
        esiti += e
        if n:
            purezze.append((purezza, n))
    peso = sum(n for _, n in purezze)
    media = sum(p * n for p, n in purezze) / peso if peso else 0.0
    return esiti, media
