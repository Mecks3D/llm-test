"""Generazione dei dataset per stadio/split (FASE2_PIANO.md §5).

Unico punto d'ingresso ammesso per scrivere i dataset del curriculum
(decisione 8 del piano): non si chiama `mondo.generatore` direttamente
altrove. Rifiuta seed che violano le finestre normative (train/dev <
1.000.000, esame >= 1.000.000) e sequenze composte più lunghe del ctx
del config — fallendo rumorosamente, mai troncando.

Stdlib puro: niente torch (la conversione in id avviene al caricamento,
in `cervello/dati.py`).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from mondo import dati_mondo as dm
from mondo.domande import Domanda, genera_domande, genera_domande_tempo
from mondo.generatore import _lunghezza_storia
from mondo.grafo import NON_LO_SO, Grafo, evento_a_grafo
from mondo.numeri import VALORE_A_LEMMA, lemma_numero
from mondo.simulatore import Storia, genera_storia

from cervello.sequenza import blocco_stato_a_token, componi_esempio, grafo_a_token

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PERCORSO_CONFIG_DEFAULT = PROJECT_ROOT / "configs" / "v1.yaml"

SPLIT_VALIDI = ("train", "dev", "esame")

# I tre tipi dell'esperimento "tempo" (fasi/FASE2_PIANO_TEMPO.md §2): mai nel
# curriculum ufficiale a meno che uno stadio non li elenchi esplicitamente.
TIPI_TEMPO: frozenset[str] = frozenset({"posizione_tempo", "azione_tempo", "azione_luogo"})


def carica_config(percorso: str | Path = _PERCORSO_CONFIG_DEFAULT) -> dict[str, Any]:
    with open(percorso, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _config_stadio(stadio: int, config: dict) -> dict:
    if stadio not in config["stadi"]:
        raise ValueError(
            f"stadio {stadio} non definito in configs/*.yaml (stadi disponibili: "
            f"{sorted(config['stadi'])})"
        )
    return config["stadi"][stadio]


def finestra_seed(stadio: int, split: str, config: dict) -> range:
    """Finestra di seed normativa (disgiunta per costruzione) per stadio/split."""
    if split not in SPLIT_VALIDI:
        raise ValueError(f"split sconosciuto: {split!r} (attesi {SPLIT_VALIDI})")
    ds = config["dataset"]
    if split == "train":
        inizio, n = 100_000 * (stadio - 1), ds["train_storie"]
    elif split == "dev":
        inizio, n = 800_000 + 10_000 * (stadio - 1), ds["dev_storie"]
    else:  # esame
        inizio, n = 1_000_000 + 10_000 * (stadio - 1), ds["esame_storie"]
    return range(inizio, inizio + n)


def _verifica_seed(seed: int, split: str) -> None:
    if split == "esame":
        if seed < 1_000_000:
            raise ValueError(f"seed d'esame non valido, riservato < 1.000.000: {seed}")
    else:
        if seed >= 1_000_000:
            raise ValueError(
                f"seed di {split} non valido: {seed} è riservato agli esami (>= 1.000.000)"
            )


def _n_tick(stadio: int, seed: int, config: dict) -> int:
    """Lunghezza della storia in tick. `storie_corte` di uno stadio può
    essere `True` (range normativo 3-6, comportamento di sempre, byte
    identico) o un dizionario `{"min": ..., "max": ...}` per un curriculum
    con storie ancora più corte (es. il curriculum a cast crescente,
    `configs/v1_grad*.yaml`)."""
    storie_corte = _config_stadio(stadio, config)["storie_corte"]
    if storie_corte:
        if isinstance(storie_corte, dict):
            minimo, massimo = storie_corte["min"], storie_corte["max"]
        else:
            minimo, massimo = 3, 6
        return random.Random(f"stadio{stadio}-{seed}").randint(minimo, massimo)
    return _lunghezza_storia(seed)


def _cast_persone(config: dict) -> tuple[dm.Persona, ...] | None:
    """Cast ridotto opzionale (`dataset.cast` nel config, elenco di id
    persona): sottoinsieme esplicito di `dm.PERSONE`, con lo stesso ordine.
    Assente -> `None` (cast pieno, comportamento invariato)."""
    id_cast = config["dataset"].get("cast")
    if id_cast is None:
        return None
    id_richiesti = set(id_cast)
    persone = tuple(p for p in dm.PERSONE if p.id in id_richiesti)
    mancanti = id_richiesti - {p.id for p in persone}
    if mancanti:
        raise ValueError(f"dataset.cast contiene id sconosciuti: {sorted(mancanti)}")
    return persone


def _cast_per_seed(config: dict, seed: int) -> tuple[dm.Persona, ...] | None:
    """Cast della storia per questo seed. `dataset.cast_rotante: true`
    (esperimento "tempo", fasi/FASE2_PIANO_TEMPO.md §4.1) dà una sola
    persona a rotazione deterministica (`seed % len(dm.PERSONE)`), diversa
    da `dataset.cast` (fisso per l'intero dataset) — le due chiavi non
    possono coesistere. Assente -> `_cast_persone(config)`, comportamento
    di sempre, byte identico."""
    ds = config["dataset"]
    cast_rotante = ds.get("cast_rotante", False)
    if cast_rotante and ds.get("cast") is not None:
        raise ValueError("dataset.cast e dataset.cast_rotante non possono essere entrambi presenti")
    if cast_rotante:
        return (dm.PERSONE[seed % len(dm.PERSONE)],)
    return _cast_persone(config)


def _lemma_per_relazione(grafo: Grafo, relazione: str) -> str:
    for arco in grafo.archi:
        if arco.relazione == relazione:
            return grafo.nodi[arco.dipendente].lemma
    raise ValueError(f"nessun arco con relazione {relazione!r} nel grafo")


def _d1_d2_d3(storia: Storia, domanda: Domanda) -> tuple[bool, bool, bool] | None:
    """Le tre proprietà anti-scorciatoia per una domanda "posizione"
    (fasi/FASE2_PIANO_ANTISCORCIATOIA.md §2.1): D1 oro diverso dal luogo
    più frequente della storia, D2 distanza dalla coda >= 3 eventi, D3 oro
    diverso dal luogo dell'evento di ultima menzione. `None` se la
    risposta è "non-lo-so" (l'oro non esiste, le proprietà non si
    applicano). Le proprietà si calcolano sugli eventi della storia del
    record (quella eventualmente troncata, se `troncamenti` è attivo —
    vedi §3)."""
    if domanda.grafo_risposta == NON_LO_SO:
        return None

    bersaglio = _lemma_per_relazione(domanda.grafo_domanda, "nsubj")
    oro = _lemma_per_relazione(domanda.grafo_risposta, "obl:luogo")
    eventi = storia.eventi
    e_persona = bersaglio in storia.stato_finale.persone

    if e_persona:
        indici = [i for i, e in enumerate(eventi) if e.agente == bersaglio or e.destinatario == bersaglio]
    else:
        indici = [
            i for i, e in enumerate(eventi)
            if e.azione != "cercare" and (e.oggetto == bersaglio or e.argomento == bersaglio)
        ]

    ultima_menzione = indici[-1]
    distanza_coda = len(eventi) - 1 - ultima_menzione
    luoghi = [e.luogo for e in eventi if e.luogo is not None]
    piu_frequente = Counter(luoghi).most_common(1)[0][0]

    d1 = oro != piu_frequente
    d2 = distanza_coda >= 3
    d3 = oro != eventi[ultima_menzione].luogo
    return d1, d2, d3


def _classifica_domanda_posizione(storia: Storia, domanda: Domanda) -> str:
    """difficile/facile/non-lo-so per una domanda "posizione": "difficile"
    se ALMENO UNA delle proprietà D1/D2/D3 vale (selezione anti-scorciatoia,
    fasi/FASE2_PIANO_ANTISCORCIATOIA.md §2.1). Vedi `_tracking_puro` per la
    congiunzione delle tre (fasi/FASE2_PIANO_DIAGNOSI.md, A3)."""
    proprieta = _d1_d2_d3(storia, domanda)
    if proprieta is None:
        return "non-lo-so"
    return "difficile" if any(proprieta) else "facile"


def _tracking_puro(storia: Storia, domanda: Domanda) -> bool:
    """Vero se TUTTE le euristiche scorciatoia sbagliano (D1 ∧ D2 ∧ D3,
    fasi/FASE2_PIANO_DIAGNOSI.md §2, A3): il sotto-insieme dove ogni punto
    guadagnato è binding vero, non fortuna. Falso per "non-lo-so" (l'oro
    non esiste, le euristiche non si applicano)."""
    proprieta = _d1_d2_d3(storia, domanda)
    return proprieta is not None and all(proprieta)


_LEMMA_A_VALORE_NUM: dict[str, int] = {lemma: valore for valore, lemma in VALORE_A_LEMMA.items()}


def _tracking_puro_tempo(storia: Storia, pid: str, domanda: Domanda, n_tick: int) -> bool:
    """Analogo di `_tracking_puro` per "posizione_tempo" (esperimento
    "tempo", fasi/FASE2_PIANO_TEMPO.md §4.3): vero se l'oro non è né il
    luogo più frequente della storia né la posizione finale del
    protagonista, e il tick chiesto è lontano dalla coda (`n_tick - t >= 3`
    — né la scorciatoia di frequenza, né lo stato finale, né roba fresca).
    Falso per "non-lo-so" (l'oro non esiste)."""
    if domanda.grafo_risposta == NON_LO_SO:
        return False
    oro = _lemma_per_relazione(domanda.grafo_risposta, "obl:luogo")
    t = _LEMMA_A_VALORE_NUM[_lemma_per_relazione(domanda.grafo_domanda, "obl:tempo")]
    luoghi = [e.luogo for e in storia.eventi if e.luogo is not None]
    piu_frequente = Counter(luoghi).most_common(1)[0][0]
    posizione_finale = storia.stato_finale.luogo_effettivo(pid)
    return oro != piu_frequente and oro != posizione_finale and (n_tick - t) >= 3


def _seleziona_posizione(
    storia: Storia, candidate: list[Domanda], n_per_tipo: int, quota_difficili: float,
    seed: int, n_tick: int,
) -> list[tuple[Domanda, str]]:
    """Selezione anti-scorciatoia (piano §2.2): sovracampiona le domande
    "difficili" mantenendo la quota di non-lo-so (~20% di `n_per_tipo`).
    Ritorna coppie (Domanda, difficolta), già mescolate."""
    gruppi: dict[str, list[Domanda]] = {"non-lo-so": [], "difficile": [], "facile": []}
    for d in candidate:
        gruppi[_classifica_domanda_posizione(storia, d)].append(d)

    rng = random.Random(f"anti-{seed}-{n_tick}")
    n_nls = min(round(0.2 * n_per_tipo), len(gruppi["non-lo-so"]))
    n_diff = min(round(quota_difficili * (n_per_tipo - n_nls)), len(gruppi["difficile"]))
    presi_nls = rng.sample(gruppi["non-lo-so"], n_nls)
    presi_diff = rng.sample(gruppi["difficile"], n_diff)
    scelti: list[tuple[Domanda, str]] = (
        [(d, "non-lo-so") for d in presi_nls] + [(d, "difficile") for d in presi_diff]
    )

    def _completa(etichetta: str, gruppo: list[Domanda]) -> None:
        restanti = n_per_tipo - len(scelti)
        if restanti <= 0:
            return
        gia_scelti = {id(d) for d, _ in scelti}
        candidati_extra = [d for d in gruppo if id(d) not in gia_scelti]
        presi = rng.sample(candidati_extra, min(restanti, len(candidati_extra)))
        scelti.extend((d, etichetta) for d in presi)

    _completa("facile", gruppi["facile"])
    _completa("difficile", gruppi["difficile"])
    _completa("non-lo-so", gruppi["non-lo-so"])

    rng.shuffle(scelti)
    return scelti


def _componi_e_valida(
    stadio: int, seed: int, tipo: str, token_eventi: list[list[str]],
    tok_domanda: list[str], tok_risposta: list[str], ctx: int,
) -> dict:
    composto = componi_esempio(token_eventi, tok_domanda, tok_risposta)
    if len(composto) > ctx:
        raise ValueError(
            f"stadio {stadio} seed {seed} tipo {tipo!r}: la sequenza composta "
            f"({len(composto)} token) supera ctx={ctx}"
        )
    return {"tipo": tipo, "domanda": tok_domanda, "risposta": tok_risposta}


def _blocchi_stato_oro(
    seed: int, n_tick: int, cast: tuple[dm.Persona, ...] | None, storia: Storia,
) -> dict[int, tuple[str, list[tuple[str, str]]]]:
    """Stato-oro per ogni tick con eventi (Fase B, fasi/FASE2_PIANO_STATO.md §4).

    Per ogni tick in cui è successo qualcosa, la posizione EFFETTIVA di ciascuna
    persona del cast a fine tick — inclusa la posizione iniziale di chi non ha
    ancora agito. Fonte: lo stato del simulatore troncato al tick t
    (`stato_finale.luogo_effettivo`), la semantica già verificata dal piano
    tempo (§1.7, riusata da §1.9): rieseguire la storia troncata è il prefisso
    esatto di quella piena (stesso seed, motore deterministico). Ordine
    deterministico = ordine del cast (`dm.PERSONE` o il sottoinsieme del
    config), non l'ordine di menzione.

    Ritorna `{tick: (lemma_ordinale, [(persona, luogo), ...])}`.
    """
    persone_cast = cast if cast is not None else dm.PERSONE
    cast_ids = [p.id for p in persone_cast]
    tick_con_eventi = sorted({e.t for e in storia.eventi})
    blocchi: dict[int, tuple[str, list[tuple[str, str]]]] = {}
    for t in tick_con_eventi:
        stato_t = genera_storia(seed=seed, n_tick=t, persone=cast).stato_finale
        posizioni = [(pid, stato_t.luogo_effettivo(pid)) for pid in cast_ids]
        blocchi[t] = (lemma_numero(t), posizioni)
    return blocchi


def _token_storia_con_stato(
    storia: Storia, blocchi: dict[int, tuple[str, list[tuple[str, str]]]],
) -> list[list[str]]:
    """Sequenza-storia con i blocchi [STATO] interlacciati: per ogni tick, gli
    eventi del tick seguiti dal blocco stato di fine tick. Ritorna una lista di
    token-list (eventi e blocchi come elementi separati), pronta per
    `componi_esempio`/`_componi_e_valida`: concatenandola si ottiene la storia
    interlacciata."""
    token_storia: list[list[str]] = []
    tick_corrente: int | None = None
    for e in storia.eventi:
        if tick_corrente is not None and e.t != tick_corrente:
            token_storia.append(blocco_stato_a_token(*blocchi[tick_corrente]))
        token_storia.append(grafo_a_token(evento_a_grafo(e)))
        tick_corrente = e.t
    if tick_corrente is not None:
        token_storia.append(blocco_stato_a_token(*blocchi[tick_corrente]))
    return token_storia


def genera_record(
    stadio: int, seed: int, config: dict, *, split: str = "train",
    troncamento: int | None = None,
) -> dict:
    """Genera il record JSONL per una storia (in memoria, non scrive nulla).

    `troncamento`, se dato, ferma la storia al tick k invece della lunghezza
    piena (solo per il train, piano fasi/FASE2_PIANO_ANTISCORCIATOIA.md §3):
    il record porta in più il campo `"troncamento": k`. Le chiavi opzionali
    `dataset.anti_scorciatoia` e `dataset.troncamenti` (piano §2/§3/§4) si
    applicano SOLO quando `split == "train"`; senza di esse (o per dev/esame)
    il comportamento è quello di sempre, byte per byte."""
    tipi_ammessi = set(_config_stadio(stadio, config)["tipi"])
    ds = config["dataset"]
    n_per_tipo = ds["n_per_tipo"]
    ctx = ds["ctx"]

    n_tick = troncamento if troncamento is not None else _n_tick(stadio, seed, config)
    cast = _cast_per_seed(config, seed)
    storia = genera_storia(seed=seed, n_tick=n_tick, persone=cast)

    # Fase B (fasi/FASE2_PIANO_STATO.md §4): supervisione densa in-sequenza,
    # SOLO nel train. La storia si interlaccia con un blocco [STATO] a fine di
    # ogni tick con eventi; dev/esame restano la distribuzione ufficiale senza
    # stato (byte-identico quando dataset.stato è assente).
    stato_attivo = split == "train" and ds.get("stato", False)
    if stato_attivo:
        blocchi = _blocchi_stato_oro(seed, n_tick, cast, storia)
        token_eventi = _token_storia_con_stato(storia, blocchi)
    else:
        token_eventi = [grafo_a_token(evento_a_grafo(e)) for e in storia.eventi]
    storia_flat = [t for tok in token_eventi for t in tok]

    anti_cfg = ds.get("anti_scorciatoia") if split == "train" else None
    n_candidate = anti_cfg.get("candidate_per_tipo", 999) if anti_cfg else n_per_tipo

    seme_domande = f"domande-{seed}" if troncamento is None else f"domande-{seed}-t{troncamento}"
    rng_domande = random.Random(seme_domande)
    candidate = genera_domande(storia, rng_domande, n_per_tipo=n_candidate)

    if tipi_ammessi & TIPI_TEMPO:
        # RNG separato da "domande-{seed}" (decisione §1.3 del piano tempo):
        # le estrazioni dei tipi esistenti non cambiano. Percorso normale
        # (filtro tipi_ammessi sotto), MAI la selezione anti-scorciatoia.
        seme_tempo = f"domande-tempo-{seed}" if troncamento is None else f"domande-tempo-{seed}-t{troncamento}"
        candidate += genera_domande_tempo(storia, random.Random(seme_tempo), n_per_tipo=n_candidate, n_tick=n_tick)

    esempi: list[dict] = []
    for d in candidate:
        if d.tipo not in tipi_ammessi:
            continue
        if d.tipo == "posizione" and anti_cfg is not None:
            continue  # gestita sotto dalla selezione anti-scorciatoia dedicata
        tok_domanda = grafo_a_token(d.grafo_domanda)
        tok_risposta = grafo_a_token(d.grafo_risposta)
        esempi.append(_componi_e_valida(stadio, seed, d.tipo, token_eventi, tok_domanda, tok_risposta, ctx))

    if anti_cfg is not None and "posizione" in tipi_ammessi:
        candidate_posizione = [d for d in candidate if d.tipo == "posizione"]
        selezionate = _seleziona_posizione(
            storia, candidate_posizione, n_per_tipo, anti_cfg["quota_difficili"], seed, n_tick,
        )
        for d, difficolta in selezionate:
            tok_domanda = grafo_a_token(d.grafo_domanda)
            tok_risposta = grafo_a_token(d.grafo_risposta)
            esempio = _componi_e_valida(stadio, seed, d.tipo, token_eventi, tok_domanda, tok_risposta, ctx)
            esempio["difficolta"] = difficolta
            esempi.append(esempio)

    record = {"stadio": stadio, "seed": seed, "storia": storia_flat, "esempi": esempi}
    if troncamento is not None:
        record["troncamento"] = troncamento
    return record


def _record_per_seed(stadio: int, seed: int, config: dict, split: str) -> list[dict]:
    """Tutti i record di un seed: il record pieno e, se `dataset.troncamenti`
    è attivo (solo train, piano §3), un record aggiuntivo per ogni tick
    intermedio k in [3, n_tick_pieno)."""
    record = [genera_record(stadio, seed, config, split=split)]
    if split == "train" and config["dataset"].get("troncamenti", False):
        n_tick_pieno = _n_tick(stadio, seed, config)
        for k in range(3, n_tick_pieno):
            record.append(genera_record(stadio, seed, config, split=split, troncamento=k))
    return record


def genera_dataset(stadio: int, split: str, config: dict) -> list[dict]:
    """Genera tutti i record di uno stadio/split. Rifiuta (ValueError, nessun
    file scritto) se un seed della finestra viola il vincolo del proprio split."""
    record = []
    for seed in finestra_seed(stadio, split, config):
        _verifica_seed(seed, split)
        record.extend(_record_per_seed(stadio, seed, config, split))
    return record


def percorso_dataset(stadio: int, split: str, config: dict) -> Path:
    dati_dir = PROJECT_ROOT / config["percorsi"]["dati_dir"]
    return dati_dir / f"stadio{stadio}" / f"{split}.jsonl"


def genera_esame_tracking(stadio: int, config: dict) -> list[dict]:
    """Split d'esame aggiuntivo e permanente (fasi/FASE2_PIANO_DIAGNOSI.md,
    A3): solo domande "posizione" dove TUTTE le euristiche scorciatoia
    sbagliano (`_tracking_puro`, D1 ∧ D2 ∧ D3). Stessi seed e stesse
    domande candidate dell'esame ufficiale (mai train, mai troncamenti):
    ogni domanda qui è già presente in `esame.jsonl`, questo file la
    AFFIANCA (vincolo permanente 2 di FASE2_PIANO_DIAGNOSI.md §1), non la
    sostituisce. Storie senza nessuna domanda "tracking puro" sono escluse
    dal risultato."""
    tipi_ammessi = set(_config_stadio(stadio, config)["tipi"])
    if "posizione" not in tipi_ammessi:
        return []
    ds = config["dataset"]
    ctx = ds["ctx"]

    record: list[dict] = []
    for seed in finestra_seed(stadio, "esame", config):
        _verifica_seed(seed, "esame")
        n_tick = _n_tick(stadio, seed, config)
        storia = genera_storia(seed=seed, n_tick=n_tick, persone=_cast_per_seed(config, seed))
        token_eventi = [grafo_a_token(evento_a_grafo(e)) for e in storia.eventi]

        rng_domande = random.Random(f"domande-{seed}")
        candidate = genera_domande(storia, rng_domande, n_per_tipo=ds["n_per_tipo"])

        esempi = [
            _componi_e_valida(
                stadio, seed, d.tipo, token_eventi,
                grafo_a_token(d.grafo_domanda), grafo_a_token(d.grafo_risposta), ctx,
            )
            for d in candidate if d.tipo == "posizione" and _tracking_puro(storia, d)
        ]
        if esempi:
            storia_flat = [t for tok in token_eventi for t in tok]
            record.append({"stadio": stadio, "seed": seed, "storia": storia_flat, "esempi": esempi})
    return record


def percorso_esame_tracking(stadio: int, config: dict) -> Path:
    dati_dir = PROJECT_ROOT / config["percorsi"]["dati_dir"]
    return dati_dir / f"stadio{stadio}" / "tracking.jsonl"


def scrivi_esame_tracking(stadio: int, config: dict) -> Path:
    record = genera_esame_tracking(stadio, config)
    percorso = percorso_esame_tracking(stadio, config)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    with open(percorso, "w", encoding="utf-8") as f:
        for r in record:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    return percorso


def genera_tracking_tempo(stadio: int, config: dict) -> list[dict]:
    """Split diagnostico aggiuntivo e permanente per l'esperimento "tempo"
    (analogo ad A3/`tracking.jsonl`, fasi/FASE2_PIANO_TEMPO.md §4.3): solo
    domande "posizione_tempo" dove `_tracking_puro_tempo` vale. Stessi seed
    e stesse domande candidate dell'esame ufficiale (mai train): ogni
    domanda qui è già presente in `esame.jsonl`, questo file la AFFIANCA,
    non la sostituisce. Storie senza nessuna domanda "tracking puro tempo"
    sono escluse dal risultato."""
    tipi_ammessi = set(_config_stadio(stadio, config)["tipi"])
    if "posizione_tempo" not in tipi_ammessi:
        return []
    ds = config["dataset"]
    ctx = ds["ctx"]

    record: list[dict] = []
    for seed in finestra_seed(stadio, "esame", config):
        _verifica_seed(seed, "esame")
        n_tick = _n_tick(stadio, seed, config)
        cast = _cast_per_seed(config, seed)
        if cast is None or len(cast) != 1:
            raise ValueError(
                f"tracking_tempo richiede un cast di una sola persona (dataset.cast_rotante: true): seed {seed}"
            )
        pid = cast[0].id
        storia = genera_storia(seed=seed, n_tick=n_tick, persone=cast)
        token_eventi = [grafo_a_token(evento_a_grafo(e)) for e in storia.eventi]

        rng_domande_tempo = random.Random(f"domande-tempo-{seed}")
        candidate = genera_domande_tempo(storia, rng_domande_tempo, n_per_tipo=ds["n_per_tipo"], n_tick=n_tick)

        esempi = [
            _componi_e_valida(
                stadio, seed, d.tipo, token_eventi,
                grafo_a_token(d.grafo_domanda), grafo_a_token(d.grafo_risposta), ctx,
            )
            for d in candidate if d.tipo == "posizione_tempo" and _tracking_puro_tempo(storia, pid, d, n_tick)
        ]
        if esempi:
            storia_flat = [t for tok in token_eventi for t in tok]
            record.append({"stadio": stadio, "seed": seed, "storia": storia_flat, "esempi": esempi})
    return record


def percorso_tracking_tempo(stadio: int, config: dict) -> Path:
    dati_dir = PROJECT_ROOT / config["percorsi"]["dati_dir"]
    return dati_dir / f"stadio{stadio}" / "tracking_tempo.jsonl"


def scrivi_tracking_tempo(stadio: int, config: dict) -> Path:
    record = genera_tracking_tempo(stadio, config)
    percorso = percorso_tracking_tempo(stadio, config)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    with open(percorso, "w", encoding="utf-8") as f:
        for r in record:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    return percorso


def scrivi_dataset(stadio: int, split: str, config: dict) -> Path:
    record = genera_dataset(stadio, split, config)
    percorso = percorso_dataset(stadio, split, config)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    with open(percorso, "w", encoding="utf-8") as f:
        for r in record:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    return percorso


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_PERCORSO_CONFIG_DEFAULT))
    ap.add_argument("--stadio", type=int, required=True)
    ap.add_argument("--split", choices=(*SPLIT_VALIDI, "tracking", "tracking-tempo"), default=None)
    args = ap.parse_args()

    config = carica_config(args.config)
    # "tracking"/"tracking-tempo" vanno SOLO se richiesti esplicitamente: il
    # default senza --split resta train+dev+esame, byte-identico a prima.
    split_da_fare = [args.split] if args.split else list(SPLIT_VALIDI)
    for split in split_da_fare:
        if split == "tracking":
            percorso = scrivi_esame_tracking(args.stadio, config)
        elif split == "tracking-tempo":
            percorso = scrivi_tracking_tempo(args.stadio, config)
        else:
            percorso = scrivi_dataset(args.stadio, split, config)
        n = sum(1 for _ in open(percorso, encoding="utf-8"))
        print(f"stadio {args.stadio} {split}: {n} storie -> {percorso}")


if __name__ == "__main__":
    _cli()
