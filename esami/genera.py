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
from mondo.domande import Domanda, genera_domande
from mondo.generatore import _lunghezza_storia
from mondo.grafo import NON_LO_SO, Grafo, evento_a_grafo
from mondo.simulatore import Storia, genera_storia

from cervello.sequenza import componi_esempio, grafo_a_token

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PERCORSO_CONFIG_DEFAULT = PROJECT_ROOT / "configs" / "v1.yaml"

SPLIT_VALIDI = ("train", "dev", "esame")


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
    if _config_stadio(stadio, config)["storie_corte"]:
        return random.Random(f"stadio{stadio}-{seed}").randint(3, 6)
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


def _lemma_per_relazione(grafo: Grafo, relazione: str) -> str:
    for arco in grafo.archi:
        if arco.relazione == relazione:
            return grafo.nodi[arco.dipendente].lemma
    raise ValueError(f"nessun arco con relazione {relazione!r} nel grafo")


def _classifica_domanda_posizione(storia: Storia, domanda: Domanda) -> str:
    """difficile/facile/non-lo-so per una domanda "posizione", secondo la
    selezione anti-scorciatoia (fasi/FASE2_PIANO_ANTISCORCIATOIA.md §2.1). Le
    proprietà si calcolano sugli eventi della storia del record (quella
    eventualmente troncata, se `troncamenti` è attivo — vedi §3)."""
    if domanda.grafo_risposta == NON_LO_SO:
        return "non-lo-so"

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
    return "difficile" if (d1 or d2 or d3) else "facile"


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


def genera_record(stadio: int, seed: int, config: dict, *, split: str = "train") -> dict:
    """Genera il record JSONL per una storia (in memoria, non scrive nulla).

    La chiave opzionale `dataset.anti_scorciatoia` (piano
    fasi/FASE2_PIANO_ANTISCORCIATOIA.md §2/§4) si applica SOLO quando
    `split == "train"`; senza di essa (o per dev/esame) il comportamento è
    quello di sempre, byte per byte."""
    tipi_ammessi = set(_config_stadio(stadio, config)["tipi"])
    ds = config["dataset"]
    n_per_tipo = ds["n_per_tipo"]
    ctx = ds["ctx"]

    n_tick = _n_tick(stadio, seed, config)
    storia = genera_storia(seed=seed, n_tick=n_tick, persone=_cast_persone(config))
    token_eventi = [grafo_a_token(evento_a_grafo(e)) for e in storia.eventi]
    storia_flat = [t for tok in token_eventi for t in tok]

    anti_cfg = ds.get("anti_scorciatoia") if split == "train" else None
    n_candidate = anti_cfg.get("candidate_per_tipo", 999) if anti_cfg else n_per_tipo

    rng_domande = random.Random(f"domande-{seed}")
    candidate = genera_domande(storia, rng_domande, n_per_tipo=n_candidate)

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

    return {"stadio": stadio, "seed": seed, "storia": storia_flat, "esempi": esempi}


def genera_dataset(stadio: int, split: str, config: dict) -> list[dict]:
    """Genera tutti i record di uno stadio/split. Rifiuta (ValueError, nessun
    file scritto) se un seed della finestra viola il vincolo del proprio split."""
    record = []
    for seed in finestra_seed(stadio, split, config):
        _verifica_seed(seed, split)
        record.append(genera_record(stadio, seed, config, split=split))
    return record


def percorso_dataset(stadio: int, split: str, config: dict) -> Path:
    dati_dir = PROJECT_ROOT / config["percorsi"]["dati_dir"]
    return dati_dir / f"stadio{stadio}" / f"{split}.jsonl"


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
    ap.add_argument("--split", choices=SPLIT_VALIDI, default=None)
    args = ap.parse_args()

    config = carica_config(args.config)
    split_da_fare = [args.split] if args.split else list(SPLIT_VALIDI)
    for split in split_da_fare:
        percorso = scrivi_dataset(args.stadio, split, config)
        n = sum(1 for _ in open(percorso, encoding="utf-8"))
        print(f"stadio {args.stadio} {split}: {n} storie -> {percorso}")


if __name__ == "__main__":
    _cli()
