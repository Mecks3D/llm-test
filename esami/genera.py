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
from pathlib import Path
from typing import Any

import yaml

from mondo import dati_mondo as dm
from mondo.domande import genera_domande
from mondo.generatore import _lunghezza_storia
from mondo.grafo import evento_a_grafo
from mondo.simulatore import genera_storia

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


def genera_record(stadio: int, seed: int, config: dict) -> dict:
    """Genera il record JSONL per una storia (in memoria, non scrive nulla)."""
    tipi_ammessi = set(_config_stadio(stadio, config)["tipi"])
    n_per_tipo = config["dataset"]["n_per_tipo"]
    ctx = config["dataset"]["ctx"]

    storia = genera_storia(
        seed=seed, n_tick=_n_tick(stadio, seed, config), persone=_cast_persone(config),
    )
    token_eventi = [grafo_a_token(evento_a_grafo(e)) for e in storia.eventi]
    storia_flat = [t for tok in token_eventi for t in tok]

    rng_domande = random.Random(f"domande-{seed}")
    esempi: list[dict] = []
    for d in genera_domande(storia, rng_domande, n_per_tipo=n_per_tipo):
        if d.tipo not in tipi_ammessi:
            continue
        tok_domanda = grafo_a_token(d.grafo_domanda)
        tok_risposta = grafo_a_token(d.grafo_risposta)

        composto = componi_esempio(token_eventi, tok_domanda, tok_risposta)
        if len(composto) > ctx:
            raise ValueError(
                f"stadio {stadio} seed {seed} tipo {d.tipo!r}: la sequenza composta "
                f"({len(composto)} token) supera ctx={ctx}"
            )
        esempi.append({"tipo": d.tipo, "domanda": tok_domanda, "risposta": tok_risposta})

    return {"stadio": stadio, "seed": seed, "storia": storia_flat, "esempi": esempi}


def genera_dataset(stadio: int, split: str, config: dict) -> list[dict]:
    """Genera tutti i record di uno stadio/split. Rifiuta (ValueError, nessun
    file scritto) se un seed della finestra viola il vincolo del proprio split."""
    record = []
    for seed in finestra_seed(stadio, split, config):
        _verifica_seed(seed, split)
        record.append(genera_record(stadio, seed, config))
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
