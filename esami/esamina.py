"""Esame: decodifica greedy + confronto grafo vs grafo (FASE2_PIANO.md §8).

Regola non negoziabile #4: la valutazione è sempre grafo vs grafo, mai
stringa vs stringa. Una sequenza generata malformata NON fa crashare
l'esame: conta come risposta errata di categoria "malformata".
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mondo.grafo import NON_LO_SO, Grafo

from cervello.modello import ConfigModello, Modello
from cervello.sequenza import DOMANDA, FINE, RISPOSTA, STORIA, token_a_grafo
from cervello.vocabolario import Vocabolario, carica_vocabolario

from .genera import PROJECT_ROOT, carica_config, percorso_dataset

CATEGORIE = ("esatto", "invenzione", "astensione_errata", "malformata", "errore")


def dispositivo(config: dict) -> str:
    d = config.get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def decodifica_greedy(
    modello: Modello, vocab: Vocabolario, ids_prefisso: list[int], ctx: int, device: str,
) -> list[int]:
    """Genera id greedy (argmax) dal prefisso fino a [FINE] o al tetto `ctx`.
    Ritorna SOLO gli id generati (non il prefisso). Deterministica."""
    id_fine = vocab.id(FINE)
    era_training = modello.training
    modello.eval()
    ids = list(ids_prefisso)
    with torch.no_grad():
        while len(ids) < ctx:
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits = modello(x)
            prossimo = int(torch.argmax(logits[0, -1]).item())
            ids.append(prossimo)
            if prossimo == id_fine:
                break
    if era_training:
        modello.train()
    return ids[len(ids_prefisso):]


def _categoria(oro: Grafo, generato: Grafo | None) -> str:
    if generato is None:
        return "malformata"
    if generato == oro:
        return "esatto"
    if oro == NON_LO_SO:
        return "invenzione"
    if generato == NON_LO_SO:
        return "astensione_errata"
    return "errore"


@dataclass(frozen=True)
class EsitoEsempio:
    tipo: str
    categoria: str
    esatto: bool


def valuta_esempio(
    modello: Modello, vocab: Vocabolario, storia_flat: list[str], esempio: dict,
    ctx: int, device: str,
) -> EsitoEsempio:
    """Valuta un esempio: dà [STORIA]...[DOMANDA]...[RISPOSTA] al modello,
    decodifica greedy, e confronta il grafo risultante con quello-verità."""
    prefisso_token = [STORIA, *storia_flat, DOMANDA, *esempio["domanda"], RISPOSTA]
    prefisso_ids = [vocab.id(t) for t in prefisso_token]

    generati_ids = decodifica_greedy(modello, vocab, prefisso_ids, ctx, device)
    generati_token = [vocab.token(i) for i in generati_ids]
    if generati_token and generati_token[-1] == FINE:
        generati_token = generati_token[:-1]

    grafo_oro = token_a_grafo(esempio["risposta"], "fatto")
    try:
        grafo_generato = token_a_grafo(generati_token, "fatto")
    except ValueError:
        grafo_generato = None

    categoria = _categoria(grafo_oro, grafo_generato)
    return EsitoEsempio(tipo=esempio["tipo"], categoria=categoria, esatto=categoria == "esatto")


def campiona_per_valutazione(record: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Campiona `n` (storia, esempio) da `record`, restituiti come record a
    un solo esempio (stesso formato di `esami/genera.py`, riusabile da
    `valuta_dataset`)."""
    coppie = [(r["storia"], es) for r in record for es in r["esempi"]]
    rng.shuffle(coppie)
    return [{"storia": storia, "esempi": [es]} for storia, es in coppie[:n]]


def valuta_dataset(
    modello: Modello, vocab: Vocabolario, record: list[dict], ctx: int, device: str,
) -> dict[str, Any]:
    """Valuta un intero dataset (dev o esame). Ritorna un dict JSON-
    serializzabile con esattezza totale/per tipo e conteggi di calibrazione
    (invenzioni, astensioni_errate, malformate — PROGETTO.md, onestà
    epistemica)."""
    totali = {c: 0 for c in CATEGORIE}
    per_tipo: dict[str, dict[str, int]] = {}
    n = 0

    for r in record:
        for esempio in r["esempi"]:
            n += 1
            esito = valuta_esempio(modello, vocab, r["storia"], esempio, ctx, device)
            totali[esito.categoria] += 1
            d = per_tipo.setdefault(esito.tipo, {c: 0 for c in CATEGORIE} | {"n": 0})
            d[esito.categoria] += 1
            d["n"] += 1

    esattezza = totali["esatto"] / n if n else 0.0
    esattezza_per_tipo = {t: (d["esatto"] / d["n"] if d["n"] else 0.0) for t, d in per_tipo.items()}

    return {
        "n_esempi": n,
        "esattezza": esattezza,
        "esattezza_per_tipo": esattezza_per_tipo,
        "conteggi": totali,
        "conteggi_per_tipo": per_tipo,
    }


def _carica_record(percorso: Path) -> list[dict]:
    with open(percorso, encoding="utf-8") as f:
        return [json.loads(riga) for riga in f]


def _carica_modello(config: dict, percorso_checkpoint: str, device: str) -> Modello:
    vocab = carica_vocabolario()
    cfg_modello = ConfigModello(vocab_size=vocab.dimensione, ctx=config["dataset"]["ctx"], **config["modello"])
    modello = Modello(cfg_modello).to(device)
    stato = torch.load(percorso_checkpoint, map_location=device)
    modello.load_state_dict(stato["modello"])
    modello.eval()
    return modello


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--stadio", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    config = carica_config(args.config) if args.config else carica_config()
    device = dispositivo(config)
    vocab = carica_vocabolario()
    modello = _carica_modello(config, args.checkpoint, device)

    record = _carica_record(percorso_dataset(args.stadio, "esame", config))
    esito = valuta_dataset(modello, vocab, record, config["dataset"]["ctx"], device)

    dir_risultati = PROJECT_ROOT / config["percorsi"]["risultati_dir"] / config["nome_run"]
    dir_risultati.mkdir(parents=True, exist_ok=True)
    percorso_out = dir_risultati / f"esame_stadio{args.stadio}.json"
    with open(percorso_out, "w", encoding="utf-8") as f:
        json.dump(esito, f, ensure_ascii=False, indent=2)

    soglia = config["stadi"][args.stadio]["soglia"]
    print(f"stadio {args.stadio}: esattezza {esito['esattezza']:.4f} (soglia {soglia}, n={esito['n_esempi']})")
    for tipo, acc in sorted(esito["esattezza_per_tipo"].items()):
        d = esito["conteggi_per_tipo"][tipo]
        print(f"  {tipo}: {acc:.4f} (n={d['n']})")
    c = esito["conteggi"]
    print(f"invenzioni={c['invenzione']} astensioni_errate={c['astensione_errata']} malformate={c['malformata']}")
    print(f"-> {percorso_out}")

    raise SystemExit(0 if esito["esattezza"] >= soglia else 1)


if __name__ == "__main__":
    _cli()
