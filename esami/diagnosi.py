"""Diagnosi qualitativa dell'esame (fasi/FASE2_PIANO_ANTISCORCIATOIA.md §5.3).

Rende permanenti le metriche dell'analisi qualitativa del checkpoint
`v1_facile` (2026-07-08, vissuta finora in script di scratchpad non
committati): oltre a esattezza e conteggi (come `esami/esamina.py`), calcola
per le domande "posizione" con oro noto le baseline euristiche di
frequenza/recency, l'esattezza condizionata (oro==luogo più frequente,
fasce di distanza dall'ultima menzione, tracking puro), l'anatomia degli
errori e l'esattezza per entità bersaglio — le stesse proprietà D1/D2/D3
usate dalla selezione anti-scorciatoia in `esami/genera.py` (qui NON
influenzano il training, solo la diagnosi).

La storia di ogni record si rigenera dal seed (`genera_storia` + `n_tick` +
cast del config, deterministico) invece di fidarsi solo dei token linearizzati
già presenti nel record: serve la struttura `Evento` per calcolare le
proprietà di tracking.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mondo.grafo import NON_LO_SO, Grafo
from mondo.simulatore import Storia, genera_storia

from cervello.sequenza import token_a_grafo
from cervello.vocabolario import carica_vocabolario

from .esamina import CATEGORIE, _carica_modello, dispositivo, valuta_esempio
from .genera import (
    PROJECT_ROOT,
    _cast_per_seed,
    _n_tick,
    carica_config,
    percorso_dataset,
    percorso_esame_tracking,
    percorso_tracking_tempo,
)

BUCKET_DISTANZA = ("0", "1-2", "3-5", ">=6")


def _bucket_distanza(distanza: int) -> str:
    if distanza == 0:
        return "0"
    if distanza <= 2:
        return "1-2"
    if distanza <= 5:
        return "3-5"
    return ">=6"


def _lemma_per_relazione(grafo: Grafo, relazione: str) -> str | None:
    for arco in grafo.archi:
        if arco.relazione == relazione:
            return grafo.nodi[arco.dipendente].lemma
    return None


def _bersaglio_e_oro(esempio: dict) -> tuple[str, str | None]:
    grafo_domanda = token_a_grafo(esempio["domanda"], "fatto")
    bersaglio = _lemma_per_relazione(grafo_domanda, "nsubj")
    grafo_risposta = token_a_grafo(esempio["risposta"], "fatto")
    if grafo_risposta == NON_LO_SO:
        return bersaglio, None
    return bersaglio, _lemma_per_relazione(grafo_risposta, "obl:luogo")


def _proprieta_posizione(storia: Storia, bersaglio: str, oro: str) -> dict[str, Any] | None:
    """Proprietà D1/D2/D3 (piano §2.1) più i dettagli che servono
    all'anatomia degli errori. `None` se il bersaglio non è mai menzionato
    da un evento valido (non dovrebbe capitare quando l'oro è noto)."""
    eventi = storia.eventi
    e_persona = bersaglio in storia.stato_finale.persone
    if e_persona:
        indici = [i for i, e in enumerate(eventi) if e.agente == bersaglio or e.destinatario == bersaglio]
    else:
        indici = [
            i for i, e in enumerate(eventi)
            if e.azione != "cercare" and (e.oggetto == bersaglio or e.argomento == bersaglio)
        ]
    if not indici:
        return None

    um = indici[-1]
    distanza_coda = len(eventi) - 1 - um
    luoghi = [e.luogo for e in eventi if e.luogo is not None]
    piu_frequente = Counter(luoghi).most_common(1)[0][0] if luoghi else None
    luogo_ultima_menzione = eventi[um].luogo
    ultimo_evento_luogo = eventi[-1].luogo if eventi else None
    luoghi_finali_altre_persone = {
        p: storia.stato_finale.persone[p].luogo
        for p in storia.stato_finale.persone if p != bersaglio
    }

    return {
        "oro_uguale_piu_frequente": oro == piu_frequente,
        "distanza_coda": distanza_coda,
        "d3_tracking_puro": oro != luogo_ultima_menzione,
        "piu_frequente": piu_frequente,
        "ultimo_evento_luogo": ultimo_evento_luogo,
        "luogo_ultima_menzione": luogo_ultima_menzione,
        "luoghi_finali_altre_persone": luoghi_finali_altre_persone,
    }


def _classifica_errore(generato_luogo: str | None, prop: dict[str, Any]) -> str:
    if generato_luogo == prop["piu_frequente"]:
        return "piu_frequente"
    if generato_luogo == prop["ultimo_evento_luogo"]:
        return "ultimo_evento"
    if generato_luogo in prop["luoghi_finali_altre_persone"].values():
        return "luogo_finale_di_altra_persona"
    if generato_luogo == prop["luogo_ultima_menzione"]:
        return "ultima_menzione_stantia"
    return "altro"


def _rateo(esatto: int, n: int) -> float:
    return esatto / n if n else 0.0


def esegui_diagnosi(
    modello: Any, vocab: Any, record: list[dict], config: dict, stadio: int,
    ctx: int, device: str, max_esempi: int | None = None,
) -> dict[str, Any]:
    """Valuta `record` (stesso formato di `esami/genera.py`) e calcola le
    metriche del piano §5.3. Solo le domande "posizione" con oro noto
    entrano nelle metriche 2-5 (baseline/condizionata/anatomia/per-entità):
    le altre contano solo per esattezza/conteggi complessivi (metrica 1)."""
    coppie = [(r, es) for r in record for es in r["esempi"]]
    if max_esempi is not None:
        coppie = coppie[:max_esempi]

    totali = {c: 0 for c in CATEGORIE}
    n = 0
    n_posizione_oro_noto = 0
    esatti_baseline = {"ultima_menzione": 0, "piu_frequente": 0, "ultimo_evento": 0, "modello": 0}
    condizionata: dict[str, dict[str, dict[str, int]]] = {
        "oro_uguale_piu_frequente": {k: {"esatto": 0, "n": 0} for k in ("si", "no")},
        "distanza_coda": {b: {"esatto": 0, "n": 0} for b in BUCKET_DISTANZA},
        "d3_tracking_puro": {k: {"esatto": 0, "n": 0} for k in ("si", "no")},
    }
    anatomia_errori: Counter = Counter()
    per_entita: dict[str, dict[str, int]] = {}

    cache_storie: dict[tuple, Storia] = {}

    def _storia_di(r: dict) -> Storia:
        chiave = (r["seed"], r.get("troncamento"))
        if chiave not in cache_storie:
            n_tick = r["troncamento"] if r.get("troncamento") is not None else _n_tick(stadio, r["seed"], config)
            cache_storie[chiave] = genera_storia(seed=r["seed"], n_tick=n_tick, persone=_cast_per_seed(config, r["seed"]))
        return cache_storie[chiave]

    for r, esempio in coppie:
        n += 1
        esito = valuta_esempio(modello, vocab, r["storia"], esempio, ctx, device)
        totali[esito.categoria] += 1

        if esempio["tipo"] != "posizione":
            continue
        bersaglio, oro = _bersaglio_e_oro(esempio)
        if oro is None:
            continue

        storia = _storia_di(r)
        prop = _proprieta_posizione(storia, bersaglio, oro)
        if prop is None:
            continue

        n_posizione_oro_noto += 1
        if prop["luogo_ultima_menzione"] == oro:
            esatti_baseline["ultima_menzione"] += 1
        if prop["piu_frequente"] == oro:
            esatti_baseline["piu_frequente"] += 1
        if prop["ultimo_evento_luogo"] == oro:
            esatti_baseline["ultimo_evento"] += 1
        if esito.esatto:
            esatti_baseline["modello"] += 1

        chiave_pf = "si" if prop["oro_uguale_piu_frequente"] else "no"
        condizionata["oro_uguale_piu_frequente"][chiave_pf]["n"] += 1
        bucket = _bucket_distanza(prop["distanza_coda"])
        condizionata["distanza_coda"][bucket]["n"] += 1
        chiave_d3 = "si" if prop["d3_tracking_puro"] else "no"
        condizionata["d3_tracking_puro"][chiave_d3]["n"] += 1
        if esito.esatto:
            condizionata["oro_uguale_piu_frequente"][chiave_pf]["esatto"] += 1
            condizionata["distanza_coda"][bucket]["esatto"] += 1
            condizionata["d3_tracking_puro"][chiave_d3]["esatto"] += 1

        entita = per_entita.setdefault(bersaglio, {"esatto": 0, "n": 0})
        entita["n"] += 1
        if esito.esatto:
            entita["esatto"] += 1

        if esito.categoria == "errore":
            grafo_generato = token_a_grafo(esito.token_generati, "fatto")
            generato_luogo = _lemma_per_relazione(grafo_generato, "obl:luogo")
            anatomia_errori[_classifica_errore(generato_luogo, prop)] += 1

    return {
        "n_esempi": n,
        "esattezza": _rateo(totali["esatto"], n),
        "conteggi": totali,
        "n_posizione_oro_noto": n_posizione_oro_noto,
        "baseline": {k: _rateo(v, n_posizione_oro_noto) for k, v in esatti_baseline.items()},
        "condizionata": {
            gruppo: {k: {"esattezza": _rateo(v["esatto"], v["n"]), "n": v["n"]} for k, v in sotto.items()}
            for gruppo, sotto in condizionata.items()
        },
        "anatomia_errori": dict(anatomia_errori),
        "per_entita": {
            e: {"esattezza": _rateo(d["esatto"], d["n"]), "n": d["n"]} for e, d in per_entita.items()
        },
    }


def _carica_record(percorso: Path) -> list[dict]:
    with open(percorso, encoding="utf-8") as f:
        return [json.loads(riga) for riga in f]


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stadio", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=("train", "dev", "esame", "tracking", "tracking-tempo"), default="esame")
    ap.add_argument("--max-esempi", type=int, default=None)
    args = ap.parse_args()

    config = carica_config(args.config)
    device = dispositivo(config)
    vocab = carica_vocabolario()
    modello = _carica_modello(config, args.checkpoint, device)

    if args.split == "tracking":
        percorso_in = percorso_esame_tracking(args.stadio, config)
    elif args.split == "tracking-tempo":
        # Split dell'esperimento "tempo" (fasi/FASE2_PIANO_TEMPO.md §4.3): le
        # metriche 2-5 sono specifiche di "posizione" (stato finale) e non si
        # applicano a "posizione_tempo" (stato a un tick passato), quindi qui
        # restano a zero — esattezza/conteggi (metrica 1) restano comunque
        # corretti, generici rispetto al tipo (vedi il loop sotto).
        percorso_in = percorso_tracking_tempo(args.stadio, config)
    else:
        percorso_in = percorso_dataset(args.stadio, args.split, config)
    record = _carica_record(percorso_in)
    esito = esegui_diagnosi(
        modello, vocab, record, config, args.stadio, config["dataset"]["ctx"], device,
        max_esempi=args.max_esempi,
    )

    dir_risultati = PROJECT_ROOT / config["percorsi"]["risultati_dir"] / config["nome_run"]
    dir_risultati.mkdir(parents=True, exist_ok=True)
    # split "esame" (il default di sempre) mantiene il nome storico;
    # gli altri (incluso "tracking", A3) si affiancano con un nome distinto.
    suffisso = "" if args.split == "esame" else f"_{args.split}"
    percorso_out = dir_risultati / f"diagnosi_stadio{args.stadio}{suffisso}.json"
    with open(percorso_out, "w", encoding="utf-8") as f:
        json.dump(esito, f, ensure_ascii=False, indent=2)

    print(f"stadio {args.stadio} ({args.split}): esattezza {esito['esattezza']:.4f} (n={esito['n_esempi']})")
    print(f"  posizione con oro noto: n={esito['n_posizione_oro_noto']}")
    for chiave, v in esito["baseline"].items():
        print(f"  baseline {chiave}: {v:.4f}")
    print(f"  anatomia errori (solo categoria 'errore'): {esito['anatomia_errori']}")
    print(f"-> {percorso_out}")


if __name__ == "__main__":
    _cli()
