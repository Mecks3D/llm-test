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

Sezione aggiuntiva "tempo" (fasi/FASE2_PIANO_TEMPO.md §8, nota aperta):
anatomia degli errori per i tre tipi dell'esperimento "tempo"
(`posizione_tempo`, `azione_tempo`, `azione_luogo`), per distinguere gli
errori di TRACKING (il contenuto generato appartiene a un altro tick/luogo)
dagli errori di GENERAZIONE (tick giusto ma verbo o argomenti sbagliati).
Additiva: per i run senza tipi tempo la sezione resta vuota e il resto del
JSON è identico a prima.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mondo.domande import _evento_al_tick, _grafo_evento_senza_tempo, _posizione_al_tick
from mondo.grafo import NON_LO_SO, Grafo
from mondo.simulatore import Storia, genera_storia

from cervello.sequenza import token_a_grafo
from cervello.vocabolario import carica_vocabolario

from .esamina import CATEGORIE, _carica_modello, dispositivo, valuta_esempio
from .genera import (
    PROJECT_ROOT,
    TIPI_TEMPO,
    _LEMMA_A_VALORE_NUM,
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


# --- Sezione "tempo": anatomia degli errori dei tipi dell'esperimento
# fasi/FASE2_PIANO_TEMPO.md (posizione_tempo, azione_tempo, azione_luogo).
# Riusa gli helper di stato PRIVATI di mondo/domande.py (`_posizione_al_tick`,
# `_evento_al_tick`, `_grafo_evento_senza_tempo`): la diagnosi deve usare
# ESATTAMENTE la semantica del generatore delle domande, una copia locale
# potrebbe divergere (stesso precedente di `_lunghezza_storia` in genera.py).


def _canonico(grafo: Grafo, *, senza_tempo: bool = False) -> tuple[str, frozenset[tuple[str, str]]]:
    """Forma canonica `(lemma radice, {(relazione, lemma)})` per confronti di
    CONTENUTO tra grafi evento/fatto, insensibile all'ordine di costruzione;
    con `senza_tempo` ignora l'argomento `obl:tempo`. La radice è sempre il
    nodo 0 (convenzione di `grafo_a_token`/`token_a_grafo`)."""
    rami = frozenset(
        (a.relazione, grafo.nodi[a.dipendente].lemma)
        for a in grafo.archi
        if not (senza_tempo and a.relazione == "obl:tempo")
    )
    return grafo.nodi[0].lemma, rami


def _rami_per_relazione(grafo: Grafo) -> dict[str, str]:
    return {a.relazione: grafo.nodi[a.dipendente].lemma for a in grafo.archi}


def _tick_domanda(grafo_domanda: Grafo) -> int | None:
    lemma = _lemma_per_relazione(grafo_domanda, "obl:tempo")
    return _LEMMA_A_VALORE_NUM.get(lemma) if lemma is not None else None


def _nuovo_accumulatore_tempo(tipo: str) -> dict[str, Any]:
    acc: dict[str, Any] = {
        "n_oro_noto": 0,
        "esatti": 0,
        "anatomia_errori": Counter(),
        "distanza_tick_generato": Counter(),
    }
    if tipo in ("posizione_tempo", "azione_tempo"):
        acc["per_distanza_coda"] = {b: {"esatto": 0, "n": 0} for b in BUCKET_DISTANZA}
    if tipo in ("azione_tempo", "azione_luogo"):
        acc.update(
            per_n_archi_oro={},
            errori_verbo_giusto=0,
            relazioni_sbagliate=Counter(),
            relazioni_mancanti=Counter(),
            relazioni_in_piu=Counter(),
            origine_uguale_luogo_generato=0,
        )
    if tipo == "azione_tempo":
        acc["per_oro"] = {k: {"esatto": 0, "n": 0} for k in ("evento", "dormire_derivato")}
    if tipo == "azione_luogo":
        acc["luogo_richiesto_nel_generato"] = 0
    return acc


def _diagnosi_posizione_tempo(
    acc: dict, storia: Storia, pid: str, t: int, n_tick: int, esito: Any,
) -> None:
    """Anatomia di "posizione_tempo": il luogo generato È una posizione del
    protagonista, ma a quale distanza dal tick chiesto? Errori a distanza 1-2
    = tracking quasi giusto; posizione finale / luogo più frequente = le
    scorciatoie note; "mai" = luogo che non è mai stato una sua posizione."""
    acc["n_oro_noto"] += 1
    bucket = _bucket_distanza(n_tick - t)
    acc["per_distanza_coda"][bucket]["n"] += 1
    if esito.esatto:
        acc["esatti"] += 1
        acc["per_distanza_coda"][bucket]["esatto"] += 1
    if esito.categoria != "errore":
        return

    generato_luogo = _lemma_per_relazione(token_a_grafo(esito.token_generati, "fatto"), "obl:luogo")
    posizioni = {tt: _posizione_al_tick(storia, pid, tt) for tt in range(1, n_tick + 1)}
    tick_del_generato = [
        tt for tt, luogo in posizioni.items()
        if luogo is not None and luogo == generato_luogo and tt != t
    ]
    distanza_min = min((abs(tt - t) for tt in tick_del_generato), default=None)
    acc["distanza_tick_generato"][_bucket_distanza(distanza_min) if distanza_min is not None else "mai"] += 1

    luoghi = [e.luogo for e in storia.eventi if e.luogo is not None]
    piu_frequente = Counter(luoghi).most_common(1)[0][0] if luoghi else None
    if distanza_min is not None and distanza_min <= 2:
        acc["anatomia_errori"]["posizione_tick_vicino"] += 1
    elif generato_luogo == storia.stato_finale.luogo_effettivo(pid):
        acc["anatomia_errori"]["posizione_finale"] += 1
    elif generato_luogo == piu_frequente:
        acc["anatomia_errori"]["piu_frequente"] += 1
    elif distanza_min is not None:
        acc["anatomia_errori"]["posizione_tick_lontano"] += 1
    else:
        acc["anatomia_errori"]["mai_posizione"] += 1


def _diagnosi_azione(
    acc: dict, tipo: str, storia: Storia, pid: str, grafo_domanda: Grafo,
    t: int | None, n_tick: int, grafo_oro: Grafo, esito: Any,
) -> None:
    """Anatomia di "azione_tempo"/"azione_luogo". La domanda della nota
    aperta di FASE2_PIANO_TEMPO.md §8: gli errori sono di TRACKING (il
    contenuto generato è l'evento di un ALTRO tick/luogo) o di GENERAZIONE
    (tick giusto ma verbo/argomenti sbagliati)? Le categorie, in ordine di
    verifica: evento_di_altro_tick|luogo, verbo_sbagliato, struttura_diversa
    (stesso verbo, relazioni diverse), solo_origine_sbagliata (tutto giusto
    tranne `obl:origine` — che è la posizione al tick precedente, quindi a
    sua volta tracking), argomenti_sbagliati (con il conteggio delle
    relazioni sbagliate), solo_ordine_diverso (contenuto canonico identico,
    ordine di superficie diverso)."""
    n_archi = str(len(grafo_oro.archi))
    acc["per_n_archi_oro"].setdefault(n_archi, {"esatto": 0, "n": 0})["n"] += 1
    chiave_oro = bucket = None
    if tipo == "azione_tempo":
        bucket = _bucket_distanza(n_tick - t)
        acc["per_distanza_coda"][bucket]["n"] += 1
        chiave_oro = "evento" if _evento_al_tick(storia, pid, t) is not None else "dormire_derivato"
        acc["per_oro"][chiave_oro]["n"] += 1
    acc["n_oro_noto"] += 1
    if esito.esatto:
        acc["esatti"] += 1
        acc["per_n_archi_oro"][n_archi]["esatto"] += 1
        if tipo == "azione_tempo":
            acc["per_distanza_coda"][bucket]["esatto"] += 1
            acc["per_oro"][chiave_oro]["esatto"] += 1
    if esito.categoria != "errore":
        return

    grafo_generato = token_a_grafo(esito.token_generati, "fatto")
    gen_radice, _ = _canonico(grafo_generato)
    oro_radice, _ = _canonico(grafo_oro)
    gen_senza_tempo = _canonico(grafo_generato, senza_tempo=True)
    if gen_radice == oro_radice:
        acc["errori_verbo_giusto"] += 1

    # 1) tracking sbagliato: il contenuto generato (a meno del tempo) è
    # l'evento del protagonista a un ALTRO tick / in un ALTRO luogo.
    if tipo == "azione_tempo":
        altri_tick = [
            e.t for e in storia.eventi
            if e.agente == pid and e.t != t and _canonico(_grafo_evento_senza_tempo(e)) == gen_senza_tempo
        ]
        if altri_tick:
            acc["anatomia_errori"]["evento_di_altro_tick"] += 1
            acc["distanza_tick_generato"][_bucket_distanza(min(abs(tt - t) for tt in altri_tick))] += 1
            return
    else:
        luogo_chiesto = _lemma_per_relazione(grafo_domanda, "obl:luogo")
        if _lemma_per_relazione(grafo_generato, "obl:luogo") == luogo_chiesto:
            acc["luogo_richiesto_nel_generato"] += 1
        altri_luoghi = [
            e for e in storia.eventi
            if e.agente == pid and e.luogo not in (None, luogo_chiesto)
            and _canonico(_grafo_evento_senza_tempo(e)) == gen_senza_tempo
        ]
        if altri_luoghi:
            acc["anatomia_errori"]["evento_di_altro_luogo"] += 1
            return

    # 2) generazione sbagliata: dal verbo agli argomenti.
    if gen_radice != oro_radice:
        acc["anatomia_errori"]["verbo_sbagliato"] += 1
        return
    rami_gen = _rami_per_relazione(grafo_generato)
    rami_oro = _rami_per_relazione(grafo_oro)
    if set(rami_gen) != set(rami_oro):
        acc["anatomia_errori"]["struttura_diversa"] += 1
        for rel in sorted(set(rami_oro) - set(rami_gen)):
            acc["relazioni_mancanti"][rel] += 1
        for rel in sorted(set(rami_gen) - set(rami_oro)):
            acc["relazioni_in_piu"][rel] += 1
        return
    diverse = sorted(rel for rel in rami_oro if rami_gen[rel] != rami_oro[rel])
    if diverse == ["obl:origine"]:
        acc["anatomia_errori"]["solo_origine_sbagliata"] += 1
        if rami_gen["obl:origine"] == rami_gen.get("obl:luogo"):
            acc["origine_uguale_luogo_generato"] += 1
    elif diverse:
        acc["anatomia_errori"]["argomenti_sbagliati"] += 1
        for rel in diverse:
            acc["relazioni_sbagliate"][rel] += 1
    else:
        acc["anatomia_errori"]["solo_ordine_diverso"] += 1


def _serializza_tempo(acc: dict[str, Any]) -> dict[str, Any]:
    fuori: dict[str, Any] = {
        "n_oro_noto": acc["n_oro_noto"],
        "esattezza": _rateo(acc["esatti"], acc["n_oro_noto"]),
        "anatomia_errori": dict(acc["anatomia_errori"]),
        "distanza_tick_generato": dict(acc["distanza_tick_generato"]),
    }
    for chiave in ("per_distanza_coda", "per_oro", "per_n_archi_oro"):
        if chiave in acc:
            fuori[chiave] = {
                k: {"esattezza": _rateo(v["esatto"], v["n"]), "n": v["n"]}
                for k, v in sorted(acc[chiave].items())
            }
    for chiave in ("errori_verbo_giusto", "origine_uguale_luogo_generato", "luogo_richiesto_nel_generato"):
        if chiave in acc:
            fuori[chiave] = acc[chiave]
    for chiave in ("relazioni_sbagliate", "relazioni_mancanti", "relazioni_in_piu"):
        if chiave in acc:
            fuori[chiave] = dict(acc[chiave])
    return fuori


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
    tempo: dict[str, dict[str, Any]] = {}

    cache_storie: dict[tuple, tuple[Storia, int]] = {}

    def _storia_e_n_tick(r: dict) -> tuple[Storia, int]:
        chiave = (r["seed"], r.get("troncamento"))
        if chiave not in cache_storie:
            n_tick = r["troncamento"] if r.get("troncamento") is not None else _n_tick(stadio, r["seed"], config)
            storia = genera_storia(seed=r["seed"], n_tick=n_tick, persone=_cast_per_seed(config, r["seed"]))
            cache_storie[chiave] = (storia, n_tick)
        return cache_storie[chiave]

    def _storia_di(r: dict) -> Storia:
        return _storia_e_n_tick(r)[0]

    for r, esempio in coppie:
        n += 1
        esito = valuta_esempio(modello, vocab, r["storia"], esempio, ctx, device)
        totali[esito.categoria] += 1

        if esempio["tipo"] in TIPI_TEMPO:
            grafo_oro = token_a_grafo(esempio["risposta"], "fatto")
            if grafo_oro == NON_LO_SO:
                continue  # raro (inizio-storia-nel-sonno), l'anatomia non si applica
            grafo_domanda = token_a_grafo(esempio["domanda"], "fatto")
            pid = _lemma_per_relazione(grafo_domanda, "nsubj")
            t = _tick_domanda(grafo_domanda)
            if esempio["tipo"] != "azione_luogo" and t is None:
                continue  # domanda senza tick leggibile: non dovrebbe succedere
            storia, n_tick = _storia_e_n_tick(r)
            acc = tempo.setdefault(esempio["tipo"], _nuovo_accumulatore_tempo(esempio["tipo"]))
            if esempio["tipo"] == "posizione_tempo":
                _diagnosi_posizione_tempo(acc, storia, pid, t, n_tick, esito)
            else:
                _diagnosi_azione(acc, esempio["tipo"], storia, pid, grafo_domanda, t, n_tick, grafo_oro, esito)
            continue

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
        "tempo": {tipo: _serializza_tempo(acc) for tipo, acc in sorted(tempo.items())},
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
    for tipo, sezione in esito["tempo"].items():
        print(f"  [tempo] {tipo}: esattezza {sezione['esattezza']:.4f} (n={sezione['n_oro_noto']})")
        print(f"    anatomia errori: {sezione['anatomia_errori']}")
        if "errori_verbo_giusto" in sezione:
            n_errori = sum(sezione["anatomia_errori"].values())
            print(f"    errori con verbo giusto: {sezione['errori_verbo_giusto']}/{n_errori}")
    print(f"-> {percorso_out}")


if __name__ == "__main__":
    _cli()
