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
from cervello.sequenza import (
    APERTA,
    CHIUSA,
    DOMANDA,
    FINE,
    RISPOSTA,
    STATO,
    STORIA,
    VERBO_STATO,
    token_a_grafo,
)
from cervello.vocabolario import Vocabolario, carica_vocabolario

from .genera import PROJECT_ROOT, carica_config, percorso_dataset

CATEGORIE = ("esatto", "invenzione", "astensione_errata", "malformata", "errore")


def dispositivo(config: dict) -> str:
    d = config.get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def _prossimo_id(modello: Modello, ids: list[int], device: str) -> int:
    """Argmax del token successivo dato il contesto `ids` (greedy, deterministico)."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = modello(x)
    return int(torch.argmax(logits[0, -1]).item())


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
            prossimo = _prossimo_id(modello, ids, device)
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
    token_generati: list[str]


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
    return EsitoEsempio(
        tipo=esempio["tipo"], categoria=categoria, esatto=categoria == "esatto",
        token_generati=generati_token,
    )


# ---------------------------------------------------------------------------
# Fase B: decodifica interlacciata d'esame (fasi/FASE2_PIANO_STATO.md §5)
#
# All'esame i blocchi [STATO] li GENERA il modello (decisione 1): l'esame non è
# più un singolo decode dopo [RISPOSTA] ma un decode interlacciato — eventi
# teacher-forced tick per tick, e a ogni confine il modello genera in free-run
# il blocco di stato, che si appende al contesto. La domanda resta dopo la
# storia, la risposta si genera come sempre.
# ---------------------------------------------------------------------------

# Cap difensivi: il vero stop del blocco è la radice != trovarsi (inizio del
# tick successivo) o un token di controllo. Questi limiti evitano solo che un
# modello mai addestrato generi all'infinito (cancello T4: forma, non qualità).
_MAX_GRUPPI_BLOCCO = 16       # etichetta di tick + posizioni (cast pieno = 6)
_MAX_TOKEN_GRUPPO = 24        # ( trovarsi ( nsubj p ) ( obl:luogo l ) ) = 11


@dataclass(frozen=True)
class EsitoEsempioStato:
    tipo: str
    categoria: str
    esatto: bool
    token_generati: list[str]
    blocchi_generati: list[list[str]]  # token generati per ogni blocco [STATO]


def _raggruppa_eventi_per_tick(storia_flat: list[str]) -> list[tuple[str, list[list[str]]]]:
    """Divide la storia-eventi (piatta, senza stato — la distribuzione d'esame
    ufficiale) nei suoi grafi-evento e li raggruppa per tick, nell'ordine della
    storia. Il tick è il lemma di `obl:tempo` dell'evento. Ritorna
    `[(tick_lemma, [evento_token, ...]), ...]`."""
    eventi: list[list[str]] = []
    i, n = 0, len(storia_flat)
    while i < n:
        if storia_flat[i] != APERTA:
            raise ValueError(f"atteso {APERTA!r} all'inizio di un evento, trovato {storia_flat[i]!r}")
        depth, j = 0, i
        while j < n:
            if storia_flat[j] == APERTA:
                depth += 1
            elif storia_flat[j] == CHIUSA:
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            raise ValueError("evento con parentesi non chiusa nella storia")
        eventi.append(storia_flat[i : j + 1])
        i = j + 1

    per_tick: list[tuple[str, list[list[str]]]] = []
    for ev in eventi:
        tick = ev[ev.index("obl:tempo") + 1]
        if per_tick and per_tick[-1][0] == tick:
            per_tick[-1][1].append(ev)
        else:
            per_tick.append((tick, [ev]))
    return per_tick


def _genera_blocco_stato(
    modello: Modello, vocab: Vocabolario, ids: list[int], ctx: int, device: str,
) -> list[int]:
    """Free-run del contenuto di un blocco [STATO] (`ids` termina con [STATO]).

    Genera gruppi `( ... )` finché il modello non emette l'inizio del tick
    successivo — un gruppo con radice diversa da `trovarsi` — o un token che non
    apre un gruppo (token di controllo). Il primo gruppo è l'etichetta di tick
    (`( obl:tempo <ord> )`), accettata comunque; i successivi solo se `trovarsi`.
    Modifica `ids` in place (appende il blocco) e ritorna gli id generati del
    contenuto. Deterministico."""
    id_ap, id_ch = vocab.id(APERTA), vocab.id(CHIUSA)
    inizio = len(ids)
    n_gruppi = 0
    while n_gruppi < _MAX_GRUPPI_BLOCCO and len(ids) < ctx:
        prossimo = _prossimo_id(modello, ids, device)
        if prossimo != id_ap:
            break  # non apre un gruppo: il blocco è finito (token non consumato)
        ids.append(prossimo)  # "("
        if len(ids) >= ctx:
            del ids[-1:]
            break
        radice_id = _prossimo_id(modello, ids, device)
        radice = vocab.token(radice_id)
        # Il primo gruppo è l'etichetta di tick (radice obl:tempo), accettata
        # comunque; i successivi solo se posizione (radice trovarsi). Una radice
        # diversa è l'inizio del tick successivo: chiudi il blocco senza
        # includere questo gruppo (bastano "(" e la radice per riconoscerlo).
        if n_gruppi > 0 and radice != VERBO_STATO:
            del ids[-1:]  # rimuovi il "(" ; la radice non è stata appesa
            break
        ids.append(radice_id)
        # completa il gruppo bilanciato ("(" apre a profondità 1)
        depth, n_tok = 1, 2
        while depth > 0 and len(ids) < ctx and n_tok < _MAX_TOKEN_GRUPPO:
            t = _prossimo_id(modello, ids, device)
            ids.append(t)
            n_tok += 1
            if t == id_ap:
                depth += 1
            elif t == id_ch:
                depth -= 1
        if depth != 0:
            del ids[len(ids) - n_tok:]  # gruppo non chiuso entro il cap: tronca
            break
        n_gruppi += 1
    return ids[inizio:]


def _genera_prefisso_stato(
    modello: Modello, vocab: Vocabolario, storia_flat: list[str], ctx: int, device: str,
) -> tuple[list[int], list[list[str]]]:
    """Free-run interlacciato dei blocchi [STATO] lungo la storia (§5): eventi
    teacher-forced tick per tick, blocco [STATO] generato dal modello a ogni fine
    tick. Ritorna gli id fino a PRIMA di [DOMANDA] e i blocchi generati (token).

    Dipende SOLO dalla storia, non dalla domanda: per una storia con più domande
    (l'esame ha fino a `n_per_tipo` esempi per storia) il prefisso si genera una
    volta e si riusa su tutte — byte-identico, perché la generazione è greedy e
    deterministica. Presuppone il modello già in `eval()` (lo garantisce il
    chiamante); avvolto in `no_grad` dai chiamanti."""
    ids = [vocab.id(STORIA)]
    blocchi_generati: list[list[str]] = []
    for _tick, eventi_tick in _raggruppa_eventi_per_tick(storia_flat):
        for ev_tok in eventi_tick:  # eventi dati, teacher-forced
            ids.extend(vocab.id(t) for t in ev_tok)
        ids.append(vocab.id(STATO))  # cue del blocco (non imparato a emettere)
        gen = _genera_blocco_stato(modello, vocab, ids, ctx, device)
        blocchi_generati.append([vocab.token(i) for i in gen])
    return ids, blocchi_generati


def _completa_risposta_stato(
    modello: Modello, vocab: Vocabolario, ids_prefisso: list[int],
    blocchi_generati: list[list[str]], esempio: dict, ctx: int, device: str,
) -> EsitoEsempioStato:
    """Dato il prefisso storia+stato (da `_genera_prefisso_stato`), appende
    [DOMANDA]+domanda+[RISPOSTA] e decodifica la risposta. MODIFICA `ids_prefisso`
    in place: chi lo riusa su più domande passa una copia (`list(...)`)."""
    ids = ids_prefisso
    ids.append(vocab.id(DOMANDA))
    ids.extend(vocab.id(t) for t in esempio["domanda"])
    ids.append(vocab.id(RISPOSTA))
    generati_ids = decodifica_greedy(modello, vocab, ids, ctx, device)

    generati_token = [vocab.token(i) for i in generati_ids]
    if generati_token and generati_token[-1] == FINE:
        generati_token = generati_token[:-1]

    grafo_oro = token_a_grafo(esempio["risposta"], "fatto")
    try:
        grafo_generato = token_a_grafo(generati_token, "fatto")
    except ValueError:
        grafo_generato = None

    categoria = _categoria(grafo_oro, grafo_generato)
    return EsitoEsempioStato(
        tipo=esempio["tipo"], categoria=categoria, esatto=categoria == "esatto",
        token_generati=generati_token, blocchi_generati=blocchi_generati,
    )


def valuta_esempio_stato(
    modello: Modello, vocab: Vocabolario, storia_flat: list[str], esempio: dict,
    ctx: int, device: str,
) -> EsitoEsempioStato:
    """Valuta un esempio con decodifica interlacciata (§5): eventi teacher-
    forced tick per tick, blocchi [STATO] generati dal modello, poi la risposta.
    Metrica primaria invariata (risposta finale, grafo vs grafo); i blocchi
    generati tornano per la metrica ausiliaria di `esami/diagnosi.py` (§6).

    Per un dataset con più domande sulla stessa storia usare `valuta_dataset`
    (`stato=True`), che genera il prefisso una sola volta per storia."""
    era_training = modello.training
    modello.eval()
    with torch.no_grad():
        ids, blocchi_generati = _genera_prefisso_stato(modello, vocab, storia_flat, ctx, device)
        esito = _completa_risposta_stato(modello, vocab, ids, blocchi_generati, esempio, ctx, device)
    if era_training:
        modello.train()
    return esito


def campiona_per_valutazione(record: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Campiona `n` (storia, esempio) da `record`, restituiti come record a
    un solo esempio (stesso formato di `esami/genera.py`, riusabile da
    `valuta_dataset`)."""
    coppie = [(r["storia"], es) for r in record for es in r["esempi"]]
    rng.shuffle(coppie)
    return [{"storia": storia, "esempi": [es]} for storia, es in coppie[:n]]


MAX_CAMPIONI_NON_ESATTI = 10


def valuta_dataset(
    modello: Modello, vocab: Vocabolario, record: list[dict], ctx: int, device: str,
    stato: bool = False,
) -> dict[str, Any]:
    """Valuta un intero dataset (dev o esame). Ritorna un dict JSON-
    serializzabile con esattezza totale/per tipo, conteggi di calibrazione
    (invenzioni, astensioni_errate, malformate — PROGETTO.md, onestà
    epistemica) e i primi campioni non esatti (token generati vs oro,
    per diagnosi: le malformate si guardano, non si indovinano).

    `stato=True` (Fase B): decodifica interlacciata (§5), il modello genera i
    blocchi [STATO] lungo la storia prima di rispondere. Default False:
    comportamento invariato, byte-identico. Con `stato=True` il prefisso
    storia+stato (la parte cara del decode) si genera UNA volta per storia e si
    riusa su tutte le sue domande (byte-identico, ma ~`n_per_tipo`× più veloce
    sull'esame)."""
    totali = {c: 0 for c in CATEGORIE}
    per_tipo: dict[str, dict[str, int]] = {}
    campioni_non_esatti: list[dict] = []
    n = 0

    era_training = modello.training
    if stato:
        modello.eval()
    for r in record:
        if stato:
            with torch.no_grad():
                ids_storia, blocchi = _genera_prefisso_stato(modello, vocab, r["storia"], ctx, device)
        for esempio in r["esempi"]:
            n += 1
            if stato:
                with torch.no_grad():
                    esito = _completa_risposta_stato(
                        modello, vocab, list(ids_storia), blocchi, esempio, ctx, device,
                    )
            else:
                esito = valuta_esempio(modello, vocab, r["storia"], esempio, ctx, device)
            totali[esito.categoria] += 1
            d = per_tipo.setdefault(esito.tipo, {c: 0 for c in CATEGORIE} | {"n": 0})
            d[esito.categoria] += 1
            d["n"] += 1
            if not esito.esatto and len(campioni_non_esatti) < MAX_CAMPIONI_NON_ESATTI:
                campioni_non_esatti.append({
                    "tipo": esito.tipo,
                    "categoria": esito.categoria,
                    "generato": esito.token_generati,
                    "oro": list(esempio["risposta"]),
                })
    if stato and era_training:
        modello.train()

    esattezza = totali["esatto"] / n if n else 0.0
    esattezza_per_tipo = {t: (d["esatto"] / d["n"] if d["n"] else 0.0) for t, d in per_tipo.items()}

    return {
        "n_esempi": n,
        "esattezza": esattezza,
        "esattezza_per_tipo": esattezza_per_tipo,
        "conteggi": totali,
        "conteggi_per_tipo": per_tipo,
        "campioni_non_esatti": campioni_non_esatti,
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
    stato = config["dataset"].get("stato", False)
    esito = valuta_dataset(modello, vocab, record, config["dataset"]["ctx"], device, stato=stato)

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
