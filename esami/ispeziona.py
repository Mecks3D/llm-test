"""Ispezione degli head di attenzione (fasi/FASE2_PIANO_DIAGNOSI.md §2, A1).

Il modello è piccolo apposta per essere ispezionabile (PROGETTO.md): finora
si sono guardati solo gli output (esamina.py/diagnosi.py), mai i meccanismi
interni. Il meccanismo canonico per "trova l'ultima occorrenza di questo
lemma e copia il contesto" è l'induction head (Olsson et al. 2022). Qui si
calcolano, per ogni head di ogni layer, due punteggi su un campione di
esempi "posizione" con oro noto:

- **stessa entità**: attenzione dal token del bersaglio nella domanda (nsubj)
  verso le sue menzioni nella storia — sia la somma su tutte, sia la sola
  ultima (quella che determina l'oro). Alta su una head = quella head sa
  "ritrovare" l'entità di cui si parla.
- **induction classico**: attenzione, per ogni occorrenza ripetuta di un
  token di contenuto, verso il token immediatamente successivo alla sua
  occorrenza precedente — la firma generica del meccanismo di induction,
  indipendente dal compito specifico.

Nessuna variazione al modello: `F.scaled_dot_product_attention` (usato da
`cervello/modello.py`) non restituisce i pesi post-softmax, quindi il
forward pass si ricalcola qui "a mano" solo per l'attenzione, riusando i
sottomoduli già addestrati (niente seconda implementazione della rete che
potrebbe divergere — verificato bit a bit contro `Modello.forward` in
`tests/test_ispeziona.py`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from mondo.grafo import NON_LO_SO

from cervello.modello import Modello
from cervello.sequenza import CHIUSA, DOMANDA, STORIA, token_a_grafo
from cervello.vocabolario import RELAZIONI_UD, TOKEN_SPECIALI, Vocabolario, carica_vocabolario

from .esamina import _carica_modello, dispositivo
from .genera import PROJECT_ROOT, carica_config, percorso_dataset

STRUTTURALI = frozenset(TOKEN_SPECIALI) | frozenset(RELAZIONI_UD)


def _forward_ispezionato(
    modello: Modello, ids: list[int], device: str,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Rifà il forward pass di `modello` (batch=1), calcolando l'attenzione
    esplicitamente invece che col kernel fuso. Ritorna (pesi_per_layer,
    logits): `pesi_per_layer[l]` ha shape (n_head, T, T); i logits servono
    solo al test di fedeltà, per confrontarli con `modello(x)`."""
    era_training = modello.training
    modello.eval()
    x = torch.tensor([ids], dtype=torch.long, device=device)
    T = x.shape[1]
    if T > modello.config.ctx:
        raise ValueError(f"sequenza di lunghezza {T} supera ctx={modello.config.ctx}")
    pos = torch.arange(T, device=device)
    maschera = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    pesi_per_layer: list[torch.Tensor] = []
    with torch.no_grad():
        h = modello.dropout(modello.tok_emb(x) + modello.pos_emb(pos))
        for blocco in modello.blocchi:
            attn = blocco.attn
            xn = blocco.ln1(h)
            B, Tt, C = xn.shape
            q, k, v = attn.qkv(xn).split(C, dim=2)
            q = q.view(B, Tt, attn.n_head, attn.d_testa).transpose(1, 2)
            k = k.view(B, Tt, attn.n_head, attn.d_testa).transpose(1, 2)
            v = v.view(B, Tt, attn.n_head, attn.d_testa).transpose(1, 2)
            punteggi = (q @ k.transpose(-2, -1)) / (attn.d_testa ** 0.5) + maschera
            pesi = F.softmax(punteggi, dim=-1)
            pesi_per_layer.append(pesi[0])
            y = (pesi @ v).transpose(1, 2).contiguous().view(B, Tt, C)
            h = h + blocco.dropout_resid(attn.proj(y))
            h = h + blocco.dropout_resid(blocco.mlp(blocco.ln2(h)))
        h = modello.ln_f(h)
        logits = modello.testa(h)
    if era_training:
        modello.train()
    return pesi_per_layer, logits[0]


def pesi_attenzione(modello: Modello, ids: list[int], device: str) -> list[torch.Tensor]:
    """Pesi di attenzione post-softmax per ogni layer, shape (n_head, T, T)."""
    pesi, _ = _forward_ispezionato(modello, ids, device)
    return pesi


def _bersaglio_domanda(domanda: Sequence[str]) -> tuple[int, list[str]] | None:
    """Indice locale (in `domanda`) e token del lemma bersaglio (nsubj):
    un solo token per una persona, lemma+ordinale per un'istanza di
    risorsa (`( nsubj mela secondo )`). `None` se "nsubj" non compare."""
    for i, t in enumerate(domanda):
        if t == "nsubj":
            j = i + 1
            if domanda[j + 1] == CHIUSA:
                return j, [domanda[j]]
            return j, [domanda[j], domanda[j + 1]]
    return None


def _posizioni_menzione(storia_flat: Sequence[str], bersaglio: Sequence[str]) -> list[int]:
    """Indici locali (in `storia_flat`) di ogni occorrenza consecutiva
    esatta di `bersaglio` (1 token per una persona, 2 per un'istanza)."""
    n = len(bersaglio)
    bersaglio = list(bersaglio)
    return [i for i in range(len(storia_flat) - n + 1) if list(storia_flat[i:i + n]) == bersaglio]


def _somma_entita(
    pesi: list[torch.Tensor], storia_flat: list[str], esempio: dict,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per l'esempio dato: attenzione dal token bersaglio nella domanda
    verso (a) la somma di tutte le sue menzioni nella storia, (b) la sola
    ultima. Shape (n_layer, n_head) ciascuno. `None` se l'esempio è
    "non-lo-so", se manca "nsubj" nella domanda, o se il bersaglio non è
    mai menzionato letteralmente nella storia."""
    if token_a_grafo(esempio["risposta"], "fatto") == NON_LO_SO:
        return None
    trovato = _bersaglio_domanda(esempio["domanda"])
    if trovato is None:
        return None
    j, bersaglio = trovato
    menzioni = _posizioni_menzione(storia_flat, bersaglio)
    if not menzioni:
        return None

    offset_storia = 1  # dopo [STORIA]
    offset_domanda = 1 + len(storia_flat) + 1  # dopo [STORIA] storia [DOMANDA]
    pos_query = offset_domanda + j
    pos_storia = [offset_storia + i for i in menzioni]
    pos_ultima = max(pos_storia)

    n_layer = len(pesi)
    n_head = pesi[0].shape[0]
    ultima = torch.zeros(n_layer, n_head)
    tutte = torch.zeros(n_layer, n_head)
    for l, p in enumerate(pesi):
        riga = p[:, pos_query, :]  # (n_head, T)
        tutte[l] = riga[:, pos_storia].sum(dim=1)
        ultima[l] = riga[:, pos_ultima]
    return ultima, tutte


def induction_esempio(pesi: list[torch.Tensor], prefisso: Sequence[str]) -> tuple[torch.Tensor, int]:
    """Punteggio di induction classico: per ogni token di contenuto (non
    strutturale) che ripete un'occorrenza precedente, l'attenzione verso il
    token immediatamente successivo a quella occorrenza precedente. Somma
    su tutte le posizioni valide + il loro conteggio (per la media)."""
    n_layer = len(pesi)
    n_head = pesi[0].shape[0]
    totale = torch.zeros(n_layer, n_head)
    conteggio = 0
    ultima_occorrenza: dict[str, int] = {}
    for t, tok in enumerate(prefisso):
        if tok in STRUTTURALI:
            continue
        if tok in ultima_occorrenza:
            target = ultima_occorrenza[tok] + 1
            for l, p in enumerate(pesi):
                totale[l] += p[:, t, target]
            conteggio += 1
        ultima_occorrenza[tok] = t
    return totale, conteggio


def esegui_ispezione(
    modello: Modello, vocab: Vocabolario, record: list[dict], ctx: int, device: str,
    max_esempi: int | None = None,
) -> dict[str, Any]:
    """Valuta un campione di esempi "posizione" e aggrega i due punteggi
    per ogni head di ogni layer."""
    coppie = [(r, es) for r in record for es in r["esempi"] if es["tipo"] == "posizione"]
    n_totali = len(coppie)
    if max_esempi is not None:
        coppie = coppie[:max_esempi]

    n_layer, n_head = modello.config.n_layer, modello.config.n_head
    somma_ultima = torch.zeros(n_layer, n_head)
    somma_tutte = torch.zeros(n_layer, n_head)
    somma_induction = torch.zeros(n_layer, n_head)
    n_entita = 0
    n_induction = 0

    for r, esempio in coppie:
        storia_flat = r["storia"]
        prefisso = [STORIA, *storia_flat, DOMANDA, *esempio["domanda"]]
        if len(prefisso) > ctx:
            continue
        ids = [vocab.id(t) for t in prefisso]
        pesi = pesi_attenzione(modello, ids, device)

        ind_tot, ind_n = induction_esempio(pesi, prefisso)
        somma_induction += ind_tot
        n_induction += ind_n

        esito = _somma_entita(pesi, storia_flat, esempio)
        if esito is not None:
            ultima, tutte = esito
            somma_ultima += ultima
            somma_tutte += tutte
            n_entita += 1

    media_ultima = somma_ultima / n_entita if n_entita else somma_ultima
    media_tutte = somma_tutte / n_entita if n_entita else somma_tutte
    media_induction = somma_induction / n_induction if n_induction else somma_induction

    mappa = [
        {
            "layer": l,
            "testa": h,
            "stessa_entita_ultima": media_ultima[l, h].item(),
            "stessa_entita_tutte": media_tutte[l, h].item(),
            "induction": media_induction[l, h].item(),
        }
        for l in range(n_layer) for h in range(n_head)
    ]

    def _top(chiave: str, k: int = 5) -> list[dict]:
        return sorted(mappa, key=lambda d: d[chiave], reverse=True)[:k]

    return {
        "n_esempi_totali": n_totali,
        "n_esempi_usati": len(coppie),
        "n_esempi_entita": n_entita,
        "n_osservazioni_induction": n_induction,
        "mappa": mappa,
        "top_stessa_entita_ultima": _top("stessa_entita_ultima"),
        "top_induction": _top("induction"),
    }


def _carica_record(percorso: Path) -> list[dict]:
    with open(percorso, encoding="utf-8") as f:
        return [json.loads(riga) for riga in f]


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stadio", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=("train", "dev", "esame"), default="esame")
    ap.add_argument("--max-esempi", type=int, default=50)
    args = ap.parse_args()

    config = carica_config(args.config)
    device = dispositivo(config)
    vocab = carica_vocabolario()
    modello = _carica_modello(config, args.checkpoint, device)

    record = _carica_record(percorso_dataset(args.stadio, args.split, config))
    esito = esegui_ispezione(
        modello, vocab, record, config["dataset"]["ctx"], device, max_esempi=args.max_esempi,
    )

    dir_risultati = PROJECT_ROOT / config["percorsi"]["risultati_dir"] / config["nome_run"]
    dir_risultati.mkdir(parents=True, exist_ok=True)
    percorso_out = dir_risultati / f"ispezione_stadio{args.stadio}.json"
    with open(percorso_out, "w", encoding="utf-8") as f:
        json.dump(esito, f, ensure_ascii=False, indent=2)

    print(
        f"stadio {args.stadio} ({args.split}): "
        f"{esito['n_esempi_entita']}/{esito['n_esempi_usati']} esempi con bersaglio noto e menzionato "
        f"(di {esito['n_esempi_totali']} totali), {esito['n_osservazioni_induction']} osservazioni di induction"
    )
    print("top head per 'stessa entità (ultima menzione)':")
    for d in esito["top_stessa_entita_ultima"]:
        print(
            f"  layer {d['layer']} testa {d['testa']}: {d['stessa_entita_ultima']:.3f} "
            f"(tutte le menzioni: {d['stessa_entita_tutte']:.3f}, induction: {d['induction']:.3f})"
        )
    print("top head per induction classico:")
    for d in esito["top_induction"]:
        print(f"  layer {d['layer']} testa {d['testa']}: {d['induction']:.3f}")
    print(f"-> {percorso_out}")


if __name__ == "__main__":
    _cli()
