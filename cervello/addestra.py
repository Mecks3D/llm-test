"""Training loop e orchestrazione del curriculum (FASE2_PIANO.md §7).

`python -m cervello.addestra --config configs/v1.yaml` esegue in sequenza
stadio 1 -> esame 1 -> stadio 2 -> esame 2 -> stadio 3 -> esame 3,
fermandosi al primo esame fallito (exit code != 0). Il training dello
stadio N parte dai pesi dello stadio N-1 e usa come train la
CONCATENAZIONE dei train degli stadi <= N (ripasso: si aggiunge, non si
sostituisce). Con `--stadio N` esegue solo quello stadio.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from esami.esamina import campiona_per_valutazione, valuta_dataset
from esami.genera import PROJECT_ROOT, carica_config, percorso_dataset

from .dati import Batch, carica_esempi, componi_esempio, genera_batch, impacchetta_batch
from .modello import ConfigModello, Modello
from .vocabolario import Vocabolario, carica_vocabolario

_PERCORSO_CONFIG_DEFAULT = PROJECT_ROOT / "configs" / "v1.yaml"


def dispositivo(config: dict) -> str:
    d = config.get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def crea_ottimizzatore(modello: Modello, config: dict) -> torch.optim.AdamW:
    """AdamW con weight decay solo sui pesi 2D (matrici), non su bias/norme."""
    t = config["training"]
    decadono, non_decadono = [], []
    for p in modello.parameters():
        if not p.requires_grad:
            continue
        (decadono if p.dim() >= 2 else non_decadono).append(p)
    gruppi = [
        {"params": decadono, "weight_decay": t["weight_decay"]},
        {"params": non_decadono, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(gruppi, lr=t["lr"], betas=(t["beta1"], t["beta2"]))


def lr_per_step(step: int, config: dict) -> float:
    """Warmup lineare, poi cosine decay fino a lr/10."""
    t = config["training"]
    lr, warmup, max_step = t["lr"], t["warmup_step"], t["max_step"]
    if step < warmup:
        return lr * (step + 1) / warmup
    progresso = min((step - warmup) / max(1, max_step - warmup), 1.0)
    coeff = 0.5 * (1 + math.cos(math.pi * progresso))
    lr_min = lr / 10
    return lr_min + coeff * (lr - lr_min)


def _imposta_lr(ottimizzatore: torch.optim.Optimizer, lr: float) -> None:
    for gruppo in ottimizzatore.param_groups:
        gruppo["lr"] = lr


def calcola_loss(modello: Modello, batch: Batch, device: str) -> torch.Tensor:
    """Cross-entropy next-token mascherata sulla sola risposta."""
    input_ids = batch.input.to(device)
    bersaglio = batch.bersaglio.to(device)
    maschera = batch.maschera.to(device).float()
    logits = modello(input_ids)
    loss_tok = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), bersaglio.reshape(-1), reduction="none",
    )
    return (loss_tok * maschera.reshape(-1)).sum() / maschera.reshape(-1).sum().clamp(min=1)


def _gruppi_di_accumulo(
    esempi: list[list[str]], vocab: Vocabolario, batch_size: int, accumulo: int,
    rng: random.Random,
) -> Iterator[list[Batch]]:
    gruppo: list[Batch] = []
    for b in genera_batch(esempi, vocab, batch_size, rng):
        gruppo.append(b)
        if len(gruppo) == accumulo:
            yield gruppo
            gruppo = []
    if gruppo:
        yield gruppo


def _scrivi_log(percorso: Path, riga: dict) -> None:
    with open(percorso, "a", encoding="utf-8") as f:
        f.write(json.dumps(riga, ensure_ascii=False))
        f.write("\n")


def _valuta_dev(
    modello: Modello, vocab: Vocabolario, dev_record: list[dict], config: dict,
    device: str, rng: random.Random,
) -> dict[str, Any]:
    campione = campiona_per_valutazione(dev_record, config["training"]["dev_campione"], rng)
    composti = [
        componi_esempio([c["storia"]], c["esempi"][0]["domanda"], c["esempi"][0]["risposta"])
        for c in campione
    ]
    # Micro-batch della stessa taglia del training: un batch unico da
    # dev_campione sequenze lunghe fino a ctx non sta in memoria su GPU.
    batch_size = config["training"]["batch"]
    somma_loss, n_token = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(composti), batch_size):
            batch = impacchetta_batch(composti[i : i + batch_size], vocab)
            token_batch = int(batch.maschera.sum().item())
            somma_loss += calcola_loss(modello, batch, device).item() * token_batch
            n_token += token_batch
    loss_dev = somma_loss / max(n_token, 1)
    esiti = valuta_dataset(modello, vocab, campione, config["dataset"]["ctx"], device)
    return {"loss_dev": loss_dev, "esattezza_dev": esiti["esattezza"]}


def addestra_stadio(
    modello: Modello, ottimizzatore: torch.optim.Optimizer, config: dict, stadio: int,
    esempi_train: list[list[str]], dev_record: list[dict], vocab: Vocabolario, device: str,
    percorso_log: Path, seme: int,
) -> dict[str, Any]:
    """Allena finché l'esattezza dev supera soglia+0.01 per 2 valutazioni
    consecutive, o si raggiunge max_step. Logga ogni `intervallo_valutazione`
    step (loss train/dev, esattezza dev, token/sec) su `percorso_log`."""
    t = config["training"]
    soglia = config["stadi"][stadio]["soglia"]
    max_step, intervallo = t["max_step"], t["intervallo_valutazione"]

    step = 0
    epoca = 0
    passaggi_consecutivi = 0
    token_dal_log = 0
    t_ultimo_log = time.time()
    ultima_valutazione: dict[str, Any] = {"loss_dev": None, "esattezza_dev": 0.0}

    modello.train()
    while step < max_step:
        rng_epoca = random.Random(f"{seme}-stadio{stadio}-epoca{epoca}")
        for micro_batches in _gruppi_di_accumulo(esempi_train, vocab, t["batch"], t["accumulo"], rng_epoca):
            step += 1
            _imposta_lr(ottimizzatore, lr_per_step(step, config))

            ottimizzatore.zero_grad()
            loss_train = 0.0
            for mb in micro_batches:
                loss = calcola_loss(modello, mb, device) / len(micro_batches)
                loss.backward()
                loss_train += loss.item()
                token_dal_log += mb.input.numel()
            torch.nn.utils.clip_grad_norm_(modello.parameters(), t["grad_clip"])
            ottimizzatore.step()

            if step % intervallo == 0 or step >= max_step:
                modello.eval()
                rng_dev = random.Random(f"{seme}-dev-stadio{stadio}-step{step}")
                ultima_valutazione = _valuta_dev(modello, vocab, dev_record, config, device, rng_dev)
                modello.train()

                ora = time.time()
                token_al_secondo = token_dal_log / max(ora - t_ultimo_log, 1e-9)
                t_ultimo_log, token_dal_log = ora, 0

                riga = {
                    "step": step, "stadio": stadio, "loss_train": loss_train,
                    "loss_dev": ultima_valutazione["loss_dev"],
                    "esattezza_dev": ultima_valutazione["esattezza_dev"],
                    "token_al_secondo": token_al_secondo,
                }
                _scrivi_log(percorso_log, riga)
                print(f"[stadio {stadio}] step {step}: loss_train={loss_train:.4f} "
                      f"loss_dev={ultima_valutazione['loss_dev']:.4f} "
                      f"esattezza_dev={ultima_valutazione['esattezza_dev']:.4f} "
                      f"tok/s={token_al_secondo:.0f}")

                if ultima_valutazione["esattezza_dev"] >= soglia + 0.01:
                    passaggi_consecutivi += 1
                else:
                    passaggi_consecutivi = 0
                if passaggi_consecutivi >= 2:
                    return {"step_finale": step, **ultima_valutazione}

            if step >= max_step:
                break
        epoca += 1

    return {"step_finale": step, **ultima_valutazione}


def _carica_record(percorso: Path) -> list[dict]:
    with open(percorso, encoding="utf-8") as f:
        return [json.loads(riga) for riga in f]


def esegui_curriculum(config: dict, percorso_config: Path, solo_stadio: int | None = None) -> int:
    """Orchestrazione multi-stadio. Ritorna l'exit code (0 se tutti gli
    stadi eseguiti superano il proprio esame)."""
    vocab = carica_vocabolario()
    device = dispositivo(config)
    torch.manual_seed(config["seed_torch"])
    # Determinismo (FASE2_PIANO.md §2): su CPU il run è riproducibile byte
    # per byte; su GPU le operazioni senza variante deterministica emettono
    # un warning (warn_only) — sono le fonti residue da documentare nel log.
    torch.use_deterministic_algorithms(True, warn_only=True)

    nome_run = config["nome_run"]
    dir_risultati = PROJECT_ROOT / config["percorsi"]["risultati_dir"] / nome_run
    dir_risultati.mkdir(parents=True, exist_ok=True)
    percorso_log = dir_risultati / "log.jsonl"
    shutil.copy(percorso_config, dir_risultati / "config.yaml")

    cfg_modello = ConfigModello(vocab_size=vocab.dimensione, ctx=config["dataset"]["ctx"], **config["modello"])
    modello = Modello(cfg_modello).to(device)

    # Con --stadio N (N non minimo) si riprende dal checkpoint dello stadio
    # precedente: il training dello stadio N parte SEMPRE dai pesi di N-1,
    # mai da pesi casuali (FASE2_PIANO.md §7).
    if solo_stadio is not None and solo_stadio > min(config["stadi"]):
        stadio_prec = max(s for s in config["stadi"] if s < solo_stadio)
        percorso_prec = dir_risultati / f"stadio{stadio_prec}.pt"
        if not percorso_prec.exists():
            raise FileNotFoundError(
                f"per riprendere dallo stadio {solo_stadio} serve il checkpoint "
                f"dello stadio {stadio_prec}: {percorso_prec} non esiste "
                f"(eseguire prima lo stadio {stadio_prec} o il curriculum completo)"
            )
        stato = torch.load(percorso_prec, map_location=device)
        modello.load_state_dict(stato["modello"])
        print(f"ripresa dallo stadio {stadio_prec}: caricato {percorso_prec}")

    print(f"modello: {modello.numero_parametri()} parametri, device={device}")
    ottimizzatore = crea_ottimizzatore(modello, config)

    stadi = [solo_stadio] if solo_stadio is not None else sorted(config["stadi"])
    esempi_train_cumulativi: list[list[str]] = []
    stadi_precedenti = [s for s in sorted(config["stadi"]) if solo_stadio is None or s < solo_stadio]
    for s in stadi_precedenti:
        esempi_train_cumulativi.extend(carica_esempi(percorso_dataset(s, "train", config)))

    for stadio in stadi:
        esempi_train_cumulativi.extend(carica_esempi(percorso_dataset(stadio, "train", config)))
        dev_record = _carica_record(percorso_dataset(stadio, "dev", config))

        addestra_stadio(
            modello, ottimizzatore, config, stadio, esempi_train_cumulativi, dev_record,
            vocab, device, percorso_log, config["seed_torch"],
        )

        percorso_ckpt = dir_risultati / f"stadio{stadio}.pt"
        torch.save({"modello": modello.state_dict()}, percorso_ckpt)
        print(f"checkpoint -> {percorso_ckpt}")

        esame_record = _carica_record(percorso_dataset(stadio, "esame", config))
        modello.eval()
        esito_esame = valuta_dataset(modello, vocab, esame_record, config["dataset"]["ctx"], device)
        modello.train()

        percorso_esame = dir_risultati / f"esame_stadio{stadio}.json"
        with open(percorso_esame, "w", encoding="utf-8") as f:
            json.dump(esito_esame, f, ensure_ascii=False, indent=2)

        soglia = config["stadi"][stadio]["soglia"]
        print(f"stadio {stadio}: esattezza esame {esito_esame['esattezza']:.4f} (soglia {soglia}) -> {percorso_esame}")
        if esito_esame["esattezza"] < soglia:
            print(f"stadio {stadio} FALLITO: esattezza {esito_esame['esattezza']:.4f} < soglia {soglia}")
            return 1

    return 0


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_PERCORSO_CONFIG_DEFAULT))
    ap.add_argument("--stadio", type=int, default=None)
    args = ap.parse_args()

    config = carica_config(args.config)
    codice = esegui_curriculum(config, Path(args.config), solo_stadio=args.stadio)
    raise SystemExit(codice)


if __name__ == "__main__":
    _cli()
