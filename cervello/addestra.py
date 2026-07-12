"""Training loop e orchestrazione del curriculum (FASE2_PIANO.md §7).

`python -m cervello.addestra --config configs/v1.yaml` esegue in sequenza
stadio 1 -> esame 1 -> stadio 2 -> esame 2 -> stadio 3 -> esame 3,
fermandosi al primo esame fallito (exit code != 0). Il training dello
stadio N parte dai pesi dello stadio N-1 e usa come train la
CONCATENAZIONE dei train degli stadi <= N (ripasso: si aggiunge, non si
sostituisce). Con `--stadio N` esegue solo quello stadio.

Checkpoint intra-stadio: a ogni valutazione si salva (atomicamente)
`stadio<N>_parziale.pt` con modello, ottimizzatore, stato RNG e posizione
nel curriculum; se il file esiste al lancio, il training dello stadio N
riprende da lì invece che da capo (su CPU la ripresa riproduce byte per
byte il run ininterrotto). Il parziale si cancella a stadio completato.
Con `--copia-sicurezza DIR` (su Colab: una cartella di Drive) ogni file
appena scritto viene replicato in DIR e i checkpoint mancanti in locale
vengono recuperati da lì al lancio: i file si scrivono sempre in locale e
si copiano, mai direttamente sul mount di Drive (inaffidabile in scrittura
dentro un training loop).
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
    # Fase B: se il train ha i blocchi [STATO], la dev valuta col decode
    # interlacciato dell'esame (§5) — stessa distribuzione del training, il
    # modello genera lo stato prima di rispondere. Gated: senza dataset.stato è
    # il comportamento di sempre, byte-identico.
    stato = config["dataset"].get("stato", False)
    t0 = time.time()
    esiti = valuta_dataset(modello, vocab, campione, config["dataset"]["ctx"], device, stato=stato)
    # Il decode interlacciato è la parte cara: cronometrarlo dice subito, dalla
    # prima riga, se un eval su tutti gli `esame_storie` è sostenibile (§6).
    secondi_eval = time.time() - t0
    return {"loss_dev": loss_dev, "esattezza_dev": esiti["esattezza"], "secondi_eval": secondi_eval}


def _salva_parziale(
    percorso: Path, modello: Modello, ottimizzatore: torch.optim.Optimizer,
    stadio: int, step: int, epoca: int, step_inizio_epoca: int,
    passaggi_consecutivi: int, *, miglior_esattezza_dev: float = 0.0,
) -> None:
    """Checkpoint intra-stadio, scritto in modo atomico (file temporaneo +
    rename): un'interruzione durante la scrittura non corrompe il parziale
    precedente. Include lo stato RNG di torch così la ripresa consuma la
    stessa sequenza casuale (dropout) del run ininterrotto, e il miglior
    `esattezza_dev` visto finora (piano anti-scorciatoia §5.2) così la
    ripresa non "dimentica" il best già visto."""
    stato = {
        "stadio": stadio,
        "step": step,
        "epoca": epoca,
        "step_inizio_epoca": step_inizio_epoca,
        "passaggi_consecutivi": passaggi_consecutivi,
        "miglior_esattezza_dev": miglior_esattezza_dev,
        "modello": modello.state_dict(),
        "ottimizzatore": ottimizzatore.state_dict(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = percorso.with_suffix(".tmp")
    torch.save(stato, tmp)
    tmp.replace(percorso)


def carica_parziale(
    percorso: Path, modello: Modello, ottimizzatore: torch.optim.Optimizer,
    stadio: int, device: str,
) -> dict[str, Any]:
    """Carica un checkpoint intra-stadio dentro modello e ottimizzatore e
    ritorna la posizione di ripresa (step, epoca, step_inizio_epoca,
    passaggi_consecutivi, miglior_esattezza_dev) da passare ad
    `addestra_stadio`."""
    stato = torch.load(percorso, map_location=device)
    if stato["stadio"] != stadio:
        raise ValueError(
            f"il checkpoint parziale {percorso} appartiene allo stadio "
            f"{stato['stadio']}, non allo stadio {stadio}: rimuoverlo se non serve"
        )
    modello.load_state_dict(stato["modello"])
    ottimizzatore.load_state_dict(stato["ottimizzatore"])
    torch.set_rng_state(stato["rng_torch"].cpu())
    if stato.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in stato["rng_cuda"]])
    ripresa = {
        chiave: stato[chiave]
        for chiave in ("step", "epoca", "step_inizio_epoca", "passaggi_consecutivi")
    }
    ripresa["miglior_esattezza_dev"] = stato.get("miglior_esattezza_dev", 0.0)
    return ripresa


def _salva_best(percorso: Path, modello: Modello, step: int, esattezza_dev: float) -> None:
    """Checkpoint del miglior `esattezza_dev` visto finora nello stadio
    (piano anti-scorciatoia §5.2): stesso formato del checkpoint di fine
    stadio (solo pesi, niente ottimizzatore) più `step`/`esattezza_dev` per
    riferimento. Scrittura atomica come `_salva_parziale`."""
    stato = {"modello": modello.state_dict(), "step": step, "esattezza_dev": esattezza_dev}
    tmp = percorso.with_suffix(".tmp")
    torch.save(stato, tmp)
    tmp.replace(percorso)


def _copia_sicurezza(sorgente: Path, dir_dest: Path) -> None:
    """Copia un file nella directory di copia di sicurezza (`--copia-sicurezza`,
    su Colab una cartella di Drive). Si scrive sempre in locale e si COPIA su
    Drive, mai il contrario: scrivere direttamente sul mount di Drive dentro
    il training loop è notoriamente inaffidabile (celle bloccate, file a 0
    byte, ritardi di sync). Copia atomica rispetto ai lettori: tmp + rename."""
    dir_dest.mkdir(parents=True, exist_ok=True)
    dest = dir_dest / sorgente.name
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    shutil.copy2(sorgente, tmp)
    tmp.replace(dest)


def addestra_stadio(
    modello: Modello, ottimizzatore: torch.optim.Optimizer, config: dict, stadio: int,
    esempi_train: list[list[str]], dev_record: list[dict], vocab: Vocabolario, device: str,
    percorso_log: Path, seme: int, percorso_parziale: Path | None = None,
    ripresa: dict[str, Any] | None = None, dir_copia: Path | None = None,
    percorso_best: Path | None = None,
) -> dict[str, Any]:
    """Allena finché l'esattezza dev supera soglia+0.01 per 2 valutazioni
    consecutive, o si raggiunge max_step (a meno di `training.early_stop:
    false` nel config, che disattiva il primo criterio e forza sempre
    max_step — default assente/`true` = comportamento di sempre, byte
    identico). Logga ogni `intervallo_valutazione` step (loss train/dev,
    esattezza dev, token/sec) su `percorso_log`.
    Se `percorso_parziale` è dato, a ogni valutazione salva lì il checkpoint
    intra-stadio; con `ripresa` (da `carica_parziale`) riparte da dentro lo
    stadio invece che da step 0. Se `percorso_best` è dato, ad ogni
    valutazione che supera il massimo `esattezza_dev` visto finora nello
    stadio salva lì il modello (piano anti-scorciatoia §5.2); il massimo
    sopravvive a una ripresa perché è parte del checkpoint parziale."""
    t = config["training"]
    soglia = config["stadi"][stadio]["soglia"]
    max_step, intervallo = t["max_step"], t["intervallo_valutazione"]
    early_stop = t.get("early_stop", True)

    step = 0
    epoca = 0
    step_inizio_epoca = 0
    salta_gruppi = 0
    passaggi_consecutivi = 0
    miglior_esattezza_dev = 0.0
    if ripresa is not None:
        step = ripresa["step"]
        epoca = ripresa["epoca"]
        step_inizio_epoca = ripresa["step_inizio_epoca"]
        passaggi_consecutivi = ripresa["passaggi_consecutivi"]
        miglior_esattezza_dev = ripresa.get("miglior_esattezza_dev", 0.0)
        salta_gruppi = step - step_inizio_epoca
    token_dal_log = 0
    t_ultimo_log = time.time()
    ultima_valutazione: dict[str, Any] = {"loss_dev": None, "esattezza_dev": 0.0}

    modello.train()
    while step < max_step:
        rng_epoca = random.Random(f"{seme}-stadio{stadio}-epoca{epoca}")
        for micro_batches in _gruppi_di_accumulo(esempi_train, vocab, t["batch"], t["accumulo"], rng_epoca):
            # Ripresa: i gruppi già consumati prima dell'interruzione si
            # scartano DOPO il mescolamento di rng_epoca, così l'ordine dei
            # batch resta identico a quello del run ininterrotto.
            if salta_gruppi > 0:
                salta_gruppi -= 1
                continue
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

                secondi_eval = ultima_valutazione.get("secondi_eval")
                riga = {
                    "step": step, "stadio": stadio, "loss_train": loss_train,
                    "loss_dev": ultima_valutazione["loss_dev"],
                    "esattezza_dev": ultima_valutazione["esattezza_dev"],
                    "token_al_secondo": token_al_secondo,
                    "secondi_eval": secondi_eval,
                }
                _scrivi_log(percorso_log, riga)
                print(f"[stadio {stadio}] step {step}: loss_train={loss_train:.4f} "
                      f"loss_dev={ultima_valutazione['loss_dev']:.4f} "
                      f"esattezza_dev={ultima_valutazione['esattezza_dev']:.4f} "
                      f"tok/s={token_al_secondo:.0f}"
                      + (f" eval={secondi_eval:.0f}s" if secondi_eval is not None else ""))

                if ultima_valutazione["esattezza_dev"] > miglior_esattezza_dev:
                    miglior_esattezza_dev = ultima_valutazione["esattezza_dev"]
                    if percorso_best is not None:
                        _salva_best(percorso_best, modello, step, miglior_esattezza_dev)
                        if dir_copia is not None:
                            _copia_sicurezza(percorso_best, dir_copia)

                if ultima_valutazione["esattezza_dev"] >= soglia + 0.01:
                    passaggi_consecutivi += 1
                else:
                    passaggi_consecutivi = 0
                if early_stop and passaggi_consecutivi >= 2:
                    return {"step_finale": step, **ultima_valutazione}

                if percorso_parziale is not None:
                    _salva_parziale(
                        percorso_parziale, modello, ottimizzatore, stadio,
                        step, epoca, step_inizio_epoca, passaggi_consecutivi,
                        miglior_esattezza_dev=miglior_esattezza_dev,
                    )
                    if dir_copia is not None:
                        _copia_sicurezza(percorso_parziale, dir_copia)
                        _copia_sicurezza(percorso_log, dir_copia)

            if step >= max_step:
                break
        epoca += 1
        step_inizio_epoca = step

    return {"step_finale": step, **ultima_valutazione}


def _carica_record(percorso: Path) -> list[dict]:
    with open(percorso, encoding="utf-8") as f:
        return [json.loads(riga) for riga in f]


def esegui_curriculum(
    config: dict, percorso_config: Path, solo_stadio: int | None = None,
    copia_sicurezza: Path | None = None, pesi_iniziali: Path | None = None,
) -> int:
    """Orchestrazione multi-stadio. Ritorna l'exit code (0 se tutti gli
    stadi eseguiti superano il proprio esame). Con `copia_sicurezza` (una
    directory, su Colab tipicamente su Drive) ogni checkpoint/log/esito
    viene replicato lì appena scritto, e al lancio i checkpoint mancanti
    in locale vengono recuperati da lì (runtime Colab nuovo).

    `pesi_iniziali` fa ripartire un curriculum NUOVO (run diversa, es. un
    dataset più semplice) dai pesi di un checkpoint esterno invece che da
    pesi casuali — valido solo per il primo stadio della run: per stadi
    successivi si usa già la catena N-1 dentro la stessa run."""
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

    def _recupera_da_copia(percorso: Path) -> bool:
        """Se `percorso` manca in locale ma esiste nella copia di sicurezza
        (es. runtime Colab nuovo dopo un'interruzione), lo riporta in locale."""
        if percorso.exists() or copia_sicurezza is None:
            return percorso.exists()
        candidato = copia_sicurezza / percorso.name
        if candidato.exists():
            shutil.copy2(candidato, percorso)
            print(f"recuperato dalla copia di sicurezza: {candidato} -> {percorso}")
            return True
        return False

    # Anche il log si recupera: dopo una ripresa la curva di loss deve
    # restare intera (è una consegna di T7), non ricominciare a metà.
    _recupera_da_copia(percorso_log)

    cfg_modello = ConfigModello(vocab_size=vocab.dimensione, ctx=config["dataset"]["ctx"], **config["modello"])
    modello = Modello(cfg_modello).to(device)

    if pesi_iniziali is not None:
        if solo_stadio is not None and solo_stadio != min(config["stadi"]):
            raise ValueError(
                "--pesi-iniziali è valido solo per il primo stadio della run "
                f"(min={min(config['stadi'])}, richiesto solo_stadio={solo_stadio})"
            )
        stato = torch.load(pesi_iniziali, map_location=device)
        modello.load_state_dict(stato["modello"])
        print(f"pesi iniziali caricati da {pesi_iniziali}")

    # Con --stadio N (N non minimo) si riprende dal checkpoint dello stadio
    # precedente: il training dello stadio N parte SEMPRE dai pesi di N-1,
    # mai da pesi casuali (FASE2_PIANO.md §7). Se però esiste un checkpoint
    # PARZIALE dello stadio N (run interrotto a metà), lo stadio N-1 non
    # serve: si riprenderà dal parziale più sotto.
    if (
        solo_stadio is not None
        and solo_stadio > min(config["stadi"])
        and not _recupera_da_copia(dir_risultati / f"stadio{solo_stadio}_parziale.pt")
    ):
        stadio_prec = max(s for s in config["stadi"] if s < solo_stadio)
        percorso_prec = dir_risultati / f"stadio{stadio_prec}.pt"
        if not _recupera_da_copia(percorso_prec):
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

        percorso_parziale = dir_risultati / f"stadio{stadio}_parziale.pt"
        percorso_best = dir_risultati / f"stadio{stadio}_best.pt"
        _recupera_da_copia(percorso_best)
        ripresa = None
        if _recupera_da_copia(percorso_parziale):
            ripresa = carica_parziale(percorso_parziale, modello, ottimizzatore, stadio, device)
            print(
                f"ripresa intra-stadio da {percorso_parziale}: "
                f"step {ripresa['step']}, epoca {ripresa['epoca']}"
            )

        addestra_stadio(
            modello, ottimizzatore, config, stadio, esempi_train_cumulativi, dev_record,
            vocab, device, percorso_log, config["seed_torch"],
            percorso_parziale=percorso_parziale, ripresa=ripresa,
            dir_copia=copia_sicurezza, percorso_best=percorso_best,
        )

        percorso_ckpt = dir_risultati / f"stadio{stadio}.pt"
        torch.save({"modello": modello.state_dict()}, percorso_ckpt)
        print(f"checkpoint -> {percorso_ckpt}")
        if copia_sicurezza is not None:
            _copia_sicurezza(percorso_ckpt, copia_sicurezza)
            _copia_sicurezza(percorso_log, copia_sicurezza)
        percorso_parziale.unlink(missing_ok=True)
        if copia_sicurezza is not None:
            (copia_sicurezza / percorso_parziale.name).unlink(missing_ok=True)

        esame_record = _carica_record(percorso_dataset(stadio, "esame", config))
        modello.eval()
        # Fase B: se il train ha i blocchi [STATO], l'esame ufficiale li fa
        # GENERARE al modello (decode interlacciato, §5, "Decisione 1") — come la
        # dev. Senza `dataset.stato` è il decode di sempre, byte-identico.
        stato = config["dataset"].get("stato", False)
        esito_esame = valuta_dataset(
            modello, vocab, esame_record, config["dataset"]["ctx"], device, stato=stato,
        )
        modello.train()

        percorso_esame = dir_risultati / f"esame_stadio{stadio}.json"
        with open(percorso_esame, "w", encoding="utf-8") as f:
            json.dump(esito_esame, f, ensure_ascii=False, indent=2)
        if copia_sicurezza is not None:
            _copia_sicurezza(percorso_esame, copia_sicurezza)

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
    ap.add_argument(
        "--copia-sicurezza", type=Path, default=None,
        help="directory (es. su Drive, da Colab) dove replicare checkpoint, "
        "log ed esiti appena scritti; al lancio i checkpoint mancanti in "
        "locale vengono recuperati da lì",
    )
    ap.add_argument(
        "--pesi-iniziali", type=Path, default=None,
        help="checkpoint esterno (.pt) da cui caricare i pesi iniziali del "
        "primo stadio di questa run, invece di pesi casuali (es. per "
        "continuare l'allenamento di un modello già addestrato su un "
        "curriculum diverso/più semplice)",
    )
    args = ap.parse_args()

    config = carica_config(args.config)
    codice = esegui_curriculum(
        config, Path(args.config), solo_stadio=args.stadio,
        copia_sicurezza=args.copia_sicurezza, pesi_iniziali=args.pesi_iniziali,
    )
    raise SystemExit(codice)


if __name__ == "__main__":
    _cli()
