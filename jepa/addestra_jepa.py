"""Script di addestramento e valutazione rapida in locale per il GraphWorldJEPA.

Genera dataset dello Stadio 1 (domande di posizione con cast pieno),
allena il modello JEPA su CPU e calcola l'accuratezza d'esame.
"""
from __future__ import annotations

import time
import random
import torch
import torch.optim as optim
import torch.nn.functional as F

from mondo.simulatore import genera_storia
from mondo.domande import genera_domande
from jepa.vocabolario_grafi import (
    ENTITA2ID,
    LUOGO2ID,
    AZIONI_SUPPORTATE,
    AZIONE2ID,
)
from jepa.modello_jepa import GraphWorldJEPA


def prepara_dataset_jepa(n_storie: int = 200, seed_base: int = 100):
    """Genera storie ed estrae sequenze di eventi e domande di posizione per il JEPA."""
    esempi = []
    for i in range(n_storie):
        seed = seed_base + i
        n_tick = random.Random(f"lunghezza-{seed}").randint(8, 22)
        storia = genera_storia(seed=seed, n_tick=n_tick)
        rng_d = random.Random(f"domande-{seed}")
        domande = genera_domande(storia, rng_d, n_per_tipo=8)

        # Filtriamo le domande di tipo "posizione"
        domande_pos = [d for d in domande if d.tipo == "posizione"]

        eventi_strutturati = []
        for ev in storia.eventi:
            if ev.agente in ENTITA2ID:
                dest = ev.luogo or ev.luogo_origine
                if dest and dest in LUOGO2ID and ev.azione in AZIONE2ID:
                    eventi_strutturati.append(
                        (ENTITA2ID[ev.agente], AZIONE2ID[ev.azione], LUOGO2ID[dest])
                    )

        ev_tensor = (
            torch.tensor(eventi_strutturati, dtype=torch.long)
            if eventi_strutturati
            else torch.zeros((0, 3), dtype=torch.long)
        )

        for d in domande_pos:
            soggetto = None
            for n in d.grafo_domanda.nodi:
                if n.lemma in ENTITA2ID:
                    soggetto = n.lemma
                    break

            if soggetto:
                risposta_oro = "non lo so"
                for n in d.grafo_risposta.nodi:
                    if n.lemma in LUOGO2ID:
                        risposta_oro = n.lemma
                        break
                    elif n.lemma == "non-lo-so":
                        risposta_oro = "non lo so"
                        break

                esempi.append(
                    {
                        "eventi_tensor": ev_tensor,
                        "soggetto_id": ENTITA2ID[soggetto],
                        "risposta_oro": risposta_oro,
                    }
                )

    return esempi


def addestra_e_valuta():
    print("=" * 60)
    print("GraphWorldJEPA — Addestramento e Valutazione su CPU")
    print("=" * 60)

    start_time = time.time()
    device = torch.device("cpu")

    print("[1/3] Generazione dataset train (200 storie) e dev (80 storie)...")
    train_data = prepara_dataset_jepa(n_storie=200, seed_base=100)
    dev_data = prepara_dataset_jepa(n_storie=80, seed_base=2000)
    print(f"      Esempi train: {len(train_data)}, Esempi dev: {len(dev_data)}")

    modello = GraphWorldJEPA(d_embed=64, soglia_non_lo_so=0.45).to(device)
    optimizer = optim.AdamW(modello.parameters(), lr=1e-2, weight_decay=1e-4)

    print("\n[2/3] Addestramento del modello sul micro-mondo...")
    epochs = 8
    accum_steps = 16

    for epoch in range(1, epochs + 1):
        modello.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for idx, ex in enumerate(train_data):
            logits = modello.inizializza_stato(batch_size=1)
            logits = modello.aggiorna_stato_sequenza(logits, ex["eventi_tensor"].to(device))

            risposta_oro = ex["risposta_oro"]
            if risposta_oro in LUOGO2ID:
                target_l = torch.tensor([LUOGO2ID[risposta_oro]], device=device)
                logits_subj = logits[0, ex["soggetto_id"], :].unsqueeze(0)
                loss = F.cross_entropy(logits_subj, target_l) / accum_steps
                loss.backward()
                total_loss += loss.item() * accum_steps

                if (idx + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

        if (len(train_data) % accum_steps) != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(1, len(train_data))
        if epoch % 2 == 0 or epoch == 1:
            print(f"      Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f}")

    print("\n[3/3] Valutazione d'esame (Graph vs Graph su cast pieno)...")
    modello.eval()
    corretti = 0
    totali = len(dev_data)

    with torch.no_grad():
        for ex in dev_data:
            logits = modello.inizializza_stato(batch_size=1)
            logits = modello.aggiorna_stato_sequenza(logits, ex["eventi_tensor"].to(device))

            energie, pred = modello.calcola_energia_risposte(logits, ex["soggetto_id"])
            if pred == ex["risposta_oro"]:
                corretti += 1

    acc = corretti / max(1, totali)
    elapsed = time.time() - start_time

    print("-" * 60)
    print(f"RISULTATI GRAPH-JEPA WORLD MODEL:")
    print(f"  • Accuratezza d'esame (Stadio 1): {acc * 100:.2f}% ({corretti}/{totali})")
    print(f"  • Tempo totale (Train + Eval):   {elapsed:.2f} secondi su CPU")
    print(f"  • Baseline Transformer (v1):     57.30%")
    print("-" * 60)

    return acc


if __name__ == "__main__":
    addestra_e_valuta()
