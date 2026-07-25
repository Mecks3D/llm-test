"""Script di addestramento e valutazione rapida in locale per il GraphWorldJEPA.

Genera dataset di posizione e possesso con manipolazione di oggetti,
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
    PERSONA2ID,
    TARGET2ID,
    AZIONI_SUPPORTATE,
    AZIONE2ID,
    N_LUOGHI,
    normalizza_entita,
    normalizza_target,
)
from jepa.modello_jepa import GraphWorldJEPA


def estrai_evento_strutturato(ev) -> tuple[int, int, int] | None:
    """Mappa un Evento del simulatore nella tupla (entita_id, azione_id, target_id) per il JEPA."""
    azione = ev.azione
    if azione not in AZIONE2ID:
        return None
    act_idx = AZIONE2ID[azione]

    agente = normalizza_entita(ev.agente)
    oggetto = normalizza_entita(ev.oggetto)

    if azione == "andare":
        if not agente:
            return None
        dest = normalizza_target(ev.luogo or ev.luogo_origine)
        if dest and dest in TARGET2ID:
            return (ENTITA2ID[agente], act_idx, TARGET2ID[dest])

    elif azione in ("prendere", "raccogliere", "estrarre"):
        if not (oggetto and agente):
            return None
        targ = normalizza_target(agente)
        if targ and targ in TARGET2ID:
            return (ENTITA2ID[oggetto], act_idx, TARGET2ID[targ])

    elif azione == "posare":
        if not oggetto:
            return None
        dest = normalizza_target(ev.luogo or ev.luogo_origine)
        if dest and dest in TARGET2ID:
            return (ENTITA2ID[oggetto], act_idx, TARGET2ID[dest])

    elif azione in ("mettere", "mettere_dentro"):
        if not oggetto:
            return None
        dest = normalizza_target(ev.argomento or ev.luogo)
        if dest and dest in TARGET2ID:
            return (ENTITA2ID[oggetto], act_idx, TARGET2ID[dest])

    elif azione == "dare":
        if not (oggetto and ev.destinatario):
            return None
        dest = normalizza_target(ev.destinatario)
        if dest and dest in TARGET2ID:
            return (ENTITA2ID[oggetto], act_idx, TARGET2ID[dest])

    elif azione == "mangiare":
        if not oggetto:
            return None
        return (ENTITA2ID[oggetto], act_idx, TARGET2ID["nessuno"])

    return None


def prepara_dataset_jepa(n_storie: int = 200, seed_base: int = 100):
    """Genera storie ed estrae sequenze di eventi e domande di posizione/possesso per il JEPA."""
    esempi = []
    for i in range(n_storie):
        seed = seed_base + i
        n_tick = random.Random(f"lunghezza-{seed}").randint(8, 22)
        storia = genera_storia(seed=seed, n_tick=n_tick)
        rng_d = random.Random(f"domande-{seed}")
        domande = genera_domande(storia, rng_d, n_per_tipo=8)

        eventi_strutturati = []
        for ev in storia.eventi:
            tup = estrai_evento_strutturato(ev)
            if tup is not None:
                eventi_strutturati.append(tup)

        ev_tensor = (
            torch.tensor(eventi_strutturati, dtype=torch.long)
            if eventi_strutturati
            else torch.zeros((0, 3), dtype=torch.long)
        )

        for d in domande:
            if d.tipo == "posizione":
                soggetto = None
                for n in d.grafo_domanda.nodi:
                    ent = normalizza_entita(n.lemma)
                    if ent and ent in ENTITA2ID:
                        soggetto = ent
                        break

                if soggetto:
                    risposta_oro = "non lo so"
                    for n in d.grafo_risposta.nodi:
                        l_norm = normalizza_target(n.lemma)
                        if l_norm and l_norm in LUOGO2ID:
                            risposta_oro = l_norm
                            break
                        elif n.lemma == "non-lo-so":
                            risposta_oro = "non lo so"
                            break

                    esempi.append(
                        {
                            "eventi_tensor": ev_tensor,
                            "soggetto_id": ENTITA2ID[soggetto],
                            "tipo_domanda": "posizione",
                            "risposta_oro": risposta_oro,
                        }
                    )

            elif d.tipo == "possesso":
                oggetto = None
                for n in d.grafo_domanda.nodi:
                    ent = normalizza_entita(n.lemma)
                    if ent and ent in ENTITA2ID:
                        oggetto = ent
                        break

                if oggetto:
                    risposta_oro = "non lo so"
                    for n in d.grafo_risposta.nodi:
                        p_norm = normalizza_target(n.lemma)
                        if p_norm and p_norm in PERSONA2ID:
                            risposta_oro = p_norm
                            break
                        elif n.lemma == "nessuno":
                            risposta_oro = "nessuno"
                            break
                        elif n.lemma == "non-lo-so":
                            risposta_oro = "non lo so"
                            break

                    esempi.append(
                        {
                            "eventi_tensor": ev_tensor,
                            "soggetto_id": ENTITA2ID[oggetto],
                            "tipo_domanda": "possesso",
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
    epochs = 16
    accum_steps = 16

    for epoch in range(1, epochs + 1):
        modello.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for idx, ex in enumerate(train_data):
            logits = modello.inizializza_stato(batch_size=1)
            logits = modello.aggiorna_stato_sequenza(logits, ex["eventi_tensor"].to(device))

            risposta_oro = ex["risposta_oro"]
            s_idx = ex["soggetto_id"]
            tipo_d = ex["tipo_domanda"]

            if tipo_d == "posizione" and risposta_oro in LUOGO2ID:
                target_idx = LUOGO2ID[risposta_oro]
                logits_subj = logits[0, s_idx, :].unsqueeze(0)
                target_t = torch.tensor([target_idx], device=device)
                loss = F.cross_entropy(logits_subj, target_t) / accum_steps
                loss.backward()
                total_loss += loss.item() * accum_steps

            elif tipo_d == "possesso" and risposta_oro in PERSONA2ID:
                p_idx = PERSONA2ID[risposta_oro]
                target_col = N_LUOGHI + p_idx
                logits_subj = logits[0, s_idx, :].unsqueeze(0)
                target_t = torch.tensor([target_col], device=device)
                loss = F.cross_entropy(logits_subj, target_t) / accum_steps
                loss.backward()
                total_loss += loss.item() * accum_steps

            elif tipo_d == "possesso" and risposta_oro == "nessuno":
                # Quando la risposta è nessuno, l'oggetto è in un luogo fisico, non tenuto da una persona (target_col < N_LUOGHI)
                prob_eff = modello.ottieni_probabilita_effettive_luogo(logits)[0, s_idx, :]
                max_l_idx = torch.argmax(prob_eff).item()
                logits_subj = logits[0, s_idx, :].unsqueeze(0)
                target_t = torch.tensor([max_l_idx], device=device)
                loss = F.cross_entropy(logits_subj, target_t) / accum_steps
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
    corretti_totale = 0
    corretti_pos = 0
    totali_pos = 0
    corretti_poss = 0
    totali_poss = 0

    with torch.no_grad():
        for ex in dev_data:
            logits = modello.inizializza_stato(batch_size=1)
            logits = modello.aggiorna_stato_sequenza(logits, ex["eventi_tensor"].to(device))

            energie, pred = modello.calcola_energia_risposte(
                logits, ex["soggetto_id"], tipo_domanda=ex["tipo_domanda"]
            )

            esatto = (pred == ex["risposta_oro"])
            if esatto:
                corretti_totale += 1

            if ex["tipo_domanda"] == "posizione":
                totali_pos += 1
                if esatto:
                    corretti_pos += 1
            elif ex["tipo_domanda"] == "possesso":
                totali_poss += 1
                if esatto:
                    corretti_poss += 1

    acc_tot = corretti_totale / max(1, len(dev_data))
    acc_pos = corretti_pos / max(1, totali_pos)
    acc_poss = corretti_poss / max(1, totali_poss)
    elapsed = time.time() - start_time

    print("-" * 60)
    print(f"RISULTATI GRAPH-JEPA WORLD MODEL:")
    print(f"  • Accuratezza Totale d'Esame:   {acc_tot * 100:.2f}% ({corretti_totale}/{len(dev_data)})")
    print(f"    - Domande Posizione:          {acc_pos * 100:.2f}% ({corretti_pos}/{totali_pos})")
    print(f"    - Domande Possesso:           {acc_poss * 100:.2f}% ({corretti_poss}/{totali_poss})")
    print(f"  • Tempo totale (Train + Eval):   {elapsed:.2f} secondi su CPU")
    print(f"  • Baseline Transformer (v1):     57.30%")
    print("-" * 60)

    return acc_tot


if __name__ == "__main__":
    addestra_e_valuta()

