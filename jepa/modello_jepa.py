"""Implementazione del GraphWorldJEPA per il tracciamento di stato ed energia del micro-mondo.

Combina vincoli fisici cablati (Softmax di posizione) con un aggiornamento
dinamico a grafi e calcolo dell'energia per la risposta.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocabolario_grafi import (
    N_ENTITA,
    N_LUOGHI,
    ENTITA2ID,
    LUOGO2ID,
    ID2LUOGO,
    AZIONI_SUPPORTATE,
    AZIONE2ID,
)


class GraphWorldJEPA(nn.Module):
    """World Model a grafi latenti con predittore JEPA ed Energy Engine."""

    def __init__(
        self,
        d_embed: int = 64,
        soglia_non_lo_so: float = 0.45,
    ) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.soglia_non_lo_so = soglia_non_lo_so

        # Embeddings per le entità, luoghi e azioni
        self.embed_entita = nn.Embedding(N_ENTITA, d_embed)
        self.embed_luoghi = nn.Embedding(N_LUOGHI, d_embed)
        self.embed_azioni = nn.Embedding(len(AZIONI_SUPPORTATE), d_embed)

        # Rete di transizione di stato: prende (embed_entita, embed_azione, embed_destinazione) -> deltalogits
        self.updater = nn.Sequential(
            nn.Linear(d_embed * 3, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, N_LUOGHI),
        )

        # Inizializzazione dello stato di base (uniforme per ogni entità su tutti i luoghi)
        self.stato_base_logits = nn.Parameter(torch.zeros(N_ENTITA, N_LUOGHI))

    def inizializza_stato(self, batch_size: int = 1) -> torch.Tensor:
        """Ritorna lo stato iniziale (logits di posizione) per un batch: [batch_size, N_entita, N_luoghi]."""
        return self.stato_base_logits.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def ottieni_probabilita_posizione(self, logits: torch.Tensor) -> torch.Tensor:
        """Applica la Softmax di posizione (vincolo di esclusività spaziale: sum_l P = 1)."""
        return F.softmax(logits, dim=-1)

    def aggiorna_stato(
        self,
        logits_correnti: torch.Tensor,
        entita_idx: torch.Tensor,
        azione_idx: torch.Tensor,
        destinazione_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Esegue un passo di transizione di stato (GNN Predictor) data un'azione.
        
        Args:
            logits_correnti: [batch_size, N_entita, N_luoghi]
            entita_idx: [batch_size] indice dell'entità soggetta all'azione
            azione_idx: [batch_size] indice dell'azione
            destinazione_idx: [batch_size] indice del luogo destinazione
        """
        batch_size = logits_correnti.shape[0]

        # Estrai embeddings
        e_ent = self.embed_entita(entita_idx)  # [batch_size, d_embed]
        e_act = self.embed_azioni(azione_idx)  # [batch_size, d_embed]
        e_dest = self.embed_luoghi(destinazione_idx)  # [batch_size, d_embed]

        inp = torch.cat([e_ent, e_act, e_dest], dim=-1)  # [batch_size, d_embed * 3]
        delta = self.updater(inp)  # [batch_size, N_luoghi]

        # Aggiorna i logits dell'entità specifica
        nuovi_logits = logits_correnti.clone()
        batch_indices = torch.arange(batch_size, device=logits_correnti.device)
        nuovi_logits[batch_indices, entita_idx, :] += delta

        return nuovi_logits

    def aggiorna_stato_sequenza(
        self,
        logits_correnti: torch.Tensor,
        eventi_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Esegue tutti gli aggiornamenti di stato per gli eventi di una storia in una singola operazione vettorizzata.
        
        Args:
            logits_correnti: [1, N_entita, N_luoghi]
            eventi_tensor: [N_eventi, 3] dove le colonne sono (entita_idx, azione_idx, destinazione_idx)
        """
        if eventi_tensor.shape[0] == 0:
            return logits_correnti

        ent_idx = eventi_tensor[:, 0]
        act_idx = eventi_tensor[:, 1]
        dest_idx = eventi_tensor[:, 2]

        e_ent = self.embed_entita(ent_idx)
        e_act = self.embed_azioni(act_idx)
        e_dest = self.embed_luoghi(dest_idx)

        inp = torch.cat([e_ent, e_act, e_dest], dim=-1)
        deltas = self.updater(inp)  # [N_eventi, N_luoghi]

        nuovi_logits = logits_correnti.clone()
        nuovi_logits[0].index_add_(0, ent_idx, deltas)
        return nuovi_logits


    def calcola_energia_risposte(
        self,
        logits_finali: torch.Tensor,
        entita_idx: int,
    ) -> tuple[dict[str, float], str]:
        """Calcola l'energia per ciascun luogo candidato data l'entità quesito.
        
        Ritorna:
            energie: dizionario luogo -> energia (valori bassi = alta compatibilità)
            miglior_risposta: lemma del luogo con minima energia o "non lo so"
        """
        probabilita = self.ottieni_probabilita_posizione(logits_finali)[0, entita_idx, :]  # [N_luoghi]

        max_prob, idx_max = torch.max(probabilita, dim=0)

        energie: dict[str, float] = {}
        for l_idx, l_nome in ID2LUOGO.items():
            energie[l_nome] = float(1.0 - probabilita[l_idx].item())

        # Epistemic Honesty: se la probabilità massima è sotto la soglia, sceglie "non lo so"
        if max_prob.item() < self.soglia_non_lo_so:
            energie["non lo so"] = 0.0
            miglior_risposta = "non lo so"
        else:
            energie["non lo so"] = 1.0
            miglior_risposta = ID2LUOGO[idx_max.item()]

        return energie, miglior_risposta
