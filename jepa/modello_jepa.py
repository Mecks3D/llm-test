"""Implementazione del GraphWorldJEPA per il tracciamento di stato, posizioni e possessi.

Combina vincoli fisici cablati (Softmax ed eredità spaziale) con un aggiornamento
dinamico a grafi e calcolo dell'energia per la risposta.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocabolario_grafi import (
    N_ENTITA,
    N_LUOGHI,
    N_PERSONE,
    N_TARGETS,
    ENTITA2ID,
    LUOGO2ID,
    ID2LUOGO,
    PERSONE_ID,
    AZIONI_SUPPORTATE,
    AZIONE2ID,
    TARGET2ID,
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

        # Embeddings per entità, target (luoghi + persone + nullo) e azioni
        self.embed_entita = nn.Embedding(N_ENTITA, d_embed)
        self.embed_targets = nn.Embedding(N_TARGETS, d_embed)
        self.embed_azioni = nn.Embedding(len(AZIONI_SUPPORTATE), d_embed)

        # Rete di transizione di stato: (embed_entita, embed_azione, embed_target) -> deltalogits
        self.updater = nn.Sequential(
            nn.Linear(d_embed * 3, d_embed * 2),
            nn.ReLU(),
            nn.Linear(d_embed * 2, N_TARGETS),
        )

        # Inizializzazione dello stato di base
        stato_init = torch.zeros(N_ENTITA, N_TARGETS)
        # Le persone non possono essere 'tenute' da altre persone
        stato_init[:N_PERSONE, N_LUOGHI:] = -100.0
        self.stato_base_logits = nn.Parameter(stato_init)

    def inizializza_stato(self, batch_size: int = 1) -> torch.Tensor:
        """Ritorna lo stato iniziale per un batch: [batch_size, N_entita, N_targets]."""
        return self.stato_base_logits.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def ottieni_probabilita(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax di posizione/target su tutti i bersagli [batch_size, N_entita, N_targets]."""
        return F.softmax(logits, dim=-1)

    def ottieni_probabilita_effettive_luogo(self, logits: torch.Tensor) -> torch.Tensor:
        """Calcola la probabilità di posizione effettiva nei luoghi per ogni entità.
        
        Per le persone è la probabilità diretta nei luoghi.
        Per gli oggetti è la probabilità diretta nei luoghi + la probabilità indiretta
        tramite le persone che li portano (eredità spaziale).
        """
        prob = self.ottieni_probabilita(logits)  # [batch_size, N_entita, N_targets]
        prob_direct = prob[:, :, :N_LUOGHI]  # [batch_size, N_entita, N_luoghi]
        prob_poss = prob[:, :, N_LUOGHI : N_LUOGHI + N_PERSONE]  # [batch_size, N_entita, N_persone]
        prob_persone_luoghi = prob_direct[:, :N_PERSONE, :]  # [batch_size, N_persone, N_luoghi]

        prob_indirect = torch.matmul(prob_poss, prob_persone_luoghi)  # [batch_size, N_entita, N_luoghi]
        return prob_direct + prob_indirect

    def ottieni_probabilita_posizione(self, logits: torch.Tensor) -> torch.Tensor:
        """Alias per backward compatibility con i test."""
        return self.ottieni_probabilita_effettive_luogo(logits)

    def aggiorna_stato(
        self,
        logits_correnti: torch.Tensor,
        entita_idx: torch.Tensor,
        azione_idx: torch.Tensor,
        destinazione_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Esegue un passo di transizione di stato per un evento."""
        batch_size = logits_correnti.shape[0]

        e_ent = self.embed_entita(entita_idx)
        e_act = self.embed_azioni(azione_idx)
        e_dest = self.embed_targets(destinazione_idx)

        inp = torch.cat([e_ent, e_act, e_dest], dim=-1)
        delta = self.updater(inp)

        nuovi_logits = logits_correnti.clone()
        batch_indices = torch.arange(batch_size, device=logits_correnti.device)
        nuovi_logits[batch_indices, entita_idx, :] += delta

        return nuovi_logits

    def aggiorna_stato_sequenza(
        self,
        logits_correnti: torch.Tensor,
        eventi_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Esegue l'aggiornamento vettorizzato di stato mantenendo l'ultimo evento temporale per ciascuna entità."""
        if eventi_tensor.shape[0] == 0:
            return logits_correnti

        # Trova l'ultimo evento per ciascuna entità presente
        ultimi_eventi = {}
        for i in range(eventi_tensor.shape[0] - 1, -1, -1):
            e_id = int(eventi_tensor[i, 0].item())
            if e_id not in ultimi_eventi:
                ultimi_eventi[e_id] = i

        indices = torch.tensor(list(ultimi_eventi.values()), device=eventi_tensor.device, dtype=torch.long)
        sub_eventi = eventi_tensor[indices]  # [N_entita_uniche, 3]

        sub_ent = sub_eventi[:, 0]
        sub_act = sub_eventi[:, 1]
        sub_targ = sub_eventi[:, 2]

        e_ent = self.embed_entita(sub_ent)
        e_act = self.embed_azioni(sub_act)
        e_targ = self.embed_targets(sub_targ)

        inp = torch.cat([e_ent, e_act, e_targ], dim=-1)
        deltas = self.updater(inp)  # [N_entita_uniche, N_targets]

        nuovi_logits = logits_correnti.clone()
        nuovi_logits[0, sub_ent, :] = deltas
        return nuovi_logits

    def calcola_energia_risposte(
        self,
        logits_finali: torch.Tensor,
        entita_idx: int,
        tipo_domanda: str = "posizione",
    ) -> tuple[dict[str, float], str]:
        """Calcola l'energia per ciascun candidato data la domanda e l'entità quesito.
        
        Args:
            logits_finali: stato latente finale
            entita_idx: indice dell'entità quesito (soggetto o oggetto)
            tipo_domanda: "posizione" | "possesso"
        """
        if tipo_domanda == "posizione":
            prob_eff = self.ottieni_probabilita_effettive_luogo(logits_finali)[0, entita_idx, :]
            max_prob, idx_max = torch.max(prob_eff, dim=0)

            energie: dict[str, float] = {}
            for l_idx, l_nome in ID2LUOGO.items():
                energie[l_nome] = float(1.0 - prob_eff[l_idx].item())

            if max_prob.item() < self.soglia_non_lo_so:
                energie["non lo so"] = 0.0
                miglior_risposta = "non lo so"
            else:
                energie["non lo so"] = 1.0
                miglior_risposta = ID2LUOGO[idx_max.item()]

            return energie, miglior_risposta

        elif tipo_domanda == "possesso":
            prob_all = self.ottieni_probabilita(logits_finali)[0, entita_idx, :]
            prob_eff = self.ottieni_probabilita_effettive_luogo(logits_finali)[0, entita_idx, :]
            max_eff_prob = torch.max(prob_eff).item()

            prob_poss = prob_all[N_LUOGHI : N_LUOGHI + N_PERSONE]
            max_poss_prob, idx_max_p = torch.max(prob_poss, dim=0)

            energie: dict[str, float] = {}
            for p_idx, p_nome in enumerate(PERSONE_ID):
                energie[p_nome] = float(1.0 - prob_poss[p_idx].item())
            energie["nessuno"] = float(1.0 - torch.sum(prob_all[:N_LUOGHI]).item())

            if max_poss_prob.item() > 0.35:
                energie["non lo so"] = 1.0
                miglior_risposta = PERSONE_ID[idx_max_p.item()]
            elif max_eff_prob < self.soglia_non_lo_so:
                energie["non lo so"] = 0.0
                miglior_risposta = "non lo so"
            else:
                energie["non lo so"] = 1.0
                miglior_risposta = "nessuno"

            return energie, miglior_risposta

        else:
            raise ValueError(f"Tipo domanda non supportato: {tipo_domanda}")

