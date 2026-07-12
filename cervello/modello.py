"""Transformer decoder-only stile nanoGPT (FASE2_PIANO.md §6).

Nessuna variazione architetturale: pre-LayerNorm, GELU, embedding
posizionali appresi, weight tying, attenzione causale via
`F.scaled_dot_product_attention`. Niente RoPE, Mamba, MoE — la virtù
della v1 è essere noiosa. Deve girare su CPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ConfigModello:
    vocab_size: int
    ctx: int
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 256
    d_ff: int = 1024
    dropout: float = 0.1


class CacheKV:
    """Cache incrementale di chiavi/valori per la decodifica autoregressiva
    (una entrata `(k, v)` per layer). NON tocca il training: entra in gioco solo
    quando la si passa esplicitamente a `Modello.forward`; senza cache il forward
    è identico a prima (byte per byte). Serve solo in inferenza greedy per non
    ricalcolare il forward sull'intera sequenza a ogni token generato."""

    def __init__(self, n_layer: int) -> None:
        self._kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n_layer
        self.lunghezza = 0  # numero di posizioni già in cache (per gli embedding di posizione)

    def leggi(self, layer: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._kv[layer]

    def scrivi(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self._kv[layer] = (k, v)

    def tronca_posizioni(self, n: int) -> None:
        """Rimuove le ultime `n` posizioni da ogni layer (rollback di token
        generati e poi scartati durante la decodifica interlacciata dello stato)."""
        for i, kv in enumerate(self._kv):
            if kv is not None:
                k, v = kv
                self._kv[i] = (k[:, :, : k.size(2) - n, :], v[:, :, : v.size(2) - n, :])
        self.lunghezza -= n

    def clona(self) -> "CacheKV":
        """Copia profonda dei tensori k/v: per riusare uno stesso prefisso (storia
        + stato) su più domande senza ricalcolarlo né corrompere l'originale."""
        c = CacheKV(len(self._kv))
        c.lunghezza = self.lunghezza
        c._kv = [None if kv is None else (kv[0].clone(), kv[1].clone()) for kv in self._kv]
        return c


class _Attenzione(nn.Module):
    def __init__(self, config: ConfigModello) -> None:
        super().__init__()
        if config.d_model % config.n_head != 0:
            raise ValueError("d_model deve essere multiplo di n_head")
        self.n_head = config.n_head
        self.d_testa = config.d_model // config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=True)

    def forward(
        self, x: torch.Tensor, cache: CacheKV | None = None, layer: int | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_testa).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_testa).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_testa).transpose(1, 2)
        if cache is None:
            # Percorso di sempre (training e decodifica non-cache): invariato.
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            prec = cache.leggi(layer)
            if prec is not None:  # accoda le nuove k/v a quelle già in cache
                k = torch.cat([prec[0], k], dim=2)
                v = torch.cat([prec[1], v], dim=2)
            cache.scrivi(layer, k, v)
            L = k.size(2)
            if T == 1:
                # singola query nuova: attende a tutto il passato + sé stessa
                y = F.scaled_dot_product_attention(q, k, v)
            elif prec is None:
                # priming del prefisso a cache vuota: causale puro, stesso kernel
                # (e stesso risultato) del percorso non-cache
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                # più token nuovi su cache non vuota: maschera causale esplicita
                # (query alla posizione L-T+i attende alle key j <= L-T+i)
                idx_q = torch.arange(L - T, L, device=x.device).view(T, 1)
                idx_k = torch.arange(L, device=x.device).view(1, L)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=idx_k <= idx_q)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class _MLP(nn.Module):
    def __init__(self, config: ConfigModello) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Blocco(nn.Module):
    def __init__(self, config: ConfigModello) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = _Attenzione(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = _MLP(config)
        self.dropout_resid = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, cache: CacheKV | None = None, layer: int | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout_resid(self.attn(self.ln1(x), cache=cache, layer=layer))
        x = x + self.dropout_resid(self.mlp(self.ln2(x)))
        return x


class Modello(nn.Module):
    def __init__(self, config: ConfigModello) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.ctx, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocchi = nn.ModuleList([_Blocco(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.testa = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.testa.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_pesi)
        for nome, p in self.named_parameters():
            if nome.endswith("proj.weight") or nome.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_pesi(modulo: nn.Module) -> None:
        if isinstance(modulo, nn.Linear):
            nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
            if modulo.bias is not None:
                nn.init.zeros_(modulo.bias)
        elif isinstance(modulo, nn.Embedding):
            nn.init.normal_(modulo.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, cache: CacheKV | None = None) -> torch.Tensor:
        B, T = idx.shape
        # Con la cache `idx` sono solo i token NUOVI: le posizioni proseguono da
        # quelle già in cache. Senza cache (training) pos_inizio=0, invariato.
        pos_inizio = cache.lunghezza if cache is not None else 0
        if pos_inizio + T > self.config.ctx:
            raise ValueError(
                f"sequenza di lunghezza {pos_inizio + T} supera ctx={self.config.ctx}"
            )
        pos = torch.arange(pos_inizio, pos_inizio + T, device=idx.device)
        x = self.dropout(self.tok_emb(idx) + self.pos_emb(pos))
        for i, blocco in enumerate(self.blocchi):
            x = blocco(x, cache=cache, layer=i)
        x = self.ln_f(x)
        if cache is not None:
            cache.lunghezza = pos_inizio + T
        return self.testa(x)

    def numero_parametri(self) -> int:
        return sum(p.numel() for p in self.parameters())
