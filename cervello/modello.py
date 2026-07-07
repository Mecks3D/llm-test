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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_testa).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_testa).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_testa).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0,
        )
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout_resid(self.attn(self.ln1(x)))
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

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        if T > self.config.ctx:
            raise ValueError(f"sequenza di lunghezza {T} supera ctx={self.config.ctx}")
        pos = torch.arange(T, device=idx.device)
        x = self.dropout(self.tok_emb(idx) + self.pos_emb(pos))
        for blocco in self.blocchi:
            x = blocco(x)
        x = self.ln_f(x)
        return self.testa(x)

    def numero_parametri(self) -> int:
        return sum(p.numel() for p in self.parameters())
