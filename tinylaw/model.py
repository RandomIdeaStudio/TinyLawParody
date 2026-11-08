"""A plain GPT-2 style decoder (pre-LayerNorm, learned positions, tied embeddings).

Kept deliberately simple so that `web/transformer.js` can mirror it exactly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    context: int = 256
    n_layer: int = 6
    n_embd: int = 320
    n_head: int = 5
    layer_norm_eps: float = 1e-5

    def to_dict(self) -> dict:
        return asdict(self)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.fc_out = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(c, dim=2)
        hs = c // self.n_head
        q = q.view(b, t, self.n_head, hs).transpose(1, 2)
        k = k.view(b, t, self.n_head, hs).transpose(1, 2)
        v = v.view(b, t, self.n_head, hs).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(y)
        # tanh-approximated GELU: easy to reproduce exactly in JavaScript
        x = x + self.fc_out(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.context, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        assert t <= self.cfg.context, f"sequence {t} longer than context {self.cfg.context}"
        pos = torch.arange(t, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = x @ self.wte.weight.T  # tied output projection
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = 40, eot_id: int | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            cond = idx[:, -self.cfg.context:]
            logits, _ = self(cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
            if eot_id is not None and int(nxt[0, 0]) == eot_id:
                break
        return idx
