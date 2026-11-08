"""Resumable CPU training.

Every `--save-minutes` (default 10) and at the end, the full state (weights, optimiser,
step, schedule, data position, RNG) is written atomically to `checkpoints/latest.pt`.
`--resume` continues exactly where it stopped. Train and validation loss are logged
with timestamps to `checkpoints/train.log`.

Usage:
    python -m tinylaw.train --benchmark 120          # measure speed only
    python -m tinylaw.train --epochs 1.5             # full run
    python -m tinylaw.train --epochs 1.5 --resume    # continue
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import CHECKPOINT_DIR, DATA_DIR
from .model import GPT, GPTConfig


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Windows:
    """Non-overlapping windows of context+1 tokens, shuffled per epoch with a fixed seed.

    The batch for a given step depends only on (seed, step), so the data position is
    restored exactly on resume.
    """

    def __init__(self, tokens: np.ndarray, context: int, batch: int, seed: int):
        self.tokens = tokens
        self.context = context
        self.batch = batch
        self.seed = seed
        self.n_windows = (len(tokens) - 1) // context
        self.per_epoch = self.n_windows // batch
        self._perm_epoch = -1
        self._perm = None

    def _permutation(self, epoch: int) -> np.ndarray:
        if epoch != self._perm_epoch:
            self._perm = np.random.default_rng(self.seed + epoch).permutation(self.n_windows)
            self._perm_epoch = epoch
        return self._perm

    def batch_at(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        epoch, i = divmod(step, self.per_epoch)
        idx = self._permutation(epoch)[i * self.batch:(i + 1) * self.batch]
        rows = np.stack([self.tokens[j * self.context:j * self.context + self.context + 1]
                         for j in idx]).astype(np.int64)
        x = torch.from_numpy(rows[:, :-1])
        y = torch.from_numpy(rows[:, 1:])
        return x, y


def lr_at(step: int, total: int, peak: float, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def evaluate(model: GPT, val: Windows, batches: int, autocast) -> float:
    model.eval()
    losses = []
    n = min(batches, max(1, val.n_windows // val.batch))
    for i in range(n):
        idx = np.arange(i * val.batch, (i + 1) * val.batch) % val.n_windows
        rows = np.stack([val.tokens[j * val.context:j * val.context + val.context + 1]
                         for j in idx]).astype(np.int64)
        x, y = torch.from_numpy(rows[:, :-1]), torch.from_numpy(rows[:, 1:])
        with autocast():
            _, loss = model(x, y)
        losses.append(float(loss))
    model.train()
    return float(np.mean(losses))


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate()}


def set_rng_state(s: dict) -> None:
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def train(args) -> dict:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "latest.pt"
    log_path = out_dir / "train.log"

    tokens_dir = Path(args.data_dir) / "tokens"
    train_tokens = np.memmap(tokens_dir / "train.bin", dtype=np.uint16, mode="r")
    val_tokens = np.memmap(tokens_dir / "val.bin", dtype=np.uint16, mode="r")

    cfg = GPTConfig(vocab_size=args.vocab_size, context=args.context, n_layer=args.n_layer,
                    n_embd=args.n_embd, n_head=args.n_head)
    model = GPT(cfg)
    opt = make_optimizer(model, args.lr, args.weight_decay)
    data = Windows(train_tokens, cfg.context, args.batch, args.seed)
    val = Windows(val_tokens, cfg.context, args.batch, args.seed)
    total_steps = args.steps or max(1, int(args.epochs * data.per_epoch))
    step = 0
    best_val = float("inf")
    train_seconds = 0.0

    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step = state["step"]
        total_steps = state["total_steps"]
        best_val = state.get("best_val", best_val)
        train_seconds = state.get("train_seconds", 0.0)
        set_rng_state(state["rng"])
        log(log_path, f"resume step={step} total_steps={total_steps}")
    elif not args.benchmark:
        log(log_path, f"start params={model.num_params()} config={json.dumps(cfg.to_dict())} "
                      f"total_steps={total_steps} batch={args.batch} "
                      f"train_tokens={len(train_tokens)} val_tokens={len(val_tokens)}")

    autocast = ((lambda: torch.autocast("cpu", dtype=torch.bfloat16)) if args.bf16
                else contextlib.nullcontext)
    fwd = torch.compile(model) if args.compile else model

    def state_dict() -> dict:
        return {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                "total_steps": total_steps, "config": cfg.to_dict(), "args": vars(args),
                "best_val": best_val, "train_seconds": train_seconds, "rng": rng_state()}

    model.train()
    t_last_save = time.time()
    t_window, tok_window, loss_window = time.time(), 0, []
    t_bench = time.time()
    while step < total_steps:
        lr = lr_at(step, total_steps, args.lr, args.warmup, args.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = time.time()
        x, y = data.batch_at(step)
        with autocast():
            _, loss = fwd(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        train_seconds += time.time() - t0
        step += 1
        tok_window += x.numel()
        loss_window.append(loss.item())

        if args.benchmark:
            elapsed = time.time() - t_bench
            if step == 3:  # skip warm-up steps
                t_bench, tok_window = time.time(), 0
            if step > 3 and elapsed >= args.benchmark:
                tps = tok_window / elapsed
                result = {"tokens_per_second": round(tps, 1), "steps": step,
                          "seconds_per_step": round(elapsed / (step - 3), 3),
                          "loss": round(float(loss), 4), "bf16": args.bf16,
                          "compile": args.compile, "threads": args.threads,
                          "batch": args.batch, "train_tokens": len(train_tokens)}
                print(json.dumps(result))
                return result
            continue

        if step % args.log_every == 0 or step == total_steps:
            dt = time.time() - t_window
            msg = (f"step={step}/{total_steps} lr={lr:.2e} train_loss={np.mean(loss_window):.4f} "
                   f"tok_per_s={tok_window / max(dt, 1e-9):.0f}")
            if step % args.eval_every == 0 or step == total_steps:
                v = evaluate(model, val, args.eval_batches, autocast)
                best_val = min(best_val, v)
                msg += f" val_loss={v:.4f}"
            log(log_path, msg)
            t_window, tok_window, loss_window = time.time(), 0, []

        if time.time() - t_last_save >= args.save_minutes * 60 or step == total_steps:
            save_checkpoint(ckpt_path, state_dict())
            t_last_save = time.time()

    log(log_path, f"done step={step} train_hours={train_seconds / 3600:.2f} best_val={best_val:.4f}")
    return {"step": step, "best_val": best_val}


def log(path: Path, msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out-dir", default=str(CHECKPOINT_DIR))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--benchmark", type=float, default=0, help="measure speed for N seconds")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=0, help="override total steps")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=320)
    ap.add_argument("--n-head", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--save-minutes", type=float, default=10.0)
    return ap


def main(argv=None) -> None:
    train(parser().parse_args(argv))


if __name__ == "__main__":
    main()
