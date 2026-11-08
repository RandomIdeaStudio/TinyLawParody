"""Generate a parody law from the command line.

Usage:
    python -m tinylaw.sample "Zákon o povinném odpoledním spánku" [--temperature 0.9]
    python -m tinylaw.sample "..." --model web/model      # use the exported float16 model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from . import CHECKPOINT_DIR, DATA_DIR, WEB_MODEL_DIR
from .export import load_checkpoint, load_exported
from .prompt import build_prompt
from .tokenizer import BPETokenizer


def generate(model, tok: BPETokenizer, title: str, temperature: float = 0.9, top_k: int = 40,
             max_new_tokens: int = 300, seed: int = 0) -> str:
    prompt = build_prompt(title)
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    out = model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k,
                         eot_id=tok.eot_id, generator=g)
    new = out[0, ids.shape[1]:].tolist()
    if new and new[-1] == tok.eot_id:
        new = new[:-1]
    return prompt + tok.decode(new)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("title")
    ap.add_argument("--model", type=Path, default=None, help="exported model directory")
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "latest.pt")
    ap.add_argument("--tokenizer", type=Path, default=DATA_DIR / "tokenizer.json")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    model = load_exported(args.model) if args.model else load_checkpoint(args.checkpoint)
    print(generate(model, BPETokenizer.load(args.tokenizer), args.title, args.temperature,
                   args.top_k, args.max_new_tokens, args.seed))


if __name__ == "__main__":
    main()
