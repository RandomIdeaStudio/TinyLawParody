"""Export a trained checkpoint to `web/model/` for the browser.

Writes:
- `model.bin`: all tensors as little-endian float16, concatenated;
- `config.json`: architecture, tensor manifest (name, shape, offset), prompt template,
  sampling defaults and the project version;
- `tokenizer.json`: vocabulary and merges for `web/tokenizer.js`.

Usage:
    python -m tinylaw.export [--checkpoint checkpoints/latest.pt] [--out web/model]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import CHECKPOINT_DIR, DATA_DIR, WEB_MODEL_DIR, version
from .model import GPT, GPTConfig
from .prompt import MONTHS_GENITIVE, PROMPT_TEMPLATE
from .tokenizer import BPETokenizer

MAX_FILE_BYTES = 25 * 1024 * 1024


def tensor_order(model: GPT) -> list[tuple[str, torch.Tensor]]:
    """Tensors in a fixed order; the tied output head is not stored twice."""
    return [(n, t) for n, t in model.state_dict().items()]


def export(model: GPT, tok: BPETokenizer, out_dir: Path, extra: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, offset = [], 0
    with open(out_dir / "model.bin", "wb") as f:
        for name, t in tensor_order(model):
            arr = t.detach().cpu().float().numpy().astype("<f2")
            f.write(arr.tobytes())
            manifest.append({"name": name, "shape": list(arr.shape), "offset": offset,
                             "length": int(arr.size)})
            offset += arr.size * 2
    config = {
        "format": "tinylaw-f16-v1",
        "version": version(),
        "model": model.cfg.to_dict(),
        "params": model.num_params(),
        "tensors": manifest,
        "prompt": {"template": PROMPT_TEMPLATE, "months_genitive": MONTHS_GENITIVE},
        "sampling": {"temperature_min": 0.6, "temperature_max": 1.4,
                     "temperature_default": 0.9, "top_k": 40, "max_new_tokens": 400},
    }
    if extra:
        config.update(extra)
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8")
    (out_dir / "tokenizer.json").write_text(
        json.dumps(tok.export_web(), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8")
    for p in out_dir.iterdir():
        if p.stat().st_size > MAX_FILE_BYTES:
            raise RuntimeError(f"{p} is larger than 25 MB")
    return config


def load_checkpoint(path: Path) -> GPT:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**state["config"]))
    model.load_state_dict(state["model"])
    model.eval()
    return model


def load_exported(out_dir: Path) -> GPT:
    """Load `web/model/` back into PyTorch (float16 weights widened to float32)."""
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    model = GPT(GPTConfig(**config["model"]))
    raw = np.fromfile(out_dir / "model.bin", dtype="<f2")
    sd = {}
    for t in config["tensors"]:
        start = t["offset"] // 2
        sd[t["name"]] = torch.from_numpy(
            raw[start:start + t["length"]].astype(np.float32).reshape(t["shape"]))
    model.load_state_dict(sd)
    model.eval()
    return model


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "latest.pt")
    ap.add_argument("--tokenizer", type=Path, default=DATA_DIR / "tokenizer.json")
    ap.add_argument("--out", type=Path, default=WEB_MODEL_DIR)
    args = ap.parse_args(argv)
    cfg = export(load_checkpoint(args.checkpoint), BPETokenizer.load(args.tokenizer), args.out)
    print(f"exported {cfg['params']} parameters to {args.out}")


if __name__ == "__main__":
    main()
