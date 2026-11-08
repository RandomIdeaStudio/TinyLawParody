"""Byte-level BPE tokenizer (vocabulary 8 192) and its JavaScript-friendly export.

The exported JSON is read by `web/tokenizer.js`; a test checks that the JavaScript
encoder gives the same ids as this module.

Usage:
    python -m tinylaw.tokenizer train      # data/corpus/train.txt -> data/tokenizer.json
    python -m tinylaw.tokenizer encode     # corpus -> data/tokens/{train,val}.bin (uint16)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import DATA_DIR, EOT

VOCAB_SIZE = 8192
# GPT-2 pre-tokenisation pattern, the one ByteLevel(use_regex=True) uses.
SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 mapping of the 256 byte values to printable unicode characters."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def train_tokenizer(docs, vocab_size: int = VOCAB_SIZE):
    """Train on an iterable of document strings (without end-of-text tokens)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(docs, trainer)
    return tok


class BPETokenizer:
    """Thin wrapper around a Hugging Face tokenizer with the end-of-text token."""

    def __init__(self, hf_tokenizer):
        self.tok = hf_tokenizer
        self.eot_id = hf_tokenizer.token_to_id(EOT)

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(path))

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def export_web(self) -> dict:
        """Compact JSON for the browser: vocabulary in id order and ranked merges."""
        data = json.loads(self.tok.to_str())
        vocab = data["model"]["vocab"]
        tokens = [None] * len(vocab)
        for token, idx in vocab.items():
            tokens[idx] = token
        for added in data.get("added_tokens", []):
            if added["id"] >= len(tokens):
                tokens.extend([None] * (added["id"] - len(tokens) + 1))
            tokens[added["id"]] = added["content"]
        merges = [m if isinstance(m, str) else " ".join(m) for m in data["model"]["merges"]]
        return {
            "type": "byte-level-bpe",
            "pattern": SPLIT_PATTERN,
            "special_tokens": {EOT: self.eot_id},
            "eot_id": self.eot_id,
            "vocab": tokens,
            "merges": merges,
        }


def encode_file(tok: BPETokenizer, src: Path, dst: Path, chunk_docs: int = 2000) -> int:
    """Encode a corpus file (documents separated by EOT) to a flat uint16 array."""
    assert tok.vocab_size <= 65536
    text = src.read_text(encoding="utf-8")
    docs = [d for d in text.split(EOT) if d]
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(dst, "wb") as f:
        for i in range(0, len(docs), chunk_docs):
            batch = tok.tok.encode_batch(docs[i:i + chunk_docs])
            for enc in batch:
                ids = np.asarray(enc.ids + [tok.eot_id], dtype=np.uint16)
                f.write(ids.tobytes())
                total += len(ids)
    return total


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["train", "encode"])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args(argv)
    corpus = args.data_dir / "corpus"
    tok_path = args.data_dir / "tokenizer.json"
    if args.command == "train":
        text = (corpus / "train.txt").read_text(encoding="utf-8")
        tok = BPETokenizer(train_tokenizer(d for d in text.split(EOT) if d))
        tok.save(tok_path)
        print(f"vocabulary {tok.vocab_size} -> {tok_path}")
    else:
        tok = BPETokenizer.load(tok_path)
        stats = {}
        for split in ("train", "val"):
            stats[split + "_tokens"] = encode_file(tok, corpus / f"{split}.txt",
                                                   args.data_dir / "tokens" / f"{split}.bin")
        (args.data_dir / "tokens" / "stats.json").write_text(json.dumps(stats, indent=1))
        print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
