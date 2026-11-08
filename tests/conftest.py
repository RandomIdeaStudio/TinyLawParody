"""Shared fixtures: a tiny tokenizer and model on a few lines of synthetic text.

CI never trains the real model; everything here runs in seconds.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from tinylaw import EOT
from tinylaw.model import GPT, GPTConfig
from tinylaw.tokenizer import BPETokenizer, train_tokenizer

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "tests" / "js" / "harness.js"

SYNTHETIC = [
    "Zákon č. 1/2026 Sb.\n\nZákon o povinném odpoledním spánku\n\n§ 1\n\n"
    "(1) Každý občan je povinen spát od 13.00 do 14.30 hodin.\n"
    "(2) Ministerstvo vnitra stanoví vyhláškou a) délku spánku, b) polohu polštáře.",
    "§ 2\n\nPřestupky\n\nFyzická osoba se dopustí přestupku tím, že nespí; za přestupek "
    "lze uložit pokutu do 5 000 Kč.",
    "Byl jednou jeden král a ten měl tři dcery. Nejmladší princezna byla nejkrásnější "
    "ze všech, a proto ji drak odnesl do skleněného zámku za devatero horami.",
    "Šel Honza do světa, potkal babičku: „Kam jdeš, synku?“ — „Na zkušenou!“ Ďábel, "
    "čert, vodník… 123 456,78 Kč; 1. ledna 2026 ✓ ąę 😀",
]


def tiny_text(repeat: int = 20) -> list[str]:
    return SYNTHETIC * repeat


@pytest.fixture(scope="session")
def tiny_tokenizer() -> BPETokenizer:
    return BPETokenizer(train_tokenizer(tiny_text(), vocab_size=400))


@pytest.fixture(scope="session")
def tiny_config(tiny_tokenizer) -> GPTConfig:
    return GPTConfig(vocab_size=tiny_tokenizer.vocab_size, context=32, n_layer=2, n_embd=32,
                     n_head=2)


@pytest.fixture()
def tiny_model(tiny_config) -> GPT:
    torch.manual_seed(0)
    return GPT(tiny_config)


@pytest.fixture()
def tiny_data_dir(tmp_path, tiny_tokenizer) -> Path:
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    ids = []
    for doc in tiny_text(10):
        ids += tiny_tokenizer.encode(doc) + [tiny_tokenizer.eot_id]
    arr = np.asarray(ids, dtype=np.uint16)
    arr.tofile(tokens / "train.bin")
    arr[: len(arr) // 4].tofile(tokens / "val.bin")
    return tmp_path


def run_node(request: dict):
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for the JavaScript parity tests (install Node 22)")
    res = subprocess.run([node, str(HARNESS)], input=json.dumps(request), capture_output=True,
                         text=True, encoding="utf-8", check=False)
    if res.returncode != 0:
        raise AssertionError("node failed:\n" + res.stderr)
    return json.loads(res.stdout)


__all__ = ["EOT", "run_node", "tiny_text", "SYNTHETIC"]
