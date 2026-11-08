"""Checks of the committed model in web/model/ (skipped until the model is exported)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import run_node
from tinylaw.export import load_exported
from tinylaw.prompt import build_prompt

MODEL = Path(__file__).resolve().parent.parent / "web" / "model"
pytestmark = pytest.mark.skipif(not (MODEL / "model.bin").exists(), reason="model not exported")


def test_config_matches_weights():
    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    assert (MODEL / "model.bin").stat().st_size == 2 * sum(t["length"] for t in cfg["tensors"])
    tok = json.loads((MODEL / "tokenizer.json").read_text(encoding="utf-8"))
    assert len(tok["vocab"]) == cfg["model"]["vocab_size"]


def test_real_model_javascript_parity():
    tok_path = MODEL / "tokenizer.json"
    prompt = build_prompt("Zákon o povinném odpoledním spánku")
    ids = run_node({"cmd": "encode", "tokenizer": str(tok_path), "texts": [prompt]})[0]
    ids = ids[:24]
    model = load_exported(MODEL)
    with torch.no_grad():
        ref, _ = model(torch.tensor([ids]))
    js = np.array(run_node({"cmd": "logits", "model_dir": str(MODEL), "ids": ids}))
    np.testing.assert_allclose(js, ref[0].numpy(), atol=2e-3, rtol=1e-3)
    assert (js.argmax(-1) == ref[0].numpy().argmax(-1)).all()
