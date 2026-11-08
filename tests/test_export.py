import json

import numpy as np
import torch

from conftest import run_node
from tinylaw import export
from tinylaw.prompt import build_prompt, normalize_title


def test_export_roundtrip_and_js_parity(tmp_path, tiny_model, tiny_tokenizer):
    out = tmp_path / "model"
    cfg = export.export(tiny_model, tiny_tokenizer, out)
    assert (out / "model.bin").stat().st_size == 2 * cfg["params"]
    loaded = export.load_exported(out)
    ids = tiny_tokenizer.encode("Zákon č. 1/2026 Sb.\n\nZákon o povinném odpoledním spánku")
    ids = ids[: loaded.cfg.context]
    with torch.no_grad():
        ref, _ = loaded(torch.tensor([ids]))
        orig, _ = tiny_model(torch.tensor([ids]))
    # float16 export stays close to the float32 weights
    assert torch.allclose(ref, orig, atol=5e-2)
    js = np.array(run_node({"cmd": "logits", "model_dir": str(out), "ids": ids}))
    assert js.shape == tuple(ref[0].shape)
    np.testing.assert_allclose(js, ref[0].numpy(), atol=1e-4, rtol=1e-4)


def test_trained_like_weights_parity(tmp_path, tiny_tokenizer, tiny_config):
    # larger weights than the init, to exercise softmax and GELU ranges
    torch.manual_seed(1)
    from tinylaw.model import GPT

    m = GPT(tiny_config)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(8.0)
    out = tmp_path / "model"
    export.export(m, tiny_tokenizer, out)
    loaded = export.load_exported(out)
    ids = list(range(1, tiny_config.context + 1))
    with torch.no_grad():
        ref, _ = loaded(torch.tensor([ids]))
    js = np.array(run_node({"cmd": "logits", "model_dir": str(out), "ids": ids}))
    np.testing.assert_allclose(js, ref[0].numpy(), atol=2e-3, rtol=1e-3)


def test_prompt_python_and_js_agree(tmp_path, tiny_model, tiny_tokenizer):
    out = tmp_path / "model"
    export.export(tiny_model, tiny_tokenizer, out)
    cases = [
        ("Zákon o povinném odpoledním spánku", {}),
        ("  zákon,   kterým se zakazuje   pondělí. ", {"number": 77, "day": 13, "month": 12}),
        ("o draku na úřadě", {"year": 2027, "part": 9}),
        ("Povinné hlazení koček", {}),
        ("", {}),
        ("ZÁKON pro trpaslíky", {}),
        ("Zákonodárce a jeho kočka", {}),
    ]
    js = run_node({"cmd": "prompt", "config": str(out / "config.json"), "cases": cases})
    py = [build_prompt(t, **o) for t, o in cases]
    assert js == py


def test_normalize_title():
    assert normalize_title("Zákon o povinném spánku") == ("Zákon o povinném spánku",
                                                          "o povinném spánku")
    assert normalize_title("zákon, kterým se ruší pondělí.") == (
        "Zákon, kterým se ruší pondělí", "kterým se ruší pondělí")
    assert normalize_title("Povinné hlazení koček")[1] == "o povinné hlazení koček"
    assert normalize_title("Zákonodárce")[1] == "o zákonodárce"
    p = build_prompt("Zákon o povinném spánku")
    assert p.endswith("Parlament se usnesl na tomto zákoně České republiky:\n\n§ 1\n\n")


def test_config_contents(tmp_path, tiny_model, tiny_tokenizer):
    cfg = export.export(tiny_model, tiny_tokenizer, tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["version"] == cfg["version"]
    assert saved["model"]["n_layer"] == 2
    assert {"template", "months_genitive"} <= saved["prompt"].keys()
