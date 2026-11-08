from conftest import SYNTHETIC, run_node

from tinylaw import EOT
from tinylaw.tokenizer import BPETokenizer, bytes_to_unicode


def test_bytes_to_unicode_is_bijective():
    m = bytes_to_unicode()
    assert len(m) == 256 and len(set(m.values())) == 256


def test_roundtrip(tiny_tokenizer: BPETokenizer):
    for text in SYNTHETIC + ["  dvě  mezery\n\n\nnový řádek\t tab", "neznámé ζ слово"]:
        assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text


def test_eot_is_single_token(tiny_tokenizer: BPETokenizer):
    ids = tiny_tokenizer.encode("a" + EOT + "b")
    assert tiny_tokenizer.eot_id in ids


def test_javascript_encoder_matches_python(tmp_path, tiny_tokenizer: BPETokenizer):
    import json

    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(tiny_tokenizer.export_web(), ensure_ascii=False), encoding="utf-8")
    texts = SYNTHETIC + [
        "", " ", "\n\n", "  a  b   c\n \n", "§ 1\n\n(1) a) b) c)", "Zákon č. 123/2026 Sb.",
        "it's we're they'll I'd", "x" + EOT + "y" + EOT, "neznámé ζ слово 😀😀", "1234567890",
    ]
    js = run_node({"cmd": "encode", "tokenizer": str(path), "texts": texts})
    assert js == [tiny_tokenizer.encode(t) for t in texts]
    ids = [tiny_tokenizer.encode(t) for t in texts]
    assert run_node({"cmd": "decode", "tokenizer": str(path), "ids": ids}) == texts
