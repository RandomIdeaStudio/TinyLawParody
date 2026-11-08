"""Repository rules that can be checked automatically."""

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def locales() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((WEB / "locales").glob("*.json"))}


def test_minimum_languages():
    assert {"cs", "en"} <= locales().keys()


def test_every_key_in_every_language():
    locs = locales()
    keys = {lang: set(d) for lang, d in locs.items()}
    union = set().union(*keys.values())
    for lang, k in keys.items():
        assert not (union - k), f"{lang} is missing {sorted(union - k)}"
    for lang, d in locs.items():
        for key, value in d.items():
            assert isinstance(value, str) and value.strip(), f"{lang}:{key} is empty"


def test_every_used_key_exists():
    used = set()
    html = (WEB / "index.html").read_text(encoding="utf-8")
    used |= set(re.findall(r'data-i18n(?:-[a-z-]+)?="([^"]+)"', html))
    for js in WEB.glob("*.js"):
        used |= set(re.findall(r'\bt\("([^"]+)"', js.read_text(encoding="utf-8")))
    assert used, "no translated keys found"
    for lang, d in locales().items():
        assert not (used - d.keys()), f"{lang} is missing {sorted(used - d.keys())}"


def test_version_matches_changelog():
    v = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert f"[{v}]" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    cfg = WEB / "model" / "config.json"
    if cfg.exists():
        assert json.loads(cfg.read_text(encoding="utf-8"))["version"] == v


def test_committed_model_files_are_small():
    for p in (WEB / "model").glob("*"):
        assert p.stat().st_size < 25 * 1024 * 1024, p


def test_requirements_are_pinned():
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "--")):
            assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[A-Za-z0-9_.+\-]+", line), line


def test_no_external_requests_in_page():
    for p in list(WEB.glob("*.html")) + list(WEB.glob("*.js")) + list(WEB.glob("*.css")):
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"""(src|href)=["']https?://""", text), p
        assert "fetch(\"http" not in text and "fetch('http" not in text, p
