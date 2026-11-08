"""Download, clean and mix the training corpus.

Sources (see docs/DESIGN_MODEL.md):
- Czech laws: Hugging Face dataset roslein/Czech_legal_code (GPL-3.0).
- Public-domain Czech fairy tales from Czech Wikisource (MediaWiki API).

Usage:
    python -m tinylaw.data download     # laws CSV + Wikisource pages (cached)
    python -m tinylaw.data build        # cleaned, mixed corpus in data/corpus/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

from . import DATA_DIR, EOT, ROOT

LAWS_URL = (
    "https://huggingface.co/datasets/roslein/Czech_legal_code/resolve/main/Czech_legal_code.csv"
)
WIKISOURCE_API = "https://cs.wikisource.org/w/api.php"
WIKISOURCE_PAGE = "https://cs.wikisource.org/wiki/"
USER_AGENT = (
    "TinyLawParodyBot/0.1 (https://github.com/randomideastudio/tinylawparody; "
    "research, one request at a time)"
)

ROOT_CATEGORY = "Kategorie:Pohádky"
SKIP_CATEGORIES = {"Kategorie:Reklamní pohádky"}
COLLECTION_CATEGORY = "Kategorie:Sbírky pohádek"
AUTHOR_PAGES = ["Autor:Karel Jaromír Erben", "Autor:Božena Němcová"]
AUTHOR_LINK_PATTERN = re.compile(r"pohád|báchor|pověst", re.IGNORECASE)
# Works that are not fairy tales, or not in Czech.
EXCLUDE_PATTERNS = [
    re.compile(r"^Babička"),
    re.compile(r"nářečích"),
    re.compile(r"/O japonských pohádkách$"),
    re.compile(r"^Starobulharské"),
    re.compile(r"^(Uživatel|Wikizdroje|Diskuse|Kategorie|Autor|Šablona|Nápověda):"),
]
PD_LICENCES = {"PD old 70", "PD old 80", "PD old 100", "PD-old-70", "PD anon 70"}

VALIDATION_FRACTION = 0.01
FAIRY_TARGET_SHARE = 0.12  # share of training characters after upsampling
FAIRY_MAX_UPSAMPLE = 10
FAIRY_MIN_UPSAMPLE = 1


# --------------------------------------------------------------------------- text utils

_SPACE_CHARS = re.compile(r"[  -   　\t\f\v]")
_ZERO_WIDTH = re.compile(r"[​-‍⁠﻿­]")


def normalize_whitespace(text: str) -> str:
    """Unify line endings and unusual spaces; collapse runs; keep paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text)
    text = _SPACE_CHARS.sub(" ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- laws

_TOC_LINE = re.compile(r"^ \* ")


def clean_law(text: str) -> str:
    """Clean one law text from the CSV.

    The CSV text starts with a header (number, title, "Vyhlášeno ..."), then a table
    of contents whose lines start with " * " and are truncated copies of the body,
    then "Aktuální znění od ..." and the full body. The truncated table of contents
    is dropped (it duplicates the body and is cut mid-word); everything else is kept
    and only whitespace is normalised.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    in_toc = False
    body_started = False
    for line in lines:
        if not body_started:
            if line.startswith("Aktuální znění"):
                body_started = True
                in_toc = False
                out.append(line)
                continue
            if _TOC_LINE.match(line):
                in_toc = True
                continue
            if in_toc:
                # continuation lines and blank lines inside the table of contents
                continue
        out.append(line)
    if not body_started:  # no table of contents marker: keep the text as is
        out = lines
    return normalize_whitespace("\n".join(out))


def read_laws(csv_path: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    docs = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = clean_law(row["law_text"])
            if len(text) < 200:
                continue
            docs.append({"id": "law:" + row["law_id"], "source": "law", "text": text})
    return docs


# --------------------------------------------------------------------------- Wikisource


class WikisourceClient:
    """A slow, polite MediaWiki API client (one request at a time, with back-off)."""

    def __init__(self, delay: float = 2.0, max_retries: int = 8):
        import requests

        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.delay = delay
        self.max_retries = max_retries
        self._last = 0.0

    def get(self, **params) -> dict:
        params.setdefault("format", "json")
        params.setdefault("formatversion", "2")
        params.setdefault("maxlag", "5")
        for attempt in range(self.max_retries):
            wait = self.delay - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.session.get(WIKISOURCE_API, params=params, timeout=60)
            except Exception as exc:  # network hiccup
                print(f"[wikisource] {exc}; retrying", file=sys.stderr)
                time.sleep(5 * 2**attempt)
                continue
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                data = r.json()
                if "error" in data and data["error"].get("code") == "maxlag":
                    time.sleep(10)
                    continue
                return data
            retry_after = r.headers.get("Retry-After")
            pause = float(retry_after) if retry_after and retry_after.isdigit() else 5 * 2**attempt
            print(f"[wikisource] HTTP {r.status_code}; waiting {pause:.0f} s", file=sys.stderr)
            time.sleep(min(pause, 300))
        raise RuntimeError(f"Wikisource request failed after retries: {params}")

    def category_members(self, category: str, kind: str) -> list[str]:
        out, cont = [], {}
        while True:
            d = self.get(action="query", list="categorymembers", cmtitle=category,
                         cmtype=kind, cmlimit="500", **cont)
            out += [m["title"] for m in d["query"]["categorymembers"]]
            if "continue" not in d:
                return out
            cont = {"cmcontinue": d["continue"]["cmcontinue"]}

    def subpages(self, title: str) -> list[str]:
        out, cont = [], {}
        while True:
            d = self.get(action="query", list="allpages", apprefix=title + "/",
                         apnamespace="0", aplimit="500", apfilterredir="nonredirects", **cont)
            out += [p["title"] for p in d["query"]["allpages"]]
            if "continue" not in d:
                return out
            cont = {"apcontinue": d["continue"]["apcontinue"]}

    def links(self, title: str) -> list[str]:
        out, cont = [], {}
        while True:
            d = self.get(action="query", prop="links", titles=title, plnamespace="0",
                         pllimit="500", **cont)
            for p in d["query"]["pages"]:
                out += [link["title"] for link in p.get("links", [])]
            if "continue" not in d:
                return out
            cont = {"plcontinue": d["continue"]["plcontinue"]}

    def wikitexts(self, titles: list[str]) -> dict[str, dict]:
        """Return {title: {"wikitext", "revid", "missing"}} for up to 50 titles."""
        d = self.get(action="query", prop="revisions", rvprop="content|ids", rvslots="main",
                     titles="|".join(titles), redirects="1")
        q = d["query"]
        alias = {}
        for key in ("normalized", "redirects"):
            for m in q.get(key, []):
                alias[m["from"]] = m["to"]
        by_title = {}
        for p in q["pages"]:
            if p.get("missing") or "revisions" not in p:
                by_title[p["title"]] = {"missing": True}
                continue
            rev = p["revisions"][0]
            by_title[p["title"]] = {"wikitext": rev["slots"]["main"]["content"],
                                    "revid": rev["revid"], "title": p["title"]}
        out = {}
        for t in titles:
            final = t
            while final in alias and alias[final] != final:
                final = alias[final]
            out[t] = by_title.get(final, {"missing": True})
        return out

    def parsed_html(self, title: str) -> str:
        d = self.get(action="parse", page=title, prop="text", disablelimitreport="1",
                     disableeditsection="1", redirects="1")
        return d["parse"]["text"]


def _excluded(title: str) -> bool:
    return any(p.search(title) for p in EXCLUDE_PATTERNS)


def collect_titles(client: WikisourceClient) -> list[str]:
    """All fairy-tale pages: the category tree, collection subpages, Erben and Němcová."""
    pages: set[str] = set()
    collections: set[str] = set()
    seen_cats: set[str] = set()
    queue = [ROOT_CATEGORY]
    while queue:
        cat = queue.pop()
        if cat in seen_cats or cat in SKIP_CATEGORIES:
            continue
        seen_cats.add(cat)
        members = client.category_members(cat, "page")
        if cat == COLLECTION_CATEGORY or "Sbírky" in cat:
            collections.update(members)
        else:
            pages.update(members)
        queue += client.category_members(cat, "subcat")
    for author in AUTHOR_PAGES:
        for t in client.links(author):
            if AUTHOR_LINK_PATTERN.search(t):
                collections.add(t)
    for coll in sorted(collections):
        subs = client.subpages(coll)
        if subs:
            pages.update(subs)
            pages.discard(coll)
        else:
            pages.add(coll)
    # a page with subpages is a collection index or preface, not a tale
    titles = sorted(t for t in pages if not _excluded(t))
    parents = {t.rsplit("/", 1)[0] for t in titles if "/" in t}
    return [t for t in titles if t not in parents]


def download_wikisource(raw_dir: Path, client: WikisourceClient | None = None) -> Path:
    """Fetch every fairy-tale page into raw_dir/wikisource.jsonl (resumable cache)."""
    client = client or WikisourceClient()
    raw_dir.mkdir(parents=True, exist_ok=True)
    titles_path = raw_dir / "wikisource_titles.json"
    out_path = raw_dir / "wikisource.jsonl"
    if titles_path.exists():
        titles = json.loads(titles_path.read_text(encoding="utf-8"))
    else:
        titles = collect_titles(client)
        titles_path.write_text(json.dumps(titles, ensure_ascii=False, indent=1), encoding="utf-8")
    done: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["requested"])
    todo = [t for t in titles if t not in done]
    print(f"[wikisource] {len(titles)} titles, {len(todo)} to fetch", file=sys.stderr)
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), 50):
            batch = todo[i:i + 50]
            texts = client.wikitexts(batch)
            for t in batch:
                rec = dict(texts.get(t, {"missing": True}))
                rec["requested"] = t
                rec.setdefault("title", t)
                if not rec.get("missing") and "<pages" in rec["wikitext"]:
                    rec["html"] = client.parsed_html(t)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[wikisource] {min(i + 50, len(todo))}/{len(todo)}", file=sys.stderr)
    return out_path


# --- wikitext cleaning

def _strip_balanced(text: str, open_s: str, close_s: str, keep=None) -> str:
    """Remove balanced open_s ... close_s spans; keep(inner) may return replacement text."""
    out, i, n = [], 0, len(text)
    while i < n:
        j = text.find(open_s, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        depth, k = 1, j + len(open_s)
        while k < n and depth:
            if text.startswith(open_s, k):
                depth += 1
                k += len(open_s)
            elif text.startswith(close_s, k):
                depth -= 1
                k += len(close_s)
            else:
                k += 1
        inner = text[j + len(open_s):k - len(close_s)] if depth == 0 else text[j + len(open_s):]
        if keep is not None:
            rep = keep(inner)
            if rep:
                out.append(rep)
        i = k
    return "".join(out)


# Formatting templates whose last argument is running text.
_TEXT_TEMPLATES = {
    "prostrkaně", "proložit", "kapitálky", "malé", "velké", "center", "centrovat", "c",
    "np", "nadpis", "uprostřed", "doprava", "sc", "small", "larger", "large", "kurzíva",
    "tučně", "rozpal", "right", "nowrap", "vycentrovat", "středem", "korektury", "opraveno",
    "sic", "pozn", "lang", "cizojazyčně", "cj",
}


def _template_text(inner: str) -> str:
    parts = inner.split("|")
    name = parts[0].strip().lower().replace("_", " ")
    if name in {"sic", "opraveno", "korektury"} and len(parts) >= 2:
        return parts[1] if name != "opraveno" else parts[-1]
    if name in _TEXT_TEMPLATES and len(parts) >= 2:
        arg = parts[-1]
        return arg.split("=", 1)[1] if re.match(r"^\s*\w+\s*=", arg) else arg
    return ""


def _link(m: re.Match) -> str:
    target, _, label = m.group(1).partition("|")
    if re.match(r"\s*(soubor|file|image|obrázek|kategorie|category)\s*:", target, re.I):
        return ""
    return label if label else target.split("#")[0]


def clean_wikitext(wikitext: str) -> str:
    t = re.sub(r"<!--.*?-->", "", wikitext, flags=re.S)
    t = re.sub(r"<ref[^>/]*/>", "", t)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S | re.I)
    t = re.sub(r"<(noinclude|references|score|math|gallery)[^>]*>.*?</\1>", "", t, flags=re.S | re.I)
    t = _strip_balanced(t, "{|", "|}")
    t = _strip_balanced(t, "{{", "}}", keep=_template_text)
    t = _strip_balanced(t, "[[", "]]", keep=lambda inner: _link(re.match(r"(.*)", inner, re.S)))
    t = re.sub(r"\[https?://\S+ ([^\]]+)\]", r"\1", t)
    t = re.sub(r"<br\s*/?>\n?", "\n", t, flags=re.I)
    t = re.sub(r"</?(poem|p|div|center|span|small|big|i|b|u|s|sup|sub|font|blockquote)[^>]*>",
               "", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"__[A-Z]+__", "", t)
    lines = []
    for line in t.split("\n"):
        s = line.strip()
        if re.fullmatch(r"-{4,}|[*\s]*", s) and s:
            continue
        m = re.fullmatch(r"=+\s*(.*?)\s*=+", s)
        if m:
            s = m.group(1)
        s = re.sub(r"^[:;#*]+\s*", "", s)
        lines.append(s)
    return normalize_whitespace(html.unescape("\n".join(lines)))


class _HTMLText(HTMLParser):
    """Extract running text from a parsed Wikisource page."""

    SKIP_TAGS = {"table", "style", "script", "sup", "ol", "math", "figure"}
    SKIP_CLASSES = ("noprint", "ws-noexport", "pagenum", "mw-editsection", "reference",
                    "navigace", "textinfo", "metadata", "mw-references", "ws-header")
    BLOCK = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dd", "dt", "tr",
             "blockquote", "center"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        if self.skip_stack:
            if tag not in {"br", "img", "hr", "wbr", "meta", "link", "input"}:
                self.skip_stack.append(tag)
            return
        if tag in self.SKIP_TAGS or any(c in cls.lower() for c in self.SKIP_CLASSES):
            if tag not in {"br", "img", "hr"}:
                self.skip_stack.append(tag)
            return
        if tag in self.BLOCK:
            self.parts.append("\n\n" if tag != "br" else "\n")

    def handle_endtag(self, tag):
        if self.skip_stack:
            self.skip_stack.pop()
            return
        if tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.skip_stack:
            self.parts.append(data)


def clean_html(markup: str) -> str:
    p = _HTMLText()
    p.feed(markup)
    text = "".join(p.parts)
    # paragraphs from HTML keep single newlines inside; join wrapped lines of prose
    text = re.sub(r"(?<=\S)\n(?=\S)", " ", text)
    return normalize_whitespace(text)


def parse_textinfo(wikitext: str) -> dict:
    m = re.search(r"\{\{\s*Textinfo(.*?)\n\}\}", wikitext, flags=re.S | re.I)
    info = {}
    if not m:
        return info
    for line in m.group(1).split("\n"):
        mm = re.match(r"\s*\|\s*([^=|]+?)\s*=\s*(.*)$", line)
        if mm:
            val = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", mm.group(2)).strip()
            info[mm.group(1).strip().upper()] = val
    return info


def licence_ok(info: dict) -> bool:
    """Public domain by Wikisource's own tags; a translation needs a PD translation tag."""
    if info.get("LICENCE") not in PD_LICENCES:
        return False
    if info.get("PŘELOŽIL") or info.get("PŘEKLADATEL"):
        return info.get("LICENCE-PŘEKLAD") in PD_LICENCES
    return True


def read_fairy_tales(jsonl_path: Path) -> tuple[list[dict], list[dict]]:
    """Return (documents, source records). Every kept page is recorded with its URL."""
    docs, sources = [], []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("missing") or _excluded(rec["title"]):
                continue
            info = parse_textinfo(rec["wikitext"])
            status = "kept"
            if not licence_ok(info):
                status = "excluded: licence " + (info.get("LICENCE") or "missing")
                text = ""
            elif re.search(r"\{\{\s*(Rozcestník|Rozcestnik)", rec["wikitext"], re.I):
                status, text = "excluded: disambiguation", ""
            else:
                text = clean_html(rec["html"]) if rec.get("html") else clean_wikitext(rec["wikitext"])
                if len(text) < 300:
                    status = "excluded: too short"
            title = rec["title"].rsplit("/", 1)[-1]
            sources.append({
                "title": rec["title"],
                "url": WIKISOURCE_PAGE + rec["title"].replace(" ", "_"),
                "revid": rec.get("revid"),
                "author": info.get("AUTOR", ""),
                "translator": info.get("PŘELOŽIL") or info.get("PŘEKLADATEL", ""),
                "licence": info.get("LICENCE", ""),
                "licence_translation": info.get("LICENCE-PŘEKLAD", ""),
                "status": status,
                "chars": len(text) if status == "kept" else 0,
            })
            if status == "kept":
                docs.append({"id": "ws:" + rec["title"], "source": "fairy",
                             "text": normalize_whitespace(title + "\n\n" + text)})
    return docs, sources


# --------------------------------------------------------------------------- mixing


def is_validation(doc_id: str, fraction: float = VALIDATION_FRACTION) -> bool:
    h = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8], 16)
    return h < fraction * 0x100000000


def fairy_upsample_factor(law_chars: int, fairy_chars: int,
                          target: float = FAIRY_TARGET_SHARE) -> int:
    if fairy_chars == 0:
        return 0
    k = round(target * law_chars / ((1 - target) * fairy_chars))
    return max(FAIRY_MIN_UPSAMPLE, min(FAIRY_MAX_UPSAMPLE, k))


def mix(laws: list[dict], fairy: list[dict], seed: int = 1234) -> dict:
    """Split 1 % of documents for validation, upsample fairy tales, shuffle documents."""
    train_l = [d for d in laws if not is_validation(d["id"])]
    val_l = [d for d in laws if is_validation(d["id"])]
    train_f = [d for d in fairy if not is_validation(d["id"])]
    val_f = [d for d in fairy if is_validation(d["id"])]
    if fairy and not val_f:  # keep at least one fairy tale for validation
        val_f, train_f = train_f[:1], train_f[1:]
    k = fairy_upsample_factor(sum(len(d["text"]) for d in train_l),
                              sum(len(d["text"]) for d in train_f))
    train = train_l + train_f * k
    rng = random.Random(seed)
    rng.shuffle(train)
    val = val_l + val_f
    return {"train": train, "val": val, "upsample": k}


def write_corpus(split_docs: list[dict], path: Path) -> None:
    """One file, documents separated by the end-of-text token."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in split_docs:
            f.write(d["text"])
            f.write(EOT)


def build(data_dir: Path = DATA_DIR) -> dict:
    raw = data_dir / "raw"
    laws = read_laws(raw / "Czech_legal_code.csv")
    fairy, sources = read_fairy_tales(raw / "wikisource.jsonl")
    mixed = mix(laws, fairy)
    corpus = data_dir / "corpus"
    write_corpus(mixed["train"], corpus / "train.txt")
    write_corpus(mixed["val"], corpus / "val.txt")
    (corpus / "wikisource_sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=1), encoding="utf-8")
    stats = {
        "law_documents": len(laws),
        "law_chars": sum(len(d["text"]) for d in laws),
        "fairy_documents": len(fairy),
        "fairy_chars": sum(len(d["text"]) for d in fairy),
        "fairy_upsample": mixed["upsample"],
        "train_documents": len(mixed["train"]),
        "val_documents": len(mixed["val"]),
        "train_chars": sum(len(d["text"]) for d in mixed["train"]),
        "val_chars": sum(len(d["text"]) for d in mixed["val"]),
        "train_fairy_chars": sum(len(d["text"]) for d in mixed["train"] if d["source"] == "fairy"),
    }
    stats["train_fairy_share"] = round(stats["train_fairy_chars"] / max(1, stats["train_chars"]), 4)
    (corpus / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    write_sources_report(sources, ROOT / "docs" / "reports" / "WIKISOURCE_SOURCES.md")
    return stats


def write_sources_report(sources: list[dict], path: Path) -> None:
    kept = [s for s in sources if s["status"] == "kept"]
    lines = [
        "# Wikisource sources",
        "",
        "Generated by `python -m tinylaw.data build`. Every Czech Wikisource page used for",
        "training, with the licence tags Wikisource gives it (author and, for translations,",
        "translator). Pages whose author or translator licence is not a public-domain tag were",
        "excluded automatically.",
        "",
        f"Kept: {len(kept)} pages, {sum(s['chars'] for s in kept)} characters after cleaning. "
        f"Excluded: {len(sources) - len(kept)}.",
        "",
        "| Page | Author | Translator | Licence | Translation licence | Revision |",
        "|---|---|---|---|---|---|",
    ]
    for s in kept:
        title = s["title"].replace("|", "\\|")
        url = s["url"].replace("(", "%28").replace(")", "%29")
        lines.append(f"| [{title}]({url}) | {s['author']} | {s['translator']} | "
                     f"{s['licence']} | {s['licence_translation']} | {s['revid']} |")
    excluded = [s for s in sources if s["status"] != "kept"]
    if excluded:
        lines += ["", "## Excluded", "", "| Page | Reason |", "|---|---|"]
        for s in excluded:
            lines.append(f"| {s['title']} | {s['status']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_laws(raw_dir: Path) -> Path:
    import requests

    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / "Czech_legal_code.csv"
    if out.exists() and out.stat().st_size > 1_000_000:
        return out
    tmp = out.with_suffix(".part")
    with requests.get(LAWS_URL, headers={"User-Agent": USER_AGENT}, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(out)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["download", "build"])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args(argv)
    raw = args.data_dir / "raw"
    if args.command == "download":
        print("laws:", download_laws(raw))
        print("wikisource:", download_wikisource(raw))
    else:
        print(json.dumps(build(args.data_dir), indent=1))


if __name__ == "__main__":
    main()
