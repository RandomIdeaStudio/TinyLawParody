# TinyLawParody

A tiny language model, trained from scratch, that writes **parody Czech laws**. The page is a chat with a clerk: write anything and the clerk answers with a dead-serious legal paragraph (in Czech). Everything runs in your browser; nothing is sent anywhere.

> **Disclaimer.** All generated text is machine-made parody. It is not law, not a quotation of any law and not legal information.

**Version:** 0.1.0.

## How it works

- About 10 million parameters, a small GPT-style transformer with its own byte-level BPE tokenizer.
- Trained on CPU on the texts of Czech laws mixed with public-domain Czech fairy tales.

## Quick start

The page needs only a static web server:

```bash
python3 -m http.server -d web 8000    # then open http://localhost:8000
```

Training needs Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Full training pipeline (downloads data, trains with resume, exports to `web/model/`):

```bash
.venv/bin/python -m tinylaw.data download
.venv/bin/python -m tinylaw.data build
.venv/bin/python -m tinylaw.tokenizer train
.venv/bin/python -m tinylaw.tokenizer encode
.venv/bin/python -m tinylaw.train --epochs 2 --bf16 --resume
.venv/bin/python -m tinylaw.export
```

Command-line generation: `.venv/bin/python -m tinylaw.sample "Zákon o povinném odpoledním spánku"`.

## Data sources

| Source | Licence |
|---|---|
| [roslein/Czech_legal_code](https://huggingface.co/datasets/roslein/Czech_legal_code): full texts of Czech laws (state of 2025-02-21) | GPL-3.0 |
| [Czech Wikisource](https://cs.wikisource.org/): public-domain Czech fairy tales; the build step lists every page used in `data/WIKISOURCE_SOURCES.md` | Public domain texts |

Datasets are downloaded during training and are not part of this repository. Details and software licences: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Privacy: [`PRIVACY.md`](PRIVACY.md) (the page collects nothing).

## Licence

GPL-3.0-only, see [`LICENSE`](LICENSE). © 2026 Random Idea Studio.
