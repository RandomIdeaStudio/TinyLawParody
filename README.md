# TinyLawParody

A tiny language model, trained from scratch, that writes **parody Czech laws**. Type an absurd title and the model continues in dead-serious legal style. Everything runs in your browser; nothing is sent anywhere.

> **Disclaimer.** All generated text is machine-made parody. It is not law, not a quotation of any law and not legal information.

**Status:** first version (0.1.0), in development. Training results: [`docs/reports/TRAINING_REPORT.md`](docs/reports/TRAINING_REPORT.md).

## How it works

- About 10 million parameters, a small GPT-style transformer with its own byte-level BPE tokenizer.
- Trained on CPU on the texts of Czech laws mixed with public-domain Czech fairy tales.
- Design and numbers: [`docs/DESIGN_MODEL.md`](docs/DESIGN_MODEL.md).

## Quick start

Python 3.12 and Node.js 22 are needed.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
scripts/test.sh                       # whole test suite (Windows: scripts/test.ps1)
python3 -m http.server -d web 8000    # then open http://localhost:8000
```

Full training pipeline (downloads data, trains with resume, exports to `web/model/`):

```bash
nohup scripts/train-cloud.sh > checkpoints/pipeline.log 2>&1 &
```

Command-line generation: `.venv/bin/python -m tinylaw.sample "Zákon o povinném odpoledním spánku"`.

## Data sources

| Source | Licence |
|---|---|
| [roslein/Czech_legal_code](https://huggingface.co/datasets/roslein/Czech_legal_code): full texts of Czech laws (state of 2025-02-21) | GPL-3.0 |
| [Czech Wikisource](https://cs.wikisource.org/): public-domain Czech fairy tales; every page is listed in [`docs/reports/WIKISOURCE_SOURCES.md`](docs/reports/WIKISOURCE_SOURCES.md) | Public domain texts |

Datasets are downloaded during training and are not part of this repository. Details and software licences: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Privacy: [`PRIVACY.md`](PRIVACY.md) (the page collects nothing).

## Licence

GPL-3.0-only, see [`LICENSE`](LICENSE). © 2026 Random Idea Studio.
