# TinyLawParody

A tiny language model, trained from scratch, that writes **parody Czech laws**. Type an absurd title and the model continues in dead-serious legal style. Everything runs in your browser; nothing is sent anywhere.

> **Disclaimer.** All generated text is machine-made parody. It is not law, not a quotation of any law and not legal information.

**Status:** in development. The model is not trained yet.

## How it works

- About 10 million parameters, a small GPT-style transformer with its own byte-level BPE tokenizer.
- Trained on CPU on the texts of Czech laws mixed with public-domain Czech fairy tales.
- Design and numbers: [`docs/DESIGN_MODEL.md`](docs/DESIGN_MODEL.md).

## Data sources

| Source | Licence |
|---|---|
| [roslein/Czech_legal_code](https://huggingface.co/datasets/roslein/Czech_legal_code): full texts of Czech laws (state of 2025-02-21) | GPL-3.0 |
| [Czech Wikisource](https://cs.wikisource.org/): public-domain Czech fairy tales | Public domain texts |

Datasets are downloaded during training and are not part of this repository.

## Licence

GPL-3.0-only, see [`LICENSE`](LICENSE). © 2026 Random Idea Studio.
