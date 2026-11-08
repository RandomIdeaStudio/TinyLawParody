# Design: model, data and training

Agreed with the user on 2026-11-08. Change any decision here only after asking the user.

## Goal

A small language model trained **from scratch** (own tokenizer, own architecture, own weights) that writes funny parody Czech laws. Humour comes from the clash of serious legal style with an absurd topic given by the user, from fairy-tale words leaking into legal text, and from a "craziness" slider (sampling temperature). It is not a chatbot and gives no legal information.

## Data

| Source | What | Size | Licence | Use |
|---|---|---|---|---|
| [roslein/Czech_legal_code](https://huggingface.co/datasets/roslein/Czech_legal_code) (`Czech_legal_code.csv`) | Full texts of 3 080 valid Czech laws, state of 2025-02-21, primary legislation only | 130 MB CSV | GPL-3.0 (dataset); the laws themselves are official works not protected by Czech copyright | Main corpus |
| [Czech Wikisource](https://cs.wikisource.org/wiki/Kategorie:Pohádky), `Kategorie:Pohádky` plus fairy tales by K. J. Erben and B. Němcová | Public-domain Czech fairy tales | about 3–4 MB wikitext before cleaning | Public-domain texts; record every page URL; check translator death dates for translated tales | Mixed in, upsampled 5–10× to about 10–15 % of training tokens |

- Download in the training environment only; never commit data.
- Wikisource: fetch through the MediaWiki API (`action=query`, `prop=revisions` or `action=parse`), send a descriptive User-Agent, keep requests slow (one at a time). Strip templates, headers, footnotes and navigation. Exclude works that are not fairy tales (for example *Babička*).
- Laws: inspect the real header format of the CSV texts first (title, date, "§", "(1)", "a)") and keep it intact; normalise whitespace only. Each document ends with an end-of-text token.
- Hold out about 1 % of documents for validation.

## Tokenizer

Byte-level BPE, vocabulary **8 192**, trained on the mixed corpus (Hugging Face `tokenizers` is fine). It must be easy to port to JavaScript: export vocabulary and merges as JSON, and the JS encoder must give identical ids (tested).

## Model (the user chose the larger variant)

| | |
|---|---|
| Type | decoder-only transformer (GPT-2 style) |
| Layers / width / heads | 6 / 320 / 5 (head size 64) |
| Feed-forward | 4 × width, GELU |
| Positions | learned, context **256** tokens |
| Norm | pre-LayerNorm |
| Embeddings | tied input/output |
| Parameters | about **10 million** |

Keep the architecture plain so the JavaScript forward pass stays small and exact.

## Training (cloud, CPU only)

- Environment: Anthropic-hosted cloud session, about 4 vCPU, 16 GB RAM, 30 GB disk, no GPU.
- Expected data: about 25–35 million tokens; 1–2 passes.
- Expected speed: about 1 000–2 000 tokens/s, so **about 4–9 hours**. These are estimates: measure with a 2-minute benchmark first and report the real ETA. If the ETA is above 10 hours, ask the user (fewer tokens, or accept the time) instead of shrinking the model on your own.
- AdamW, cosine schedule with warm-up, gradient clipping, `torch.set_num_threads(4)`, fp32 on CPU (bf16 autocast only if it is measurably faster and stable).
- Expected memory: about 2–3 GB RAM during training.
- **Resumable:** save a checkpoint (weights, optimiser, schedule, data position, RNG) to `checkpoints/` every 10 minutes and at the end; `--resume` continues exactly. Log train and validation loss with timestamps to `checkpoints/train.log`.
- Run training as a background process (`nohup`) and check on it rarely (about every 10 minutes, short output) to save the user's Claude usage limits.

## Export and the web page

- Export weights as float16 (about 20 MB) plus `tokenizer.json` and `config.json` into `web/model/`. Each committed file must stay under 25 MB.
- Inference in plain JavaScript in a Web Worker; no external services, nothing leaves the visitor's browser.
- Page controls: title of the law (text field), "craziness" slider (temperature about 0.6–1.4 with top-k), Generate, Try again, Stop, language switch (cs/en), About.
- The prompt the page builds must mirror the header format of real laws in the corpus, so the model continues naturally with "§ 1".
- A parity test runs the JS forward pass on a fixed input and compares the logits with PyTorch (tolerance for float16).

## Deliverables of the first version

1. Code, tests, CI, scripts.
2. A trained model in `web/model/`.
3. `docs/reports/TRAINING_REPORT.md`: data sizes, token counts, real speed and duration, loss curve (numbers), 10 sample outputs for fixed titles, known weaknesses.
4. A pull request from a `claude/...` branch to `main`.

## Going public (soon after the first version)

The repository is private now and will be made public by the user. Before that, `RELEASE_CHECKLIST.md` must be complete: licences and credits, disclaimer, `PRIVACY.md` (the page collects nothing), no secrets in history, then GitHub Pages from `web/`.
