# Third-party notices

TinyLawParody (code and model) is licensed under **GPL-3.0-only**. It uses the data
and software below. The web page itself uses **no third-party JavaScript, fonts or
services**; everything under `web/` is our own code plus the exported model.

## Data

| Source | What we use | Licence | Credit |
|---|---|---|---|
| [roslein/Czech_legal_code](https://huggingface.co/datasets/roslein/Czech_legal_code) | `Czech_legal_code.csv`: full texts of 3 080 valid Czech laws, state of 2025-02-21, scraped from [e-sbirka.cz](https://www.e-sbirka.cz/) | GPL-3.0 (dataset). The laws themselves are official works, not protected by Czech copyright (§ 3 of Act No. 121/2000 Coll.) | Dataset by Hugging Face user roslein |
| [Czech Wikisource](https://cs.wikisource.org/) (Wikizdroje) | Fairy tales from `Kategorie:Pohádky`, its collections, and fairy tales by K. J. Erben and B. Němcová | Public domain. Only pages whose Wikisource licence tag is a public-domain tag are used; for translations the translation must also carry a public-domain tag | `python -m tinylaw.data build` writes every page with its URL, author, translator and revision to `data/WIKISOURCE_SOURCES.md`. Thanks to the Wikisource volunteers who transcribed them |

The datasets are downloaded during training and are not part of the repository. The
trained model in `web/model/` is derived from them and is distributed under GPL-3.0.

## Python packages (training only, not shipped with the page)

| Package | Version | Licence |
|---|---|---|
| torch (CPU) | 2.15.0 | BSD-3-Clause (with bundled third-party components under Apache-2.0, MIT, BSL-1.0, ISC, Zlib) |
| numpy | 2.5.4 | BSD-3-Clause (with bundled components under 0BSD, MIT, Zlib, CC0-1.0) |
| tokenizers | 0.23.3 | Apache-2.0 |
| requests | 2.34.2 | Apache-2.0 |
| huggingface_hub | 2.4.0 | Apache-2.0 |
| hf-xet | 1.8.0 | Apache-2.0 |
| anyio | 4.16.0 | MIT |
| certifi | 2026.10.23 | MPL-2.0 |
| charset-normalizer | 3.5.2 | MIT |
| click | 8.5.1 | BSD-3-Clause |
| filelock | 4.2.9 | MIT |
| fsspec | 2026.9.0 | BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| httpcore2 | 2.16.0 | BSD-3-Clause |
| httpx2 | 2.16.0 | BSD-3-Clause |
| idna | 3.20 | BSD-3-Clause |
| Jinja2 | 3.1.6 | BSD-3-Clause |
| MarkupSafe | 3.0.4 | BSD-3-Clause |
| mpmath | 1.3.0 | BSD-3-Clause |
| networkx | 3.7 | BSD-3-Clause |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| PyYAML | 6.0.3 | MIT |
| setuptools | 84.2.1 | MIT |
| sympy | 1.14.0 | BSD-3-Clause |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| urllib3 | 2.8.0 | MIT |

## JavaScript

None. The tokenizer (`web/tokenizer.js`) and the model forward pass (`web/transformer.js`) are
our own implementations. The byte-to-unicode mapping and the pre-tokenisation pattern
follow the published GPT-2 byte-level BPE scheme.

## Tools used in CI

GitHub Actions `actions/checkout`, `actions/setup-python`, `actions/setup-node` (MIT).
