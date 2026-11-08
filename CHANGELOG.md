# Changelog

Versions follow SemVer. The version lives only in `pyproject.toml`.

## [0.1.0] – unreleased

- Project set up: plan, licence and repository rules.
- Data pipeline: Czech laws (Hugging Face) and public-domain fairy tales (Czech Wikisource), cleaning, mixing, validation split.
- Byte-level BPE tokenizer (8 192), GPT model (about 10 M parameters), resumable CPU training, float16 export.
- Web page: inference in a Web Worker, Czech and English, craziness slider, Stop, About, parody disclaimer.
- `THIRD_PARTY_NOTICES.md`, `PRIVACY.md`.
- The page is a chat with a clerk who answers every message with a legal paragraph; Answer differently, New conversation, examples, settings.
- The model's script is `transformer.js` (ad blockers blocked `gpt.js`, so the model never loaded).
