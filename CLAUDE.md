# TinyLawParody

A tiny Czech language model trained from scratch on Czech laws mixed with public-domain Czech fairy tales. It writes parody laws: the user types an absurd title ("Zákon o povinném odpoledním spánku") and the model continues in dead-serious legal style. The chat runs entirely in the browser. Python + PyTorch for training, plain JavaScript for the web page. The plan, numbers and decisions are in `docs/DESIGN_MODEL.md`.

This is an experiment, not a game, and an exception to the studio's Godot rules (no Godot, no Android, no store).

## Studio rules (apply to every task in this project)
1. **Talk to the user in Czech, always.** Code, comments, identifiers, commit messages, branch names and technical documentation are in **English**.
2. **No user-visible text in code.** Every text on the web page lives in `web/locales/<language>.json` and is read through the page's `t("key")` helper. Minimum languages: Czech and English; a test checks every key exists in every language. The model's own output is Czech and is not translated. Developer-only logs and exception messages may stay English literals.
3. **Ask before guessing.** If anything is open (design choice, scope, naming, trade-off, anything hard to undo), ask the user first. Do the clear parts, ask about the open ones.
4. **Verify before saying "done".** Run `scripts/test.sh` (Linux, cloud, CI) or `scripts/test.ps1` (Windows). If something fails or was not run, say so plainly.
5. **Git hygiene.** Work on a branch, never force-push, never rewrite history, ask before pushing to `main`. Never commit downloaded datasets (`data/`), training checkpoints (`checkpoints/`), virtual environments, passwords or API keys. The repository goes **public** soon after the first version, so assume everything committed will be visible to anyone, including history.
6. **Pinned versions.** Python **3.12**; every Python dependency is pinned exactly in `requirements.txt`. Do not upgrade without asking.
7. **Release.** Before the repository is made public or a version is tagged, follow `RELEASE_CHECKLIST.md`; nothing is ready while items are open.
8. **Keep docs true.** When a rule, command or structure changes, update this file in the same change.
9. **Accessibility baseline.** Never rely on colour alone, touch targets at least 48 px, text scales with the browser font size, no flashing above 3 Hz.
10. **Entry points.** `scripts/test.sh` and `scripts/test.ps1` run the whole test suite with one command; CI is `.github/workflows/ci.yml`. CI never trains the real model: tests use a tiny model on a few lines of synthetic text.
11. **Licence.** The project (code and model) is **GPL-3.0-only** because the law dataset is GPL-3.0. Every data source is credited in `README.md` and `THIRD_PARTY_NOTICES.md`.
12. **Branches and releases.** `main` is always releasable. Work on short-lived branches (`claude/...`, `feature/...`) and merge through pull requests. A release is a tag `vX.Y.Z` on `main`; the version lives only in `pyproject.toml` and must match `CHANGELOG.md` and the About section of the page (which reads it from `web/model/config.json` or a generated `web/version.json`).
13. **Repository files.** `README.md`, `LICENSE`, `CHANGELOG.md`, `RELEASE_CHECKLIST.md`, `.editorconfig`; before going public also `PRIVACY.md` and `THIRD_PARTY_NOTICES.md`. Layout: `scripts/`, `tests/`, `docs/`, `web/`; generated reports go to `docs/reports/`; `docs/README.md` is the index.
14. **Stored values.** Numbers in configs, logs and reports are written locale-independently (dot as decimal separator).
15. **Default language of the page.** The language the visitor chose (kept in `localStorage`, wrapped in try/catch); without one the browser language if the page has it, otherwise English.
16. **About section.** The page has one: version, copyright, licence (GPL-3.0), data sources and their licences, and the parody disclaimer. Texts come from the language files.
17. **Parody disclaimer.** The page always shows, next to the output, that the text is machine-generated parody, not law and not legal information.
18. **Start every session with `git pull`** (and look at `git status`). If there are uncommitted local changes or the pull would conflict, stop and ask the user.
19. **End every task with the sync state.** Say plainly whether the changes are committed and pushed, and to which branch.

## Structure (target; build it as described in `docs/DESIGN_MODEL.md`)

- `tinylaw/`: the Python package. `data.py` (download, clean, mix), `tokenizer.py` (byte-level BPE), `model.py` (GPT), `train.py` (resumable training), `export.py` (writes `web/model/`), `sample.py` (command-line generation).
- `web/`: the static page for GitHub Pages. `index.html`, `app.js`, `worker.js` (inference off the main thread), `locales/cs.json`, `locales/en.json`, `model/` (exported weights, tokenizer and config; the only model files that are committed).
- `tests/`: pytest tests and the JavaScript parity test.
- `scripts/`: `test.sh`, `test.ps1`, `train-cloud.sh` (the full pipeline with resume).
- `data/` and `checkpoints/`: local only, ignored by git.
