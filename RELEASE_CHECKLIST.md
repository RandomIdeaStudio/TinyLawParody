# Release checklist

Based on the studio template `RELEASE_CHECKLIST_TEMPLATE.md`, reduced to what applies to a static web page with a small model. Store, mobile, monetisation, accounts, servers and multiplayer sections of the template are **N/A**: there is no app, no store, no server, no account and no payment.

Going public (making the repository public and turning on GitHub Pages) is allowed only when every item is `[x]` or `N/A` with a reason.

| | |
|---|---|
| Project | TinyLawParody |
| Version | 0.1.0 |
| Type | web page + model |
| Platform | Web (GitHub Pages) |
| Date | 2026-11-08 (review before going public) |
| Owner | Random Idea Studio |

## 1. Licence and law
- [x] `LICENSE` is GPL-3.0; `README.md` says GPL-3.0-only and © 2026 Random Idea Studio
- [x] `THIRD_PARTY_NOTICES.md` lists every dependency (Python packages, any JS) with its licence
- [x] Every data source credited with its licence; every Wikisource page URL recorded (`docs/reports/WIKISOURCE_SOURCES.md`)
- [x] Translated fairy tales: kept only when Wikisource tags the translation public domain too
- [x] The parody disclaimer is always visible above the conversation and in About
- [x] Sample outputs reviewed: no real person named in an offensive way (real signatures such as "Havel v. r." can appear, copied from real laws; see the training report)

## 2. Privacy
- [x] `PRIVACY.md`: the page collects nothing, runs in the browser, the conversation is not stored, only the language choice is kept in `localStorage`
- [x] No analytics, no external requests except the page's own files (checked by a test)

## 3. Code and build
- [x] `scripts/test.sh` / `scripts/test.ps1` and CI green
- [x] Version in `pyproject.toml`, `CHANGELOG.md` and the page's About section match (checked by a test)
- [x] No `TODO`/`FIXME` blocking the release
- [x] Each committed model file under 25 MB; no datasets or checkpoints in git history (checked on 2026-11-08: largest blob `web/model/model.bin` 19.3 MiB)
- [x] No secrets in history (manual search of all commits on 2026-11-08: no tokens, keys, passwords or personal e-mail addresses; authors use no-reply addresses)

## 4. Stability and performance
- [ ] Generation works in current Chrome, Firefox and Safari (desktop) and one phone browser. Tested so far: Chromium (desktop and 375 px phone emulation) and the owner's desktop browser
- [x] The page stays responsive while generating (Web Worker); Stop works
- [ ] Memory measured on a phone; no growth over 20 generations

## 5. Accessibility and localisation
- [x] All page texts in `web/locales/cs.json` and `en.json`, key parity test green
- [ ] Contrast, focus outlines, 48 px targets, text scales with browser font size. Done: 48 px targets, focus outlines, rem-based text; contrast not measured with a tool
- [x] Screen reader labels on controls; the finished reply is announced politely
- [x] Light and dark mode

## 6. Publishing
- [ ] Repository switched to public by the user
- [ ] GitHub Pages serves `web/` (workflow `.github/workflows/pages.yml`, source "GitHub Actions"); the live URL tested
- [ ] Tag `vX.Y.Z` on `main`

## Final approval
- [ ] All sections `[x]` or `N/A` with a reason
- [ ] Known issues written below and accepted
- [ ] **GO / NO-GO:** ____________  Date: ____________

### Known issues and postponed items

| Item | Reason | Date |
|---|---|---|
| Commit trailers and the PR description link to the private Claude Code session (`claude.ai/code/session_…`) | Only the owner can open it; removing it from commits would rewrite history | 2026-11-08 |
| Glued duplicate words from the law data ("obceobce") | Needs a cleaning change and a new training run | 2026-11-08 |
| The clerk does not remember earlier messages and does not understand meaning | Model size and training data; stated on the page | 2026-11-08 |
