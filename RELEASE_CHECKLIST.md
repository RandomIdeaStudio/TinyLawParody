# Release checklist

Based on the studio template `RELEASE_CHECKLIST_TEMPLATE.md`, reduced to what applies to a static web page with a small model. Store, mobile, monetisation, accounts, servers and multiplayer sections of the template are **N/A**: there is no app, no store, no server, no account and no payment.

Going public (making the repository public and turning on GitHub Pages) is allowed only when every item is `[x]` or `N/A` with a reason.

| | |
|---|---|
| Project | TinyLawParody |
| Version | 0.1.0 |
| Type | web page + model |
| Platform | Web (GitHub Pages) |
| Date | |
| Owner | Random Idea Studio |

## 1. Licence and law
- [ ] `LICENSE` is GPL-3.0; `README.md` says GPL-3.0-only and © 2026 Random Idea Studio
- [ ] `THIRD_PARTY_NOTICES.md` lists every dependency (Python packages, any JS) with its licence
- [ ] Every data source credited with its licence; every Wikisource page URL recorded
- [ ] Translated fairy tales: translator died more than 70 years ago, or the tale is removed
- [ ] The parody disclaimer is visible next to every output and in About
- [ ] Sample outputs reviewed: no real person named in an offensive way

## 2. Privacy
- [ ] `PRIVACY.md`: the page collects nothing, runs offline in the browser, only the language choice is kept in `localStorage`
- [ ] No analytics, no external requests except the page's own files

## 3. Code and build
- [ ] `scripts/test.sh` and CI green
- [ ] Version in `pyproject.toml`, `CHANGELOG.md` and the page's About section match
- [ ] No `TODO`/`FIXME` blocking the release
- [ ] Each committed model file under 25 MB; no datasets or checkpoints in git history
- [ ] No secrets in history (`gitleaks` or a manual search)

## 4. Stability and performance
- [ ] Generation works in current Chrome, Firefox and Safari (desktop) and one phone browser
- [ ] The page stays responsive while generating (Web Worker); Stop works
- [ ] Memory measured on a phone; no growth over 20 generations

## 5. Accessibility and localisation
- [ ] All page texts in `web/locales/cs.json` and `en.json`, key parity test green
- [ ] Contrast, focus outlines, 48 px targets, text scales with browser font size
- [ ] Screen reader labels on controls; output announced politely
- [ ] Light and dark mode

## 6. Publishing
- [ ] Repository switched to public by the user
- [ ] GitHub Pages serves `web/`; the live URL tested
- [ ] Tag `vX.Y.Z` on `main`

## Final approval
- [ ] All sections `[x]` or `N/A` with a reason
- [ ] Known issues written below and accepted
- [ ] **GO / NO-GO:** ____________  Date: ____________

### Known issues and postponed items

| Item | Reason | Date |
|---|---|---|
| | | |
