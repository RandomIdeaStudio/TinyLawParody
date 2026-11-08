"""Build the prompt for a parody law title.

The prompt mirrors the header of real laws in the corpus (after cleaning), so the model
continues naturally with "§ 1". `web/app.js` builds the same text from the template that
`export.py` writes into `web/model/config.json`; a test checks both agree.
"""

from __future__ import annotations

import re

# Placeholders: {number} {year} {title} {subject} {day} {month} {date} {date_words}
PROMPT_TEMPLATE = (
    "Zákon č. {number}/{year} Sb.\n\n"
    "{title}\n\n"
    "Vyhlášeno {date}, datum účinnosti {date}, částka {part}/{year}\n"
    "Aktuální znění od {date}\n\n"
    "{number}\n\n"
    "ZÁKON\n\n"
    "ze dne {date_words},\n\n"
    "{subject}\n\n"
    "Parlament se usnesl na tomto zákoně České republiky:\n\n"
    "§ 1\n\n"
)

MONTHS_GENITIVE = ["ledna", "února", "března", "dubna", "května", "června", "července",
                   "srpna", "září", "října", "listopadu", "prosince"]


def normalize_title(raw: str) -> tuple[str, str]:
    """Return (title line, subject line) for a user-typed title.

    The same rules are implemented in `web/app.js`:
      - collapse whitespace, strip, drop a trailing full stop;
      - remove a leading "zákon" (any case) and a following comma;
      - a clause starting "kterým/jímž/kterou/jíž" gives "Zákon, kterým ...";
      - a subject starting with a preposition (o, z, ze, k, ke, na, pro) is kept;
      - anything else gets "o " in front (first letter lower-cased).

    "Zákon o povinném spánku" -> ("Zákon o povinném spánku", "o povinném spánku")
    """
    s = re.sub(r"\s+", " ", raw).strip().rstrip(".").strip()
    s = re.sub(r"^zákon\b[,]?\s*", "", s, flags=re.IGNORECASE)
    if not s:
        s = "o ničem"
    first = s.split(" ", 1)[0].lower()
    if first in {"kterým", "jímž", "kterou", "jíž"}:
        subject = first + s[len(first):]
        return "Zákon, " + subject, subject
    if first not in {"o", "ze", "z", "ke", "k", "na", "pro"}:
        subject = "o " + s[0].lower() + s[1:]
    else:
        subject = first + s[len(first):]
    return "Zákon " + subject, subject


def build_prompt(raw_title: str, number: int = 1, year: int = 2026, day: int = 1,
                 month: int = 4, part: int = 1) -> str:
    title, subject = normalize_title(raw_title)
    return PROMPT_TEMPLATE.format(
        number=number, year=year, title=title, subject=subject, part=part,
        date=f"{day}. {month}. {year}",
        date_words=f"{day}. {MONTHS_GENITIVE[month - 1]} {year}",
    )
