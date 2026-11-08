"""The chat with the clerk (web/chat.js): request building and reply formatting."""

from conftest import run_node

from tinylaw.prompt import normalize_title


def request(message: str, random: float = 0.0) -> dict:
    return run_node({"cmd": "chat_request", "messages": [message], "random": random})[0]


def fmt(suffix: str, generated: str, reason: str = "end") -> str:
    return run_node({"cmd": "chat_format", "cases": [[suffix, generated, reason]]})[0]


def test_request_wraps_message_as_law_title():
    r = request("Ahoj, jak se máš?")
    assert r["message"] == "Ahoj, jak se máš?"
    assert r["title"] == "o vyřízení podání „Ahoj, jak se máš?“"
    # the title starts with a preposition, so the shared prompt builder keeps it as it is
    assert normalize_title(r["title"]) == ("Zákon " + r["title"], r["title"])


def test_reply_start_quotes_message_and_has_no_trailing_space():
    openings = run_node({"cmd": "chat_openings"})
    for i, opening in enumerate(openings):
        r = request("Kde jsou moje klíče?", random=(i + 0.5) / len(openings))
        assert r["promptSuffix"] == "(1)\n\nPodání „Kde jsou moje klíče?“ " + opening
        assert not r["promptSuffix"].endswith((" ", "\n"))


def test_message_is_cleaned():
    r = request('  Řekni „ano“\n\na "ne"  ')
    assert r["message"] == "Řekni ano a ne"
    long = request("slovo " * 60)
    assert len(long["message"]) <= 121 and long["message"].endswith("…")


def test_reply_cut_after_two_sentences():
    suffix = "(1)\n\nPodání „Ahoj?“ se zamítá, neboť"
    out = fmt(suffix, " není přípustné. Úřad rozhodne do 30 dnů. Třetí věta tu nemá co dělat.")
    assert out == "Podání „Ahoj?“ se zamítá, neboť není přípustné. Úřad rozhodne do 30 dnů."


def test_question_mark_in_message_is_not_a_sentence_end():
    suffix = "(1)\n\nPodání „Jak se máš? Dobře.“ se považuje za"
    out = fmt(suffix, " podepsanou.")
    assert out == "Podání „Jak se máš? Dobře.“ se považuje za podepsanou."


def test_abbreviations_do_not_end_a_sentence():
    suffix = "(1)\n\nPodání „Volno?“ se vyřizuje do 30 dnů, a to"
    out = fmt(suffix, " podle § 5 odst. 1 písm. a) zákona č. 12/2026 Sb. Další věta. A ještě jedna.")
    assert out == ("Podání „Volno?“ se vyřizuje do 30 dnů, a to podle § 5 odst. 1 písm. a) "
                   "zákona č. 12/2026 Sb. Další věta.")


def test_stray_closing_quote_is_dropped():
    suffix = "(1)\n\nPodání „Ahoj, jak se máš?“ se postoupí příslušnému úřadu, který"
    assert fmt(suffix, " je účastníkem?“") == (
        "Podání „Ahoj, jak se máš?“ se postoupí příslušnému úřadu, který je účastníkem.")
    # a quotation the model opens and closes itself stays
    assert fmt(suffix, " vede „seznam“.") == (
        "Podání „Ahoj, jak se máš?“ se postoupí příslušnému úřadu, který vede „seznam“.")


def test_unfinished_reply_ends_with_ellipsis():
    suffix = "(1)\n\nPodání „Volno?“ se vyřizuje do 30 dnů, a to"
    out = fmt(suffix, " podle § 5 odst. 1, § 6 odst. 2 a podle", reason="length")
    assert out.endswith("…")
    assert out.startswith("Podání „Volno?“ se vyřizuje do 30 dnů, a to podle")
