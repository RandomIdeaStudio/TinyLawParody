from tinylaw import data


LAW = (
    "Zákon č. 6/2019 Sb.\n\nZákon, kterým se mění zákon o dani\n\n"
    "Vyhlášeno 17. 1. 2019, datum účinnosti 1. 2. 2019, částka 3/2019\n"
    " * Čl. I - Změna zákona o dani z přidané hodn\n\n * Čl. II - Účinnost\n"
    "Aktuální znění od 1. 2. 2019\n\n6\n\nZÁKON\n\nze dne 20. prosince 2018,\n\n"
    "§ 1\n\n(1) Text s   mezerami.\r\n\n\n\n(2) Další."
)


def test_clean_law_drops_truncated_table_of_contents_keeps_header():
    out = data.clean_law(LAW)
    assert out.startswith("Zákon č. 6/2019 Sb.\n\nZákon, kterým se mění zákon o dani\n\nVyhlášeno")
    assert " * Čl." not in out
    assert "Účinnost" not in out
    assert "Vyhlášeno 17. 1. 2019, datum účinnosti 1. 2. 2019, částka 3/2019\nAktuální znění" in out
    assert "(1) Text s mezerami.\n\n(2) Další." in out


def test_clean_law_without_toc_is_kept():
    assert data.clean_law("Zákon\n\n§ 1\n\nText") == "Zákon\n\n§ 1\n\nText"


def test_normalize_whitespace():
    assert data.normalize_whitespace(" a  b \n\n\n\nc\t d​ ") == "a b\n\nc d"


WIKITEXT = """{{NavigacePaP
  | TITUL = Sbírka
  | DALŠÍ = Další
}}
{{Textinfo
  | TITULEK = Motýl
  | AUTOR = [[Autor:Hans Christian Andersen|Hans Christian Andersen]]
  | PŘELOŽIL = [[Autor:Josef Boleslavský|Josef Boleslavský]]
  | LICENCE = PD old 70
  | LICENCE-PŘEKLAD = PD old 70
}}
{{Forma|proza}}
== Motýl ==
Byl jeden '''motýl'''<ref>Poznámka.</ref> a [[Květina|květinka]] {{Prostrkaně|sedmikráska}}.<br />
Druhý řádek [[Soubor:Obr.jpg|thumb|x]] konec.
{| class="x"
| tabulka
|}
[[Kategorie:Pohádky]]
"""


def test_clean_wikitext():
    out = data.clean_wikitext(WIKITEXT)
    assert out == "Motýl\nByl jeden motýl a květinka sedmikráska.\nDruhý řádek konec."


def test_textinfo_and_licence():
    info = data.parse_textinfo(WIKITEXT)
    assert info["AUTOR"] == "Hans Christian Andersen"
    assert info["PŘELOŽIL"] == "Josef Boleslavský"
    assert data.licence_ok(info)
    assert not data.licence_ok({**info, "LICENCE-PŘEKLAD": "CC-BY-SA-4.0"})
    assert not data.licence_ok({"LICENCE": "CC-BY-SA-4.0"})
    assert data.licence_ok({"LICENCE": "PD old 70"})


def test_clean_html_skips_page_numbers_and_notes():
    markup = ('<div class="mw-parser-output"><table class="textinfo"><tr><td>X</td></tr></table>'
              '<p>Byl jeden <span class="pagenum ws-pagenum">[12]</span>král<sup class="reference">'
              '[1]</sup>.</p><p>Konec.</p><ol class="references"><li>pozn</li></ol></div>')
    assert data.clean_html(markup) == "Byl jeden král.\n\nKonec."


def test_excluded_titles():
    assert data._excluded("Babička")
    assert data._excluded("Babička/Kapitola I")
    assert not data._excluded("Zlatovláska")


def test_mix_split_and_upsample():
    laws = [{"id": f"law:{i}", "source": "law", "text": "x" * 1000} for i in range(500)]
    fairy = [{"id": f"ws:{i}", "source": "fairy", "text": "y" * 1000} for i in range(10)]
    m = data.mix(laws, fairy)
    val_ids = {d["id"] for d in m["val"]}
    train_ids = {d["id"] for d in m["train"]}
    assert not (val_ids & train_ids)
    assert any(i.startswith("ws:") for i in val_ids)
    assert 1 <= m["upsample"] <= data.FAIRY_MAX_UPSAMPLE
    assert len(m["val"]) < 0.05 * (len(laws) + len(fairy))
    # deterministic
    assert [d["id"] for d in data.mix(laws, fairy)["train"]] == [d["id"] for d in m["train"]]


def test_upsample_factor_targets_share():
    assert data.fairy_upsample_factor(880, 12) == 10  # capped
    assert data.fairy_upsample_factor(880, 120) == 1
    assert data.fairy_upsample_factor(100, 0) == 0
