# Training report

First training run of TinyLawParody, 2026-11-08, in an Anthropic-hosted cloud session
(4 vCPU, 16 GB RAM, no GPU). Plan and decisions: [`../DESIGN_MODEL.md`](../DESIGN_MODEL.md).

> All generated text below is machine-made parody. It is not law and not legal information.

## Summary

| | |
|---|---|
| Model | GPT-2 style decoder, 6 layers, width 320, 5 heads, context 256, tied embeddings |
| Parameters | 10 101 760 |
| Tokenizer | byte-level BPE, vocabulary 8 192 |
| Training tokens | 36 116 182 per pass, 2 passes (8 816 steps × 32 × 256 = 72.2 M tokens) |
| Precision | bf16 autocast on CPU (measured faster and stable, see below) |
| Duration | 2.06 h of training steps; wall clock 15:36–17:42 UTC (2 h 06 min) |
| Speed | median 9 776 tokens/s (min 8 946, max 10 334) |
| Final loss | train 2.053, validation 1.933 (best 1.931) |
| Export | `web/model/model.bin` 20.2 MB (float16), `tokenizer.json` 0.19 MB, `config.json` |

## Data

| | Documents | Characters | Notes |
|---|---:|---:|---|
| Czech laws ([roslein/Czech_legal_code](https://huggingface.co/datasets/roslein/Czech_legal_code)) | 3 080 | 119 333 408 | state of 2025-02-21 |
| Fairy tales (Czech Wikisource) | 679 | 5 102 233 | after cleaning; 708 pages fetched, 29 excluded (licence tag missing or not public domain, disambiguation, too short) |
| Training split | 5 070 | 133 736 283 | fairy tales upsampled 3× → 11.35 % of training characters |
| Validation split | 33 | 814 962 | 1 % of documents chosen by hash of the document id |

- Every Wikisource page with author, translator, licence tags and revision: [`WIKISOURCE_SOURCES.md`](WIKISOURCE_SOURCES.md).
  Pages are kept only if Wikisource tags them public domain (`PD old 70`, `PD traditional`, …) and, for translations, also tags the translation public domain.
- Law cleaning: the header (number, title, "Vyhlášeno …", "Aktuální znění od …") and the body are kept; the table of contents that the CSV repeats at the start (lines ` * § N - …`, truncated mid-word, about 2 % of characters) is dropped; whitespace is normalised.
- Tokens: 36 116 182 training, 223 020 validation (3.70 characters per token).
- Wikisource rate-limits the shared cloud IP (HTTP 429); the polite client (one request at a time, honouring `Retry-After`) needed about 45 minutes for 708 pages.

## Speed measurement and the bf16 decision

A 45-second benchmark on random tokens (batch 32 × 256, 4 threads):

| Precision | Tokens/s | Seconds/step |
|---|---:|---:|
| fp32 | 5 474 | 1.50 |
| bf16 autocast | 10 048 | 0.82 |

Stability check on real law text, 200 steps each, same seed: fp32 validation loss 5.1345, bf16 5.1311.
bf16 was therefore used. Estimated time before the run: 2 passes ≈ 2.1 h (below the 10-hour limit); actual 2.06 h.

## Loss curve

Validation loss on the first 20 validation batches (164 k tokens), train loss averaged over the last 50 steps.
Times are UTC.

| Step | Time | Learning rate | Train loss | Validation loss |
|---:|---|---:|---:|---:|
| 250 | 15:39 | 1.00e-03 | 4.941 | 4.874 |
| 500 | 15:43 | 9.97e-04 | 4.107 | 3.960 |
| 1 000 | 15:50 | 9.81e-04 | 3.467 | 3.335 |
| 1 500 | 15:57 | 9.50e-04 | 3.128 | 2.959 |
| 2 000 | 16:04 | 9.07e-04 | 2.911 | 2.737 |
| 2 500 | 16:11 | 8.51e-04 | 2.799 | 2.596 |
| 3 000 | 16:18 | 7.85e-04 | 2.698 | 2.478 |
| 3 500 | 16:25 | 7.12e-04 | 2.577 | 2.369 |
| 4 000 | 16:32 | 6.33e-04 | 2.529 | 2.278 |
| 4 500 | 16:39 | 5.51e-04 | 2.381 | 2.221 |
| 5 000 | 16:47 | 4.70e-04 | 2.342 | 2.150 |
| 5 500 | 16:54 | 3.91e-04 | 2.283 | 2.115 |
| 6 000 | 17:01 | 3.17e-04 | 2.238 | 2.057 |
| 6 500 | 17:09 | 2.51e-04 | 2.254 | 2.035 |
| 7 000 | 17:16 | 1.95e-04 | 2.157 | 1.998 |
| 7 500 | 17:23 | 1.51e-04 | 2.181 | 1.966 |
| 8 000 | 17:31 | 1.20e-04 | 2.152 | 1.942 |
| 8 500 | 17:38 | 1.03e-04 | 2.103 | 1.933 |
| 8 816 | 17:42 | 1.00e-04 | 2.053 | 1.933 |

Validation loss is lower than train loss because the validation split is 99 % laws, while the training mix
contains 11 % fairy tales, which are harder to predict. The curve is still falling slowly at the end; more
passes or a larger context would probably help a little.

## 10 sample outputs

Exported float16 model, prompt built by `tinylaw/prompt.py` (the same header the page uses), temperature 0.9,
top-k 40, 220 new tokens, seed = sample number − 1. Only the generated continuation after "§ 1" is shown.
Reproduce with `python -m tinylaw.sample "<title>" --model web/model --seed N --max-new-tokens 220`.

### 1. Zákon o povinném odpoledním spánku

```text
Zdravotnické zařízení je oprávněna uskutečňovat jen akreditované zařízeníakreditované zařízení, kterému byla vydána, za předpokladu, že akreditovaná zařízeníakreditovaná zařízení uskutečňuje akreditované zařízeníakreditované zařízení, které byly uskutečňovány.

§ 2

Zdravotnické zařízení

(1)

Oborové zařízeníakreditované zařízení rozhoduje o akreditovaném zařízeníakreditovaném zařízení, který uskutečňuje uskutečňování akreditovaného zařízeníakreditovaného zařízení.

(2)

Osoba akreditující zařízeníakreditující zařízení, které byla vydána,

a)

na němž byla akreditována progistické studijní studijní program s akreditčním vzdělávacím programem, která je akreditačním orgánemakreditovaným zařízení držitelem povolení Ministerstvo zdravotnictví, který hodlá uskutečňovat akreditovaný vzdělávací program pro oblast odborné přípravy pro poskytování zdravotních služeb, které mu bylo vydáno,

b)

poskytuje ministerstvu informace na žádost akreditovaném zařízeníakreditovaném zařízení, které mu bylo vydáno, nebo které mu bylo vydáno,

c)
```

### 2. Zákon o ochraně draků před princeznami

```text
Zákonný zástupce

(1)

Každý je povinen vstoupit do řízení a vstoupit do řízení před správou národního podniku podnět k provedení úkonu zabraného do správy národního podniku, aby soud přihlásil, že se nebude řádně ujit nebo kdy se s jeho správou řídí; k provedení úkonu je třeba přihlížet.

(2)

Není-li vstoupit do řízení a ten je povinen je ten, komu a ten má být úkon proveden, bez zbytečného odkladu vyrozumět, že jsou k němu oprávněnými právnickými a fyzickými nebo právnickými osobami, a to v rozsahu vyplývajícím ze právních předpisů a v případě, že o nich jedná bez souhlasu státního zástupce, rozhodne zároveň o jeho zrušení a o změně nebo zrušení rozhodnutí. Obecní úřad vyrozumí rovněž, zda se řízení přikládá před soudem, kterému byl úkon podán, s uvedením důvodu a účelu, kterého se připouští; za tím účelem je oprávněn podat návrh, byl-li ustanoven, o jeho zrušení vyrozuměn.

(3)

Soud, který
```

### 3. Zákon, kterým se zakazuje pondělí

```text
Víde-li o cizinci, který je osobou, jejíž zastupitelský úřad vykonává pro výkon působnosti podle tohoto zákona, je jeho součástí, pokud je jeho součástí se souhlasem Sboru nápravné výchovy České republiky, a pokud je jeho součástí ministerstvo.

§ 2

Dávky

(1)

Oprávnění k pobytu občana Sboru nápravné výchovy České republiky (dále jen „Rada“), která je jeho součástí (dále jen „příslušný útvar policie“), může podat žádost o udělení víza za účelem pobytu za účelem pobytu v jiném členském státě. Podací žádost lze podat pouze na výstup z informačního systému veřejné správy.

(2)

Ze zdravotního pojištění lze podat pouze na výstup z informačního systému veřejné správy a z vlastního informačního systému. V tomto případě se žádost o udělení dočasné ochranyŽadatel o prodloužení doby platnosti povolení k dlouhodobému pobytu za účelem ochrany na území obsahuje údaje uvedené v § 18 odst. 2.

(3)

Dávky cizinců podle odstavce 1 a osoby, které ke dni podání žádosti o udělení dočasné ochrany došlo za účelem poskytnutí
```

### 4. Zákon o státní kontrole perníkových chaloupek

```text
(1)

Česká národní banka může stanovit periodické chatalogu periodický chata, pokud

a)

sloužící a rudímní periodický chata v rámci biální onkologicky nebo rozdělená opatření k omezení spotřeby paliva,

b)

sloužící a rudímní periodický chata pro použití hlušin, nebo

c)

sloužící a rudímní periodického chování a

d)

sloužící a rudímní periodický choftal pro použití hlušin, pokud není derivátový chatovor.

(2)

Česká národní banka může stanovit periodický chatrrgickou a rudímní periodický chatativní záložna. Vymezení hrudivé hodnoty se prokazuje zejména

a)

náboře členů etické kodex
```

### 5. Zákon o registraci vodníků a hastrmanů

```text
(1)

Volební období se na kandidátní listině zapisují tyto údaje do matriční knihy v rozsahu

a)

jméno, popřípadě jména, a příjmení,

b)

datum narození,

c)

adresa místa pobytu,

d)

datum narození,

e)

adresa místa pobytu,

f)

adresa místa pobytu,

g)

počet obyvatel,

h)

pohlaví,

ch)

jméno, popřípadě jména, příjmení,

i)

rodné číslo,

j)

státní občanství.

(2)

Volební obdobíVolební období se na kandidátním listině:

a)

jméno, popřípadě jména, příjmení, rodné příjmení,

b)

rodná čísla,

c)

adresa místa pobytu,

d)

státní občanství, popřípadě více státních občanství.

(3)

Volební kampaňvolební kam
```

### 6. Zákon o povinném nošení klobouku v úterý

```text
Předmět úpravy

Tento zákon upravuje v návaznosti na přímo použitelný předpis Evropské unie1), které upravují požadavky na zvláštní mechanismy pro usměrňování, doplňkovou látkou, perorální a podplukovník v úterém propojení10) a postup, které stanoví přímo použitelný předpis Evropské unie1) k ozbrojeným konfliktem na území Unie v úter v úterém propojení10) a přímo použitelného předpisu Evropské unie na základě mezinárodních smluv20) a upravuje postupy a podmínky pro výkon jejich působnosti.20)

§ 2

Tento zákon se nevztahuje na

a)

konceptulptu,

b)

zboží s obalem v úrodním závodám a zboží uvnitř Společenství,

c)

zboží s obalkem v úrodním závodě v tuzemsku.

§ 2

Považské ustanovení

V řízení o registraci plátce stanoví prováděcí předpis prováděcí právní předpis.

§ 3

P
```

### 7. Zákon o daňovém zvýhodnění trpaslíků

```text
Předmět úpravy

(1)

Tento zákon zapracovává příslušné předpisy Evropských společenství1) a upravuje práva a povinnosti osob ze spotřebních zákonů (dále jen „nařízení o daňovém zvýhodnění“), a to včetně daníových povinností stanovených tímto zákonem a za úplatu, která je pro účely daně z přidané hodnoty stanovena k přímé spotřebě ve skládce, pokud je povinen tuto podporu zajistit prostřednictvím úplaty.

(2)

Předmětem daně z přidané hodnoty se rozumí

a)

vratná věcná věcná věcná věcná věcná věcná věcná věcná věc, která jsou pro účely daně z přidané hodnoty,

b)

vratná věcná věcná věcná věcná věcná plnění, kterou určí nebo mají povahu věci, která je pro účely daně z přidané hodnoty stanovena v okamžiku jejího vyřízení, anebo jiným účastníkem úplatu, jestliže o to požádá.

(3)

Zálohou daně z přidané hodnoty, uhradí vlastníkovi vstupní poplatek podle občanského zákoníku.

§
```

### 8. Zákon o zákazu nudy ve veřejné dopravě

```text
Předmět zákona

Tento zákon upravuje poskytování digitálního obsahu, které lze mít za splnění povinností stanovených tímto zákonem nebo zákonným způsobem, zejména pokud jde o osoby, které jsou jeho oprávnění k šíření řádného šíření řádného šíření řádného šíření řádného šíření řádného šíření řádného šíření řádného šíření, jakož i požadavky stanovené v § 37 odst. 2 a § 46 odst. 1.

§ 2

Povinnosti provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele a provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele a provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele vypořádacího systému, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele, provozovatele a provozovatel, provozovatele a provozovatele, provozovatele provozovatele, provozovatele, provozovatele, provozovatel …
```

### 9. Zákon o evidenci zakletých princů

```text
V § 1 odst. 1 písm. a) zákona č. 15/1998 Sb., o Českém rozhlasu

V § 4 odst. 2 zákona č. 15/1998 Sb., o Českém rozhlasu

§ 4

Tento zákon nabývá účinnosti dnem vyhlášení.

Havel v. r.

Dubček v. r.

Čalfa v. r.

1)

§ 25 zákona č. 526/1990 Sb., o cenách.

2)

Zákon č. 477/1992 Sb., o resortních, oborových, podnikových a dalších zdravotních pojišťovnách, ve znění zákona č. 59/1995 Sb.

3)

Zákon č. 18/1997 Sb., o mírovém zabezpečení matky a o změně a doplnění některých zákonů, ve znění zákona č. 192/1999 Sb.
Vyhláška č. 275/2002 Sb., o nakládání s nebezpečnými náposy a chemickými přípravky na ochranu rostlin.

3)
```

### 10. Zákon o minimální délce pohádky na dobrou noc

```text
Předmět úpravy

Tento zákon zapracovává příslušné předpisy Evropské unie1) a upravuje postup pro výměnu vzorků přípravků, které jsou pro účely zajištění jejich ochrany i v případě, že se jedná o humánní léčivých přípravkůléčivých přípravků, a které nejsou uvádějícími na trh nebo do oběhu, a na jeho území, jakož i k výdeji na území, které je v České republice registrováno na trh.

§ 2

Osobě, která není používána do České republiky, je povinen

a)

uveden v § 2 odst. 3 větě první, nebo

b)

v členském státě Evropské unie, je-li to účelně a musí být zajištěna v České republice podle zvláštních právních předpisů tohoto členského státu Evropské unie.

§ 3

Prodloužení platnosti přípravku

(1)

Prodloužení platnosti přípravku nebo účinků přípravku pro montáž do České republiky není možné prodloužit opakovaně.

(2)

Nedojde-li ke snížení doby platnosti přípravku na
```

## Known weaknesses

1. **Glued duplicate words from the source data.** The law CSV often repeats a defined term without a space
   (for example "obceobce", "léčivých přípravkůléčivých přípravků"); the training text contains about 161 000
   such cases and the model copies them. Cleaning them would change the agreed rule "normalise whitespace only"
   and needs a new training run (about 2 hours), so it is left for the user to decide.
2. **The title barely steers the content.** The model continues in a convincing legal style, but mostly about
   generic topics (VAT, health care, residence permits) rather than the absurd subject; with a 10 M model and
   no title–body pairs in training this is expected. Fairy-tale words rarely leak into the legal text.
3. **Repetition loops** at some seeds ("provozovatele, provozovatele, …", "věcná věcná …"). A repetition
   penalty in sampling would help; not part of the agreed design, so not added.
4. **Invented words and broken agreement** appear (for example "periodický chatativní záložna"), more at higher
   temperature. This is part of the humour, but also a limit of the model size.
5. **Real names and laws** can appear in the output, copied from signatures and references of real laws
   ("Havel v. r.", real law numbers). The outputs above were reviewed: no real person is named in an offensive way.
6. **Old orthography**: some Božena Němcová tales are in the 19th-century spelling ("w" for "v"); they are kept,
   but rarely surface.
7. **Speed in the browser**: about 30 tokens/s in desktop Chromium (400 tokens in about 12 s, including one
   context slide at 256 tokens). Phones are not measured yet (release checklist).
8. **Peak RAM during training** was not recorded; the training state checkpoint is 121 MB.
