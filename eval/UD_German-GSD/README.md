# UD German Eval Datasets

Evaluation resources derived from the [UD German-GSD](https://github.com/UniversalDependencies/UD_German-GSD) treebank 
for probing **Accusative ↔ Dative** case sensitivity via minimal-pair tests.

## Case dictionary

**File:** `german_ud_cases_dictionary.csv`

Lookup table built by scanning every UD token and grouping observed
surface forms by `(Lemma, Case, Number, Upos)`. Used by the corruption
scripts to pick a contextually plausible Acc/Dat form.


| Column   | Description                                             |
| -------- | ------------------------------------------------------- |
| `Lemma`  | UD lemma.                                               |
| `Case`   | `Acc` or `Dat`.                                         |
| `Number` | `Sing`, `Plur`, or empty.                               |
| `Upos`   | UPOS tag (`NOUN`, `PROPN`, `ADJ`, `DET`, `PRON`, …).    |
| `Forms`  | Pipe-separated surface forms for that cell (e.g. `Abbau |

## Test Sets

Each entry pairs a UD sentence (`gold_text`) with a `corrupted` version
that differs only in the case marking of one target token or one
noun-phrase span.

Filename pattern: `{source}_{target}_pairs.json`

- `source` / `target` — case in the gold sentence vs. the corruption.

No-op corruptions are filtered out and group files are deduplicated per span.

### Dataset fields


| Field             | Description                                                       |
| ----------------- | ----------------------------------------------------------------- |
| `split`           | UD split.                                                         |
| `sent_id`         | Original UD sentence id.                                          |
| `group_start`     | 0-based start index (inclusive) of the NP span.                   |
| `group_end`       | 0-based end index (inclusive) of the NP span.                     |
| `group_span_text` | Surface text of the gold NP span.                                 |
| `gold_text`       | Original UD sentence.                                             |
| `corrupted`       | Same sentence with every case-marked token in the span rewritten. |


#### Example (`accusative_dative_pairs.json`)

```json
{
  "split": "train",
  "sent_id": "train-s12",
  "group_start": 5,
  "group_end": 7,
  "group_span_text": "meinem ganzen Leben",
  "gold_text": "Jemand unfreundlicheres ist mir in meinem ganzen Leben noch nicht begegnet!",
  "corrupted": "Jemand unfreundlicheres ist mir in meinen ganze Leben noch nicht begegnet!"
}
```

## Generation Scripts

- `src/eval_dataset/create_accusative_dative_pairs.py` — Acc → Dat
- `src/eval_dataset/create_dative_accusative_pairs.py` — Dat → Acc

For each target token, the scripts:

1. Look up the lemma in the case dictionary and pick a surface form
  matching the desired target `(Case, Number, Upos)`. If multiple forms
   are available, the one whose lowercase matches the heuristic fallback
   is preferred so casing stays natural.
2. Fall back to UPOS-specific suffix heuristics when the dictionary has
  no entry (e.g. adjective `-en/-e/-es → -em` for Acc → Dat,
   noun `-en/-er → drop final letter`, plus a hand-written table for
   common determiners and pronouns).
3. Splice the corrupted form back into `gold_text` at the correct
  character offset, handling German preposition + article contractions
   (`am`, `im`, `zum`, `zur`, `ans`, `ins`, …) so that, e.g.,
   `im Rahmen` correctly becomes `in den Rahmen` after a Dat → Acc
   corruption and not `im den Rahmen`.

### Mappings used in corruption:

```py
DAT_TO_ACC_FORM: Dict[str, str] = {
    # Definite articles (singular)
    "dem": "den",   # masc/neut Dat.Sg -> Acc.Sg.Masc
    "der": "die",   # fem Dat.Sg -> Acc.Sg.Fem (also Nom.Sg.Fem)
    "den": "die",   # Dat.Pl -> Acc.Pl
    "einem": "einen",
    "einer": "eine",
    "mir": "mich",
    "dir": "dich",
    "ihm": "ihn",
    "ihr": "sie",
    "uns": "uns",   # Dat/Acc identical; kept for completeness
    "euch": "euch",
    "ihnen": "sie",
    "meinem": "meinen",
    "meiner": "meine",
    "deinem": "deinen",
    "deiner": "deine",
    "seinem": "seinen",
    "seiner": "seine",
    "ihren": "ihren",
    "ihre": "ihre",
}
```

```py
ACC_TO_DAT_FORM: Dict[str, str] = {
    "den": "dem",
    "das": "dem",
    "die": "der",
    "einen": "einem",
    "eine": "einer",
    "ein": "einem",
    "mich": "mir",
    "dich": "dir",
    "ihn": "ihm",
    "es": "ihm",
    "uns": "uns",
    "euch": "euch",
    "meinen": "meinem",
    "meine": "meiner",
    "mein": "meinem",
    "deinen": "deinem",
    "deine": "deiner",
    "dein": "deinem",
    "seinen": "seinem",
    "seine": "seiner",
    "sein": "seinem",
    "ihren": "ihrem",
    "ihre": "ihrer",
    "unseren": "unserem",
    "unsere": "unserer",
    "euren": "eurem",
    "eure": "eurer",
}
```


```py
_PARTS_TO_CONTRACTION: Dict[Tuple[str, str], str] = {
    ("an", "das"): "ans",
    ("an", "dem"): "am",
    ("auf", "das"): "aufs",
    ("bei", "dem"): "beim",
    ("durch", "das"): "durchs",
    ("für", "das"): "fürs",
    ("hinter", "das"): "hinters",
    ("hinter", "dem"): "hinterm",
    ("in", "das"): "ins",
    ("in", "dem"): "im",
    ("über", "das"): "übers",
    ("über", "dem"): "überm",
    ("um", "das"): "ums",
    ("unter", "das"): "unters",
    ("unter", "dem"): "unterm",
    ("von", "dem"): "vom",
    ("vor", "das"): "vors",
    ("vor", "dem"): "vorm",
    ("zu", "dem"): "zum",
    ("zu", "der"): "zur",
}
```
