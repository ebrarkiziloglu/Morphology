# Morphology

Minimal-pair evaluation data for **Accusative ↔ Dative** case sensitivity in German, built from [UD German-GSD](https://github.com/UniversalDependencies/UD_German-GSD) (and related treebanks for the lookup dictionary).

All generation code, the case dictionary, and the published test sets live under [`UD_Dataset/`](UD_Dataset/).

## Layout

```
UD_Dataset/
├── german_ud_lookup_dictionary.csv   # case lookup table (generated)
├── german_ud_lookup_dict_extract.py  # build dictionary from UD treebanks
├── german_ud_lookup_dict_load.py     # load / parse dictionary rows
├── german_ud_lookup_dict_duplicates.py
├── german_adj_inflection.py          # Strong/Weak ADJ rows for dictionary + corruption
├── load_ud_dataset.py                # CoNLL-U loader
├── np_group_span.py                  # NP span detection from UD deps
├── ud_corruption_utils.py            # shared corruption + export logic
├── corrupt_acc_dat_pairs.py          # Acc → Dat pair export
├── corrupt_dat_acc_pairs.py          # Dat → Acc pair export
├── span_elimination_report.py
├── dative_spans/                     # dative span inventory (for Dat→Acc typing)
├── testset/                          # generated minimal-pair JSON
│   ├── accusative_dative_pairs.json
│   ├── dative_accusative_pairs.json
│   ├── *_unconverted.json
│   └── dative_categories/            # Dat→Acc split by dative_type
└── duplicates/                       # rows/pairs with ambiguous dictionary forms
```

Treebank files are expected under `data/` at the repo root (e.g. `data/UD_German-GSD/de_gsd-ud-train.conllu`).

---

## 1. Dictionary lookup

The corruption scripts do **not** use hand-written suffix rules or article tables. Every case flip is resolved from `german_ud_lookup_dictionary.csv`: for each token, the target surface must appear on a row that matches the token’s morphological key.

### Building the dictionary

**Script:** [`UD_Dataset/german_ud_lookup_dict_extract.py`](UD_Dataset/german_ud_lookup_dict_extract.py)

Scans Acc/Dat tokens from four German UD treebanks:

- `data/UD_German-GSD`
- `data/UD_German-HDT`
- `data/UD_German-PUD`
- `data/UD_German-LIT`

For each token with `Case ∈ {Acc, Dat}` and a German-alphabet initial lemma, it aggregates observed surface forms into rows keyed by:

| Column       | Description |
| ------------ | ----------- |
| `Lemma`      | UD lemma (nouns/proper nouns keep casing; other UPOS lowercased). |
| `Case`       | `Acc` or `Dat`. |
| `Number`     | `Sing`, `Plur`, or empty. |
| `Gender`     | `Masc`, `Fem`, `Neut`, or empty. |
| `Degree`     | UD `Degree` when present. |
| `Upos`       | UPOS tag. |
| `Inflection` | For `ADJ`: `Strong`, `Weak`, or empty (see `german_adj_inflection.py`). Strong/Weak rows with identical surfaces are merged. |
| `Forms`      | `surface:count` tokens separated by ` \| `, sorted by descending frequency. |

```bash
cd UD_Dataset
python german_ud_lookup_dict_extract.py
```

Output: [`UD_Dataset/german_ud_lookup_dictionary.csv`](UD_Dataset/german_ud_lookup_dictionary.csv).

### Loading and resolving forms

**Module:** [`UD_Dataset/german_ud_lookup_dict_load.py`](UD_Dataset/german_ud_lookup_dict_load.py)

- `load_case_dictionary(path)` — builds an in-memory map `lemma → [rows]`, each row with `forms` (set) and `form_counts` (dict).
- `attach_preferred_forms(case_dict)` — sets `preferred_form` per row (highest count; tie-break case-insensitively).
- `parse_forms_field` / `format_forms_field` — round-trip the `Forms` column.

**Matching at corruption time** (`ud_corruption_utils.select_dict_surface_form`):

1. Normalize lemma for lookup (`dictionary_lookup_lemma`: lowercased unless `NOUN`/`PROPN`).
2. Find rows with `Case = target_case`, same `Number` and `Upos`, and same `Gender` when available.
3. For **plural**, if no same-gender row exists, fall back to any gender on that `(Lemma, Case, Number, Upos)` key.
4. For **adjectives**, prefer rows whose `Inflection` matches Strong/Weak context (`adj_inflection_from_context`).
5. If several surfaces remain, pick the **most frequent** in the dictionary; apply the gold token’s initial capitalization.

If lookup fails, the NP span is skipped (see unconverted JSON below). Lemmas omitted at extract time (non-German initial) keep their surface form only when that would be an identity mapping.

### Ambiguous dictionary rows

**Script:** [`UD_Dataset/german_ud_lookup_dict_duplicates.py`](UD_Dataset/german_ud_lookup_dict_duplicates.py)

Writes [`UD_Dataset/duplicates/german_ud_lookup_dictionary_duplicates.csv`](UD_Dataset/duplicates/german_ud_lookup_dictionary_duplicates.csv) — all rows where a single morph cell lists **more than one** surface form (e.g. `dem:12 | den:3`).

The pair exporters optionally mirror minimal pairs whose NP span touches such forms into `duplicates/*_pairs_duplicates.json` with a `gold_span_duplicate_forms` field. The main test sets are unchanged.

---

## 2. Corruption scripts

NP-level minimal pairs: for each `Case=Acc` or `Case=Dat` seed token, the code walks UD dependencies to the phrase head ([`np_group_span.py`](UD_Dataset/np_group_span.py)), collects the nominal span, and rewrites **every** `source_case` token in that span using the dictionary. The corrupted sentence is spliced back into `# text` with contraction-aware offsets (`im` ↔ `in` + `dem`, `zum` ↔ `zu` + `dem`, etc.; see `PARTS_TO_CONTRACTION` in [`ud_corruption_utils.py`](UD_Dataset/ud_corruption_utils.py)).

Shared pipeline: `ud_corruption_utils.run_pair_generation_main`.

| Script | Direction | Default output |
| ------ | --------- | -------------- |
| [`corrupt_acc_dat_pairs.py`](UD_Dataset/corrupt_acc_dat_pairs.py) | Acc → Dat | `testset/accusative_dative_pairs.json` |
| [`corrupt_dat_acc_pairs.py`](UD_Dataset/corrupt_dat_acc_pairs.py) | Dat → Acc | `testset/dative_accusative_pairs.json` |

Both read UD German-GSD train/dev/test from `data/UD_German-GSD/` unless `--ud-base` is set.

### Span filtering and diagnostics

A span is **exported** only if:

- Every `source_case` token in the NP span has a dictionary hit for `target_case`.
- Normalized corrupted text differs from gold (no-op pairs dropped).
- The span is not a duplicate of an already-written `(split, sent_id, group_start, group_end, corrupted)` key.

Spans that fail conversion are logged in:

- `testset/accusative_dative_unconverted.json` (Acc → Dat)
- `testset/dative_accusative_unconverted.json` (Dat → Acc)

Each unconverted record includes `first_fail_*` fields and a `reason` code (e.g. `lemma_not_in_dictionary`, `no_Acc_rows_matching_number_gender_upos`).

**Acc → Dat only:** on the `test` split, missing conversions for `NOUN` are tolerated (other UPOS must still convert).

At the end of a run, an **elimination summary** is printed (`span_elimination_report.SpanEliminationReport`).

### Dat → Acc: dative typing and category splits

[`corrupt_dat_acc_pairs.py`](UD_Dataset/corrupt_dat_acc_pairs.py) attaches a `dative_type` label from [`dative_spans/german_datives_all.csv`](UD_Dataset/dative_spans/german_datives_all.csv) (built in [`UD_German.ipynb`](UD_Dataset/UD_German.ipynb)). Labels include core arguments, preposition classes (`dative_prep_in`, …), `comparative_case_dative`, `nominal_dative_modifier`, `oblique_dative`, and others — see `DATIVE_TYPE_LABELS` in `ud_corruption_utils.py`.

The same run also writes one JSON per label under `testset/dative_categories/` (e.g. `dative_prep_mit.json`).

### Example commands

```bash
cd UD_Dataset

# Regenerate dictionary (requires all four UD dirs under ../data/)
python german_ud_lookup_dict_extract.py

# Acc → Dat
python corrupt_acc_dat_pairs.py

# Dat → Acc
python corrupt_dat_acc_pairs.py
```

Use `--log-level DEBUG` for verbose dropout samples. Duplicate-aware exports: `--duplicates-dictionary` and `--output-group-duplicates-json`.

---

## Test sets

Filename pattern: `{source_case}_{target_case}_pairs.json` (e.g. `accusative_dative_pairs.json`, `dative_accusative_pairs.json`).

### Record fields

| Field | Description |
| ----- | ----------- |
| `split` | UD split (`train`, `dev`, `test`). |
| `sent_id` | UD sentence id. |
| `group_start` | 0-based start index of the NP span (inclusive). |
| `group_end` | 0-based end index (inclusive). |
| `group_span_text` | Surface text of the span (from token forms). |
| `gold_text` | Original `# text` sentence. |
| `corrupted` | Sentence after rewriting all `source_case` tokens in the span. |
| `dative_type` | *(Dat → Acc only)* Subtype from `dative_spans/*.csv`. |
| `gold_span_duplicate_forms` | *(duplicates JSON only)* Surfaces whose dictionary row has multiple forms. |

### Example (`accusative_dative_pairs.json`)

```json
{
  "split": "train",
  "sent_id": "train-s6",
  "group_start": 16,
  "group_end": 16,
  "group_span_text": "mich",
  "gold_text": "Sauberkeit, Ordnung und Freundlichkeit brauche ich hier nicht zu erwähnen, denn das gehört für mich zum Standard, der aber auch noch übertroffen wird.",
  "corrupted": "Sauberkeit, Ordnung und Freundlichkeit brauche ich hier nicht zu erwähnen, denn das gehört für mir zum Standard, der aber auch noch übertroffen wird."
}
```

### Example (`dative_accusative_pairs.json`)

```json
{
  "split": "train",
  "sent_id": "train-s2",
  "group_start": 6,
  "group_end": 7,
  "group_span_text": "dem Rahmen",
  "gold_text": "Die Kosten sind definitiv auch im Rahmen.",
  "corrupted": "Die Kosten sind definitiv auch in den Rahmen.",
  "dative_type": "dative_prep_in"
}
```

Indices refer to the token list produced by `load_ud_dataset.parse_conllu_file` (same order as the CoNLL-U file, one node per token).

---

## Supporting modules

| Module | Role |
| ------ | ---- |
| `load_ud_dataset.py` | Parse CoNLL-U; `load_ud_dataset(base_path, splits_filemap)`. |
| `german_adj_inflection.py` | Classify attributive adjectives as Strong/Weak for dictionary rows and ADJ lookup. |
| `np_group_span.py` | Expand a case-marked token to its NP span via UD `head` / `deprel`. |
| `ud_corruption_utils.py` | Dictionary lookup, text splicing, pair iteration, dative typing, JSON export. |
| `span_elimination_report.py` | Aggregate counts for skipped spans and deduplication. |

Python dependencies are listed in [`requirements.txt`](requirements.txt) (stdlib suffices for the pair scripts; notebook tooling may need `numpy` / `pandas`).
