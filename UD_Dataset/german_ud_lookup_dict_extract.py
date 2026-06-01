import csv
import os
from collections import Counter, defaultdict

from german_adj_inflection import adj_inflection_from_context
from load_ud_dataset import load_ud_dataset, UD_GERMAN_SPLITS

_GERMAN_ALPHA = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜabcdefghijklmnopqrstuvwxyzäöüß")


def _match_first_letter_case(reference: str, word: str) -> str:
    """Match the first letter case of word to reference."""
    if not reference or not word:
        return word
    if reference[0].isupper():
        return word[0].upper() + word[1:]
    return word[0].lower() + word[1:]


def _normalize_lemma_form(lemma: str, form: str, upos: str) -> tuple[str, str]:
    """Lowercase non-nouns; sync lemma/form first-letter case to avoid duplicates."""
    if upos != "NOUN" and upos != "PROPN":
        lemma, form = lemma.lower(), form.lower()
    form = _match_first_letter_case(lemma, form)
    lemma = _match_first_letter_case(form, lemma)
    return lemma, form


def _format_forms_cell(form_counts: Counter[str]) -> str:
    """``surface:count`` tokens separated by ``|``, sorted by descending frequency."""
    ranked = sorted(form_counts.items(), key=lambda x: (-x[1], x[0].lower()))
    return " | ".join(f"{form}:{count}" for form, count in ranked)


def _merge_plural_gender_rows(
    vocab_cases: dict[str, dict[tuple, Counter[str]]],
) -> dict[str, dict[tuple, Counter[str]]]:
    """Merge ``Number=Plur`` rows across genders; ``Gender`` is left empty."""
    out: dict[str, dict[tuple, Counter[str]]] = defaultdict(dict)
    for lemma, rows in vocab_cases.items():
        for key, forms in rows.items():
            case, number, gender, degree, upos, inflection = key
            if number == "Plur":
                key = (case, number, "", degree, upos, inflection)
            bucket = out[lemma].get(key)
            if bucket is None:
                out[lemma][key] = forms.copy()
            else:
                bucket.update(forms)
    return out


def _collapse_identical_inflection_rows(
    vocab_cases: dict[str, dict[tuple, Counter[str]]],
) -> dict[str, dict[tuple, Counter[str]]]:
    """Merge Strong/Weak rows when they carry the same surface forms."""
    out: dict[str, dict[tuple, Counter[str]]] = defaultdict(dict)
    for lemma, rows in vocab_cases.items():
        grouped: dict[tuple, dict[str, Counter[str]]] = defaultdict(dict)
        for key, forms in rows.items():
            case, number, gender, degree, upos, inflection = key
            morph_key = (case, number, gender, degree, upos)
            grouped[morph_key][inflection] = forms

        for morph_key, inflection_map in grouped.items():
            strong = inflection_map.get("Strong", Counter())
            weak = inflection_map.get("Weak", Counter())
            if strong and weak and set(strong) == set(weak):
                out[lemma][(*morph_key, "")] = strong + weak
                continue
            for inflection, forms in inflection_map.items():
                out[lemma][(*morph_key, inflection)] = forms
    return out


def main():
    vocab_cases: dict[str, dict[tuple, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    ud_base_paths = [
        "../data/UD_German-GSD",
        "../data/UD_German-HDT",
        "../data/UD_German-PUD",
        "../data/UD_German-LIT",
    ]
    for i, ud_path in enumerate(ud_base_paths):
        print(f"Loading UD German from: {ud_path}")
        dataset = load_ud_dataset(ud_path, splits_filemap=UD_GERMAN_SPLITS[i])
        splits = UD_GERMAN_SPLITS[i].keys()

        for split in splits:
            if split not in dataset:
                continue
            for sent in dataset[split]:
                tokens = sent.get("tokens", [])
                for token in tokens:
                    lemma = token.get("lemma", "")
                    form = token.get("form", "")
                    upos = token.get("upos", "")
                    feats = token.get("feats", {}) or {}

                    case = feats.get("Case")
                    number = feats.get("Number")
                    gender = feats.get("Gender")
                    degree = feats.get("Degree")

                    if lemma and form and case and lemma[0] in _GERMAN_ALPHA:
                        if case in {"Acc", "Dat"}:
                            lemma, form = _normalize_lemma_form(
                                lemma, form, upos
                            )
                            if number == "Plur":
                                gender = ""
                            inflection = ""
                            if upos == "ADJ":
                                inflection = adj_inflection_from_context(
                                    token, tokens
                                ) or ""
                            vocab_cases[lemma][
                                (case, number, gender, degree, upos, inflection)
                            ][form] += 1

    vocab_cases = _merge_plural_gender_rows(vocab_cases)
    vocab_cases = _collapse_identical_inflection_rows(vocab_cases)

    count = 0
    for lemma, values in vocab_cases.items():
        count += 1
        print(f"lemma: {lemma}:")
        for key, value in values.items():
            print(f"\t{key}: {value}")
        if count > 5:
            break

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(_SCRIPT_DIR, "german_ud_lookup_dictionary.csv")
    print(f"Writing to {output_csv}...")
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Lemma",
                "Case",
                "Number",
                "Gender",
                "Degree",
                "Upos",
                "Inflection",
                "Forms",
            ]
        )
        for lemma in sorted(vocab_cases.keys()):
            cases = vocab_cases[lemma]
            for key, forms in cases.items():
                forms_str = _format_forms_cell(forms)
                writer.writerow(
                    [
                        lemma,
                        key[0],
                        key[1],
                        key[2],
                        key[3],
                        key[4],
                        key[5],
                        forms_str,
                    ]
                )

    print(f"Done! Saved {len(vocab_cases)} lemmas to {output_csv}")


if __name__ == "__main__":
    main()
