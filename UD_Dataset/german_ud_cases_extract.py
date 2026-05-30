import csv
import os
from collections import defaultdict
from typing import Dict, Set

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


def main():

    vocab_cases = defaultdict(lambda: defaultdict(set))
    ud_base_paths = [
        "../data/UD_German-GSD", "../data/UD_German-HDT",
        "../data/UD_German-PUD", "../data/UD_German-LIT"
    ]
    for i, ud_path in enumerate(ud_base_paths):

        print(f"Loading UD German-GSD from: {ud_path}")
        dataset = load_ud_dataset(ud_path, splits_filemap=UD_GERMAN_SPLITS[i])

        splits = UD_GERMAN_SPLITS[i].keys()

        for split in splits:
            if split not in dataset:
                continue
            for sent in dataset[split]:
                for token in sent.get("tokens", []):
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
                                lemma, form, upos)
                            vocab_cases[lemma][(case, number, gender, degree,
                                                upos)].add(form)

    count = 0

    for (lemma, values) in vocab_cases.items():
        count += 1
        print(f'lemma: {lemma}:')
        for (key, value) in values.items():
            print(f'\t{key}: {value}')
        if count > 5:
            break

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(_SCRIPT_DIR, "german_ud_cases_dictionary.csv")
    print(f"Writing to {output_csv}...")
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Lemma", "Case", "Number", "Gender", "Degree", "Upos", "Forms"])
        for lemma in sorted(vocab_cases.keys()):
            cases = vocab_cases[lemma]
            for key, forms in cases.items():
                forms_str = " | ".join(sorted(forms))
                writer.writerow(
                    [lemma, key[0], key[1], key[2], key[3], key[4], forms_str])

    print(f"Done! Saved {len(vocab_cases)} entries to {output_csv}")

if __name__ == "__main__":
    main()
