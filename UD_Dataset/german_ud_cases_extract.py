import csv
import os
from collections import defaultdict
from typing import Dict, Set

from load_ud_dataset import load_ud_dataset

UD_GERMAN_GSD_SPLITS = {
    'train': 'de_gsd-ud-train.conllu',
    'dev': 'de_gsd-ud-dev.conllu',
    'test': 'de_gsd-ud-test.conllu',
}

UD_GERMAN_HDT_SPLITS = {
    'train-a-1': 'de_hdt-ud-train-a-1.conllu',
    'train-a-2': 'de_hdt-ud-train-a-2.conllu',
    'train-b-1': 'de_hdt-ud-train-b-1.conllu',
    'train-b-2': 'de_hdt-ud-train-b-2.conllu',
    'dev': 'de_hdt-ud-dev.conllu',
    'test': 'de_hdt-ud-test.conllu',
}

UD_GERMAN_PUD_SPLITS = {
    'test': 'de_pud-ud-test.conllu',
}

UD_GERMAN_LIT_SPLITS = {
    'test': 'de_lit-ud-test.conllu',
}

UD_GERMAN_SPLITS = [UD_GERMAN_GSD_SPLITS, UD_GERMAN_HDT_SPLITS, UD_GERMAN_PUD_SPLITS, UD_GERMAN_LIT_SPLITS]


def main():

    vocab_cases = defaultdict(lambda: defaultdict(set))
    ud_base_paths = [
        "data/UD_German-GSD", "data/UD_German-HDT", "data/UD_German-PUD",
        "data/UD_German-LIT"
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

                    if lemma and form and case:
                        if case in {"Acc", "Dat"}:
                            vocab_cases[lemma][(case, number, gender,
                                                upos)].add(form)

    count = 0

    for (lemma, values) in vocab_cases.items():
        count += 1
        print(f'lemma: {lemma}:')
        for (key, value) in values.items():
            print(f'\t{key}: {value}')
        if count > 5:
            break

    output_csv = "german_ud_cases_dictionary.csv"
    print(f"Writing to {output_csv}...")
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Lemma", "Case", "Number", "Gender", "Upos", "Forms"])
        for lemma in sorted(vocab_cases.keys()):
            cases = vocab_cases[lemma]
            for key, forms in cases.items():
                forms_str = " | ".join(sorted(forms))
                writer.writerow(
                    [lemma, key[0], key[1], key[2], key[3], forms_str])

    print(f"Done! Saved {len(vocab_cases)} entries to {output_csv}")

if __name__ == "__main__":
    main()
