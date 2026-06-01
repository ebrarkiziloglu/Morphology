import csv
import os

from german_ud_lookup_dict_load import (
    format_forms_field,
    load_case_dictionary,
    parse_forms_field,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DICTIONARY_CSV = os.path.join(_SCRIPT_DIR, "german_ud_lookup_dictionary.csv")
DUPLICATES_CSV = os.path.join(
    _SCRIPT_DIR, "duplicates", "german_ud_lookup_dictionary_duplicates.csv"
)

dictionary = load_case_dictionary(DICTIONARY_CSV)

duplicate_entries: dict[str, list[dict]] = {}
for lemma, entries in dictionary.items():
    for entry in entries:
        if len(entry["forms"]) > 1:
            duplicate_entries.setdefault(lemma, []).append(entry)

os.makedirs(os.path.dirname(DUPLICATES_CSV), exist_ok=True)
with open(DUPLICATES_CSV, "w", encoding="utf-8", newline="") as f:
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
    for lemma in sorted(duplicate_entries.keys()):
        for entry in duplicate_entries[lemma]:
            writer.writerow(
                [
                    lemma,
                    entry["Case"],
                    entry["Number"],
                    entry["Gender"],
                    entry["Degree"],
                    entry["Upos"],
                    entry.get("Inflection", "") or "",
                    format_forms_field(entry["form_counts"]),
                ]
            )

dict_forms: set[str] = set()
with open(DUPLICATES_CSV, encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        forms, _ = parse_forms_field(row["Forms"])
        dict_forms.update(forms)

print(f"Wrote {len(duplicate_entries)} lemmas to {DUPLICATES_CSV}")
print(f"Number of duplicate surface forms: {len(dict_forms)}")
