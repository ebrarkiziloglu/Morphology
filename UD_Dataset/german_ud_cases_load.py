import csv
from typing import Set


def load_duplicate_surface_forms(csv_path: str) -> Set[str]:
    """Lowercased surface forms from all rows (no lemma/morph filter).

    Prefer :func:`load_case_dictionary` plus morph-aware matching in
    ``ud_corruption_utils`` when checking duplicates for a specific token.
    """
    forms: Set[str] = set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for part in row["Forms"].split("|"):
                part = part.strip()
                if part:
                    forms.add(part.lower())
    return forms


def load_case_dictionary(csv_path):
    case_dict = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        # csv.DictReader automatically reads the first row as column headers
        reader = csv.DictReader(f)

        for row in reader:
            lemma = row["Lemma"]
            case = row["Case"]
            number = row["Number"]
            gender = row["Gender"]
            degree = row["Degree"]
            upos = row["Upos"]

            # Split the forms by ' | ', strip whitespace, and filter out empty strings
            forms_str = set(f.strip() for f in row["Forms"].split("|")
                            if f.strip())

            if lemma not in case_dict:
                case_dict[lemma] = []

            case_dict[lemma].append({
                "Case": case,
                "Number": number,
                "Gender": gender,
                "Degree": degree,
                "Upos": upos,
                "forms": forms_str,
            })

    return case_dict


if __name__ == "__main__":
    # --- Usage Example ---
    dictionary = load_case_dictionary('german_ud_cases_dictionary.csv')

    # Lookup the entry for 'Abend'
    print(f"## Abend: {dictionary.get('Abend')}\n")
    # Output: [
    # {'Case': 'Acc', 'Number': 'Sing', 'Gender': 'Masc', 'Upos': 'NOUN', 'forms': {'Abend'}},
    # {'Case': 'Dat', 'Number': 'Sing', 'Gender': 'Masc', 'Upos': 'NOUN', 'forms': {'Abend'}},
    # {'Case': 'Dat', 'Number': 'Plur', 'Gender': 'Masc', 'Upos': 'NOUN', 'forms': {'Abenden'}}]

    print(f"## Anfang: {dictionary.get('Anfang')}\n")
    # Output: [
    # {'Case': 'Acc', 'Number': 'Sing', 'Gender': 'Masc', 'Upos': 'NOUN', 'forms': {'Anfang'}},
    # {'Case': 'Dat', 'Number': 'Sing', 'Gender': 'Masc', 'Upos': 'NOUN', 'forms': {'Anfang'}},
    # {'Case': 'Acc', 'Number': 'Plur', 'Gender': 'Masc', 'Upos': 'NOUN', 'forms': {'Anfänge'}},
    # {'Case': 'Dat', 'Number': 'Plur', 'Gender': 'Masc', 'Upos': 'NOUN', 'forms': {'Anfängen'}},
    # {'Case': 'Dat', 'Number': 'Sing', 'Gender': 'Masc', 'Upos': 'ADV', 'forms': {'Anfangs'}},
    # {'Case': 'Dat', 'Number': '', 'Gender': '', 'Upos': 'NOUN', 'forms': {'Anfang'}}]
