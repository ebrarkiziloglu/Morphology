import csv
from typing import Dict, Iterable, Set, Tuple

FORMS_FIELD_SEP = " | "


def parse_forms_field(forms_cell: str) -> Tuple[Set[str], Dict[str, int]]:
    """Parse a Forms cell of ``surface:count`` tokens separated by `` | ``."""
    forms: Set[str] = set()
    form_counts: Dict[str, int] = {}
    for part in forms_cell.split(FORMS_FIELD_SEP):
        part = part.strip()
        if not part:
            continue
        form, sep, count_str = part.rpartition(":")
        if not sep or not count_str.isdigit():
            raise ValueError(
                f"Forms entry must be 'surface:count', got {part!r} "
                f"in {forms_cell!r}"
            )
        form = form.strip()
        count = int(count_str)
        if not form:
            raise ValueError(f"empty surface in Forms cell {forms_cell!r}")
        forms.add(form)
        form_counts[form] = form_counts.get(form, 0) + count
    return forms, form_counts


def format_forms_field(form_counts: Dict[str, int]) -> str:
    """``surface:count`` tokens in descending frequency order."""
    ranked = sorted(form_counts.items(), key=lambda x: (-x[1], x[0].lower()))
    return FORMS_FIELD_SEP.join(f"{form}:{count}" for form, count in ranked)


def preferred_dictionary_form(forms: Set[str], form_counts: Dict[str, int]) -> str:
    """Pick the most frequent surface; tie-break alphabetically (case-insensitive)."""
    return max(forms, key=lambda f: (form_counts[f], f.lower()))


def merged_form_counts(
    entries: Iterable[dict],
) -> Tuple[Set[str], Dict[str, int]]:
    """Union surfaces and sum counts across dictionary rows."""
    forms: Set[str] = set()
    form_counts: Dict[str, int] = {}
    for entry in entries:
        entry_forms = entry.get("forms") or set()
        forms.update(entry_forms)
        entry_counts = entry["form_counts"]
        for form in entry_forms:
            form_counts[form] = form_counts.get(form, 0) + entry_counts[form]
    return forms, form_counts


def attach_preferred_forms(case_dict: dict) -> None:
    """Set ``preferred_form`` on each entry (most frequent when multiple surfaces)."""
    for lemma in case_dict:
        entries = case_dict[lemma]
        if isinstance(entries, dict):
            entries = [entries]
            case_dict[lemma] = entries
        for entry in entries:
            forms = entry.get("forms") or set()
            counts = entry.get("form_counts") or {}
            if not forms:
                entry["preferred_form"] = None
            elif len(forms) == 1:
                entry["preferred_form"] = next(iter(forms))
            else:
                entry["preferred_form"] = preferred_dictionary_form(forms, counts)


def load_duplicate_surface_forms(csv_path: str) -> Set[str]:
    """Lowercased surface forms from all rows (no lemma/morph filter).

    Prefer :func:`load_case_dictionary` plus morph-aware matching in
    ``ud_corruption_utils`` when checking duplicates for a specific token.
    """
    forms: Set[str] = set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row_forms, _ = parse_forms_field(row["Forms"])
            forms.update(f.lower() for f in row_forms)
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
            inflection = row.get("Inflection", "") or ""

            forms, form_counts = parse_forms_field(row["Forms"])

            if lemma not in case_dict:
                case_dict[lemma] = []

            case_dict[lemma].append({
                "Case": case,
                "Number": number,
                "Gender": gender,
                "Degree": degree,
                "Upos": upos,
                "Inflection": inflection,
                "forms": forms,
                "form_counts": form_counts,
            })

    return case_dict


if __name__ == "__main__":
    # --- Usage Example ---
    dictionary = load_case_dictionary('german_ud_lookup_dictionary.csv')

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
