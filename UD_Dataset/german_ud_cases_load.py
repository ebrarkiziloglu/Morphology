import csv



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
