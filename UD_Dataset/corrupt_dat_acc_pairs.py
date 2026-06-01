"""
Create Dative→Accusative sentence pairs from UD German-GSD.

For each sentence that contains at least one Dative (`Case=Dat`) token, this script
creates one or more examples with:

- gold_text: original sentence text
- corrupt_group: group-level corruption for the dative NP around a head noun

If a sentence has multiple Dative tokens, we create a separate example for each
convertible NP span (re-using the same gold sentence but different corrupted
variants).

Matching uses **Lemma, Number, Gender, Upos** against accusative rows in the CSV
(for Plur, any Gender is used only when no same-Gender row exists). If a dative
NP span contains a token with no matching row,
that span is skipped; other spans in the same sentence are still written.

Pairs where any NP-span token (gold or corrupted) has multiple forms in the main
lookup dictionary are also written to a separate JSON file with
``gold_span_duplicate_forms`` listing those surfaces (main output unchanged).
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from ud_corruption_utils import run_pair_generation_main
from load_ud_dataset import UD_GERMAN_GSD_SPLITS

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "data"))



def main(
    ud_base_path: str,
    output_group_json: str,
    dict_csv_path: str,
    log_level: str = "INFO",
    dative_types_csv_path: Optional[str] = None,
    duplicates_dict_csv_path: Optional[str] = None,
    output_group_duplicates_json: Optional[str] = None,
    output_dative_categories_dir: Optional[str] = None,
):
    """Main entry point for generating dative→accusative pairs."""

    if dative_types_csv_path is None:
        dative_types_csv_path = os.path.join(_SCRIPT_DIR, "dative_spans/german_datives_all.csv")

    splits = {
        'train': 'de_gsd-ud-train.conllu',
        'test': 'de_gsd-ud-test.conllu',
        'dev': 'de_gsd-ud-dev.conllu',
        # 'deneme': 'de_gsd-ud-deneme.conllu',
    }
    return run_pair_generation_main(
        ud_base_path=ud_base_path,
        splits=splits,
        output_group_json=output_group_json,
        dict_csv_path=dict_csv_path,
        source_case="Dat",
        target_case="Acc",
        direction_label="Dat→Acc",
        log_level=log_level,
        output_unconverted_json=os.path.join(
            os.path.dirname(output_group_json),
            "dative_accusative_unconverted.json",
        ),
        dative_types_csv_path=dative_types_csv_path,
        duplicates_dict_csv_path=duplicates_dict_csv_path,
        output_group_duplicates_json=output_group_duplicates_json,
        output_dative_categories_dir=output_dative_categories_dir,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dative→Accusative pair export with elimination diagnostics."
    )
    p.add_argument(
        "--log-level", default=os.environ.get("ACC_DAT_PAIR_LOG_LEVEL", "INFO")
    )
    p.add_argument("--ud-base", default=None)
    p.add_argument("--dictionary", default=None)
    p.add_argument("--output-group-json", default=None)
    p.add_argument(
        "--dative-types-csv",
        default=None,
        help="dative_spans/german_datives_all.csv for dative_type labels on each span",
    )
    p.add_argument(
        "--duplicates-dictionary",
        default=None,
        help="duplicates/german_ud_lookup_dictionary_duplicates.csv",
    )
    p.add_argument(
        "--output-group-duplicates-json",
        default=None,
        help="JSON for pairs containing duplicate-dictionary surface forms",
    )
    p.add_argument(
        "--output-dative-categories-dir",
        default=None,
        help="Directory for per-dative_type minimal-pair JSON files (default: testset/dative_categories)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        ud_base_path=(
            args.ud_base
            if args.ud_base
            else os.path.join(_PROJECT_ROOT, "UD_German-GSD")
        ),
        output_group_json=(
            args.output_group_json
            if args.output_group_json
            else os.path.join(
                _SCRIPT_DIR,
                "testset",
                "dative_accusative_pairs.json",
            )
        ),
        dict_csv_path=(
            args.dictionary
            if args.dictionary
            else os.path.join(_SCRIPT_DIR, "german_ud_lookup_dictionary.csv")
        ),
        log_level=str(args.log_level),
        dative_types_csv_path=args.dative_types_csv,
        duplicates_dict_csv_path=(
            args.duplicates_dictionary
            if args.duplicates_dictionary
            else os.path.join(
                _SCRIPT_DIR, "duplicates/german_ud_lookup_dictionary_duplicates.csv"
            )
        ),
        output_group_duplicates_json=(
            args.output_group_duplicates_json
            if args.output_group_duplicates_json
            else os.path.join(
                _SCRIPT_DIR,
                "duplicates",
                "dative_accusative_pairs_duplicates.json",
            )
        ),
        output_dative_categories_dir=(
            args.output_dative_categories_dir
            if args.output_dative_categories_dir
            else os.path.join(_SCRIPT_DIR, "testset", "dative_categories")
        ),
    )
