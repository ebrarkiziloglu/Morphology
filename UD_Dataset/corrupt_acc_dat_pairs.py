"""
Create Accusative→Dative sentence pairs from UD German-GSD.

- Token-level: for each `Case=Acc` token, surface form is taken only from
  `german_ud_cases_dictionary.csv`, matching **Lemma, Number, Gender, Upos**
  (for Plur, any Gender is used only when no same-Gender row exists).
- Group-level: accusative NP spans, same rule per Acc token.

If an accusative NP span contains a token with no matching dictionary row, that
span is skipped; other spans in the same sentence are still written. Rows where
the corrupt sentence equals `gold_text` are still omitted. The group file
deduplicates one row per NP span (same logic as the dative script).

Pairs where any NP-span token (gold or corrupted) has multiple forms in the main
lookup dictionary are also written to a separate JSON file with
``gold_span_duplicate_forms`` listing those surfaces (main output unchanged).

The script prints an elimination summary (counts + reasons) at the end.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Set

from ud_corruption_utils import run_pair_generation_main

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "data"))


def _allow_missing_for_upos(split: str) -> Optional[Set[str]]:
    """On the test split, allow NOUN tokens whose conversion is missing."""
    return {"NOUN"} if split == "test" else None


def main(
    ud_base_path: str,
    output_group_json: str,
    dict_csv_path: str,
    log_level_name: str = "INFO",
    duplicates_dict_csv_path: Optional[str] = None,
    output_group_duplicates_json: Optional[str] = None,
):
    """Main entry point for generating accusative→dative pairs."""
    lvl = (log_level_name
           or os.environ.get("ACC_DAT_PAIR_LOG_LEVEL", "").strip().upper()
           or "INFO")
    
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
        source_case="Acc",
        target_case="Dat",
        direction_label="Acc→Dat",
        log_level=lvl,
        allow_missing_for_upos_fn=_allow_missing_for_upos,
        output_unconverted_json=os.path.join(
            os.path.dirname(output_group_json),
            "accusative_dative_unconverted.json",
        ),
        duplicates_dict_csv_path=duplicates_dict_csv_path,
        output_group_duplicates_json=output_group_duplicates_json,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Accusative→Dative pair export with elimination diagnostics.",
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("ACC_DAT_PAIR_LOG_LEVEL", "INFO"),
        help="logging level for stderr (default INFO; try DEBUG)",
    )
    p.add_argument(
        "--ud-base",
        default=None,
        help="directory containing UD_German-GSD CoNLL-U files",
    )
    p.add_argument(
        "--dictionary",
        default=None,
        help="path to german_ud_cases_dictionary.csv",
    )
    p.add_argument(
        "--output-group-json",
        default=None,
        help="output JSON path",
    )
    p.add_argument(
        "--duplicates-dictionary",
        default=None,
        help="duplicates/german_ud_cases_dictionary_duplicates.csv",
    )
    p.add_argument(
        "--output-group-duplicates-json",
        default=None,
        help="JSON for pairs containing duplicate-dictionary surface forms",
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
                _SCRIPT_DIR, "testset", "accusative_dative_pairs.json"
            )
        ),
        dict_csv_path=(
            args.dictionary
            if args.dictionary
            else os.path.join(_SCRIPT_DIR, "german_ud_cases_dictionary.csv")
        ),
        log_level_name=args.log_level,
        duplicates_dict_csv_path=(
            args.duplicates_dictionary
            if args.duplicates_dictionary
            else os.path.join(
                _SCRIPT_DIR, "duplicates/german_ud_cases_dictionary_duplicates.csv"
            )
        ),
        output_group_duplicates_json=(
            args.output_group_duplicates_json
            if args.output_group_duplicates_json
            else os.path.join(
                _SCRIPT_DIR,
                "duplicates",
                "accusative_dative_pairs_duplicates.json",
            )
        ),
    )
