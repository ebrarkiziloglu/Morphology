from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Mapping

from german_adj_inflection import adj_inflection_from_context
from german_ud_lookup_dict_load import (
    attach_preferred_forms,
    merged_form_counts,
    preferred_dictionary_form,
)
from np_group_span import find_nominal_group_indices, walk_np_head_index
from span_elimination_report import SpanEliminationReport

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# German preposition + article contractions (text alignment / splicing only)
# ---------------------------------------------------------------------------

PARTS_TO_CONTRACTION: Dict[Tuple[str, str], str] = {
    ("an", "das"): "ans",
    ("an", "dem"): "am",
    ("auf", "das"): "aufs",
    ("bei", "dem"): "beim",
    ("durch", "das"): "durchs",
    ("für", "das"): "fürs",
    ("hinter", "das"): "hinters",
    ("hinter", "dem"): "hinterm",
    ("in", "das"): "ins",
    ("in", "dem"): "im",
    ("über", "das"): "übers",
    ("über", "dem"): "überm",
    ("um", "das"): "ums",
    ("unter", "das"): "unters",
    ("unter", "dem"): "unterm",
    ("von", "dem"): "vom",
    ("vor", "das"): "vors",
    ("vor", "dem"): "vorm",
    ("zu", "dem"): "zum",
    ("zu", "der"): "zur",
}

UNCONVERTED = "UNCONVERTED"


def cap_like(orig_form: str, out: str) -> str:
    if orig_form[:1].isupper():
        return out[:1].upper() + out[1:] if len(out) > 1 else out.upper()
    return out.lower()


_LEMMA_LOOKUP_TITLECASE_UPOS = frozenset({"NOUN", "PROPN"})

# Must match german_ud_lookup_dict_extract._GERMAN_ALPHA (dictionary build filter).
_GERMAN_ALPHA = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜabcdefghijklmnopqrstuvwxyzäöüß"
)


def lemma_skipped_by_dictionary_extract(lemma: str) -> bool:
    """True when *german_ud_lookup_dict_extract* omits this lemma (non-German initial)."""
    return bool(lemma) and lemma[0] not in _GERMAN_ALPHA


def dictionary_lookup_lemma(lemma: str, upos: str) -> str:
    """Lemma key for case_dict lookup; lowercased unless UPOS is NOUN or PROPN."""
    if upos == "NOUN" and lemma.lower() == "beiliegen":
        return "Beilage"
    if upos in _LEMMA_LOOKUP_TITLECASE_UPOS:
        return lemma
    return lemma.lower()


def apply_gold_initial_capitalization(gold_form: str, dict_form: str) -> str:
    """Match first-letter case of *dict_form* to *gold_form* when gold starts uppercase."""
    if not dict_form:
        return dict_form
    if gold_form[:1].isupper():
        if len(dict_form) > 1:
            return dict_form[:1].upper() + dict_form[1:]
        return dict_form.upper()
    return dict_form


def _token_morph_snapshot(
    tok: Mapping[str, Any],
    *,
    form: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "form": form if form is not None else tok.get("form", ""),
        "lemma": tok.get("lemma", ""),
        "upos": tok.get("upos", ""),
        "feats": dict(tok.get("feats") or {}),
    }


def _dictionary_forms_for_token(
    lemma: str,
    upos: str,
    feats: Mapping[str, Any],
    case_dict: dict,
    case: str,
    *,
    tokens: Optional[List[Mapping[str, Any]]] = None,
    token: Optional[Mapping[str, Any]] = None,
) -> Set[str]:
    """Surface forms from *case_dict* rows matching this token's morph key."""
    lookup_lemma = dictionary_lookup_lemma(lemma, upos)
    if lookup_lemma not in case_dict:
        return set()
    entries = case_dict[lookup_lemma]
    if isinstance(entries, dict):
        entries = [entries]
    inflection = None
    if upos == "ADJ" and tokens is not None and token is not None:
        inflection = adj_inflection_from_context(token, tokens)
    matches = dict_entries_matching_lookup(
        entries,
        target_case=case,
        number=feats.get("Number"),
        ud_gender=ud_gender_from_feats(feats),
        upos=upos,
        inflection=inflection,
    )
    forms: Set[str] = set()
    for entry in matches:
        forms.update(entry.get("forms") or set())
    return forms


def token_has_multiple_dictionary_forms(
    lemma: str,
    upos: str,
    feats: Mapping[str, Any],
    case_dict: dict,
    case: str,
) -> bool:
    """True when the matching lookup row lists more than one surface form."""
    return len(_dictionary_forms_for_token(lemma, upos, feats, case_dict, case)) > 1


def gold_span_duplicate_surface_forms(
    gold_span_tokens: Iterable[Mapping[str, Any]],
    target_span_tokens: Iterable[Mapping[str, Any]],
    case_dict: dict,
    target_case: str,
) -> List[str]:
    """Surfaces in the NP span whose lookup row has multiple forms (gold or corrupted).

    - Gold: each token in *gold_span_tokens*, using its UD ``Case`` for lookup.
    - Corrupted: each converted token in *target_span_tokens*, using *target_case*.
    """
    if not case_dict:
        return []
    out: List[str] = []
    seen_lower: Set[str] = set()

    def add(form: str) -> None:
        if not form:
            return
        key = form.lower()
        if key not in seen_lower:
            seen_lower.add(key)
            out.append(form)

    for tok in gold_span_tokens:
        feats = tok.get("feats") or {}
        case = feats.get("Case") or ""
        if not case:
            continue
        if token_has_multiple_dictionary_forms(
            tok.get("lemma", ""),
            tok.get("upos", ""),
            feats,
            case_dict,
            case,
        ):
            add(tok.get("form", ""))

    for tok in target_span_tokens:
        if token_has_multiple_dictionary_forms(
            tok.get("lemma", ""),
            tok.get("upos", ""),
            tok.get("feats") or {},
            case_dict,
            target_case,
        ):
            add(tok.get("form", ""))

    return out


def normalize_german_for_pair_compare(s: str) -> str:
    """
    Normalize German surface strings for "are these actually the same?" checks.
    """
    if not s:
        return ""
    t = s.lower()
    t = " ".join(t.split())
    t = re.sub(r"\bzu\s+der\b", "zur", t)
    t = re.sub(r"\bzu\s+dem\b", "zum", t)
    t = re.sub(r"\bim\b", "in dem", t)
    t = re.sub(r"\bin\s+der\b", "in dem", t)
    t = re.sub(r"\s*\"\s*", "\"", t)
    t = re.sub(r"\s*„\s*", "„", t)
    t = re.sub(r"\s*“\s*", "“", t)
    t = re.sub(r"\s*”\s*", "”", t)
    t = re.sub(r"\s*«\s*", "«", t)
    t = re.sub(r"\s*»\s*", "»", t)
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    return t.strip()


def tokens_to_text(tokens: List[Dict[str, Any]]) -> str:
    forms = [t.get("form", "") for t in tokens]
    text = ""
    for i, form in enumerate(forms):
        if i == 0:
            text += form
            continue
        if form in {".", ",", "!", "?", ":", ";", ")", "]", "»"}:
            text += form
        elif form in {"'s", "'"}:
            text += form
        elif form in {"(", "[", "«", "„", "“", "‚"}:
            text += " " + form
        else:
            text += " " + form
    return text


def gold_char_span_for_tokens(
    gold_text: str,
    tokens: List[Dict[str, Any]],
    span_start: int,
    span_end: int,
) -> Tuple[Optional[int], Optional[int]]:
    if (
        not gold_text
        or span_start < 0
        or span_end >= len(tokens)
        or span_start > span_end
    ):
        return None, None
    cursor = 0
    char_start: Optional[int] = None
    n = len(tokens)
    for i in range(n):
        form = tokens[i].get("form", "")
        if i > 0:
            if cursor < len(gold_text) and gold_text[cursor] == " ":
                cursor += 1
        if cursor + len(form) <= len(gold_text) and gold_text[cursor : cursor + len(form)] == form:
            pass
        else:
            j = gold_text.find(form, cursor)
            if j == -1:
                return None, None
            cursor = j
        if i == span_start:
            char_start = cursor
        if i == span_end:
            return char_start, cursor + len(form)
        cursor += len(form)
    return None, None


def group_span_surface_text(
    gold_text: str,
    tokens: List[Dict[str, Any]],
    span_indices: Tuple[int, ...],
) -> str:
    """Surface text of the span as it appears in *gold_text*.

    Uses the character range from the first to the last index in
    *span_indices*, so punctuation between cluster tokens (e.g. ``--`` in
    *US -- Konzerne*) is preserved. Falls back to joining token forms when
    alignment fails.
    """
    if not span_indices:
        return ""
    start_idx, end_idx = span_indices[0], span_indices[-1]
    if gold_text:
        cs, ce = gold_char_span_for_tokens(gold_text, tokens, start_idx, end_idx)
        if cs is not None and ce is not None:
            return gold_text[cs:ce]
    return tokens_to_text([tokens[i] for i in span_indices])


def _token_char_offsets(
    gold_text: str,
    tokens: List[Dict[str, Any]],
) -> Optional[Tuple[List[Tuple[int, int]], Dict[int, int]]]:
    if not gold_text or not tokens:
        return None
    offsets: List[Tuple[int, int]] = []
    contraction_map: Dict[int, int] = {}
    i = 0
    n = len(tokens)
    while i < n:
        form = tokens[i].get("form", "")
        if offsets:
            if cursor < len(gold_text) and gold_text[cursor] == " ":
                cursor += 1
        else:
            cursor = 0

        if cursor + len(form) <= len(gold_text) and gold_text[cursor : cursor + len(form)] == form:
            offsets.append((cursor, cursor + len(form)))
            cursor += len(form)
            i += 1
            continue

        if i + 1 < n:
            next_form = tokens[i + 1].get("form", "")
            pair_key = (form.lower(), next_form.lower())
            contraction = PARTS_TO_CONTRACTION.get(pair_key)
            if contraction:
                clen = len(contraction)
                if cursor + clen <= len(gold_text):
                    candidate = gold_text[cursor : cursor + clen]
                    if candidate.lower() == contraction:
                        end_pos = cursor + clen
                        if end_pos >= len(gold_text) or not gold_text[end_pos].isalpha():
                            span = (cursor, end_pos)
                            offsets.append(span)
                            offsets.append(span)
                            contraction_map[i] = i + 1
                            contraction_map[i + 1] = i
                            cursor = end_pos
                            i += 2
                            continue

        j = gold_text.find(form, cursor)
        if j == -1:
            return None
        offsets.append((j, j + len(form)))
        cursor = j + len(form)
        i += 1
    return offsets, contraction_map


def replace_tokens_in_gold_text(
    gold_text: str,
    tokens: List[Dict[str, Any]],
    idx_to_new_form: Dict[int, str],
) -> Optional[str]:
    if not gold_text:
        return None
    if not idx_to_new_form:
        return gold_text

    result = _token_char_offsets(gold_text, tokens)
    if result is None:
        return None
    offsets, contraction_map = result

    out = gold_text
    processed_spans: Set[Tuple[int, int]] = set()

    for idx in sorted(idx_to_new_form.keys(), reverse=True):
        if idx >= len(offsets):
            continue
        span = offsets[idx]
        if span in processed_spans:
            continue
        processed_spans.add(span)

        if idx in contraction_map:
            peer = contraction_map[idx]
            first_idx = min(idx, peer)
            second_idx = max(idx, peer)

            form1 = idx_to_new_form.get(first_idx, tokens[first_idx].get("form", ""))
            form2 = idx_to_new_form.get(second_idx, tokens[second_idx].get("form", ""))

            new_contraction = PARTS_TO_CONTRACTION.get((form1.lower(), form2.lower()))
            orig_surface = gold_text[span[0] : span[1]]
            if new_contraction:
                replacement = cap_like(orig_surface, new_contraction)
            else:
                replacement = cap_like(orig_surface, form1) + " " + form2

            out = out[: span[0]] + replacement + out[span[1] :]
        else:
            cs, ce = span
            out = out[:cs] + idx_to_new_form[idx] + out[ce:]

    return out


def fix_unmerged_contractions(gold_text: str, corrupted: str) -> str:
    for (prep, art), contraction in PARTS_TO_CONTRACTION.items():
        pattern = re.compile(
            r"\b(" + re.escape(prep) + r")\s+(" + re.escape(art) + r")\b",
            re.IGNORECASE,
        )
        gold_expanded_positions = [m.start() for m in pattern.finditer(gold_text)]

        def _make_repl(cont: str, keep_positions: list):
            def _repl(m):
                for gp in keep_positions:
                    if abs(m.start() - gp) <= 30:
                        return m.group(0)
                return cap_like(m.group(1), cont)

            return _repl

        corrupted = pattern.sub(_make_repl(contraction, gold_expanded_positions), corrupted)
    return corrupted


# ---------------------------------------------------------------------------
# Dictionary lookup helpers (strict match)
# ---------------------------------------------------------------------------


def _normalize_morph_feat_cell(cell: Any) -> str:
    """Map missing UD/dictionary values (None, '', 'none') to a single empty token."""
    s = str(cell or "").strip()
    if s.lower() == "none":
        return ""
    return s


def ud_gender_from_feats(feats: Dict[str, Any]) -> str:
    return _normalize_morph_feat_cell(feats.get("Gender"))


def number_matches_dictionary(ud_number: Any, dict_number_cell: Any) -> bool:
    u = _normalize_morph_feat_cell(ud_number)
    d = _normalize_morph_feat_cell(dict_number_cell)
    if not u and not d:
        return True
    if not u or not d:
        return False
    return u == d


def gender_matches_dictionary(ud_gender: str, dict_gender_cell: Any) -> bool:
    d = _normalize_morph_feat_cell(dict_gender_cell)
    u = _normalize_morph_feat_cell(ud_gender)
    if not u and not d:
        return True
    if not u or not d:
        return False
    parts = [p.strip() for p in d.split(",") if p.strip()]
    return u in parts


def _is_plural_number(number: Any) -> bool:
    return _normalize_morph_feat_cell(number) == "Plur"


def _dict_row_gender_empty(entry: dict) -> bool:
    return _normalize_morph_feat_cell(entry.get("Gender")) == ""


def _dict_entries_filter(
    entries: List[dict],
    *,
    target_case: str,
    number: Any,
    ud_gender: str,
    upos: str,
    require_gender: bool,
    require_no_gender: bool = False,
    inflection: Optional[str] = None,
) -> List[dict]:
    out: List[dict] = []
    for e in entries:
        if e.get("Case") != target_case:
            continue
        if not number_matches_dictionary(number, e.get("Number")):
            continue
        if e.get("Upos") != upos:
            continue
        if require_no_gender and not _dict_row_gender_empty(e):
            continue
        if require_gender and not gender_matches_dictionary(ud_gender, e.get("Gender")):
            continue
        if inflection is not None:
            row_inflection = e.get("Inflection") or ""
            if row_inflection and row_inflection != inflection:
                continue
        out.append(e)
    return out


def _dict_entries_with_inflection_preference(
    entries: List[dict],
    *,
    target_case: str,
    number: Any,
    ud_gender: str,
    upos: str,
    require_gender: bool,
    require_no_gender: bool = False,
    inflection: Optional[str],
) -> List[dict]:
    """Prefer rows tagged with *inflection*, then untagged shared endings."""
    if inflection:
        exact = _dict_entries_filter(
            entries,
            target_case=target_case,
            number=number,
            ud_gender=ud_gender,
            upos=upos,
            require_gender=require_gender,
            require_no_gender=require_no_gender,
            inflection=inflection,
        )
        if exact:
            return exact
        shared = _dict_entries_filter(
            entries,
            target_case=target_case,
            number=number,
            ud_gender=ud_gender,
            upos=upos,
            require_gender=require_gender,
            require_no_gender=require_no_gender,
            inflection="",
        )
        if shared:
            return shared
    return _dict_entries_filter(
        entries,
        target_case=target_case,
        number=number,
        ud_gender=ud_gender,
        upos=upos,
        require_gender=require_gender,
        require_no_gender=require_no_gender,
        inflection=None,
    )


def dict_entries_matching_lookup(
    entries: List[dict],
    *,
    target_case: str,
    number: Any,
    ud_gender: str,
    upos: str,
    inflection: Optional[str] = None,
) -> List[dict]:
    """Match dictionary rows by (Case, Number, Upos) and optional ADJ inflection.

    Singular: require the same ``Gender``. Plural: use only rows with empty
    ``Gender`` (all genders merged at dictionary build time).
    """
    if _is_plural_number(number):
        return _dict_entries_with_inflection_preference(
            entries,
            target_case=target_case,
            number=number,
            ud_gender=ud_gender,
            upos=upos,
            require_gender=False,
            require_no_gender=True,
            inflection=inflection,
        )
    return _dict_entries_with_inflection_preference(
        entries,
        target_case=target_case,
        number=number,
        ud_gender=ud_gender,
        upos=upos,
        require_gender=True,
        require_no_gender=False,
        inflection=inflection,
    )


def select_dict_surface_form(
    *,
    case_dict: dict,
    lemma: str,
    target_case: str,
    number: Any,
    ud_gender: str,
    upos: str,
    orig_form: str,
    tokens: Optional[List[Mapping[str, Any]]] = None,
    token: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """
    Select a surface form from the case dictionary for
    (Lemma, target_case, Number, Gender, Upos). For ``Number=Plur``, only
    gender-agnostic rows (empty ``Gender`` column) are used. Returns None if no match.
    When several surfaces match, picks the highest-frequency form from the
    dictionary (not the gold token spelling). Non-NOUN/PROPN lemmas are
    lowercased for lookup; the chosen surface inherits initial cap from *orig_form*.
    """
    lookup_lemma = dictionary_lookup_lemma(lemma, upos)
    if lookup_lemma not in case_dict:
        if lemma_skipped_by_dictionary_extract(lemma):
            return orig_form
        return None
    entries = case_dict[lookup_lemma]
    if isinstance(entries, dict):
        entries = [entries]

    inflection = None
    if upos == "ADJ" and tokens is not None and token is not None:
        inflection = adj_inflection_from_context(token, tokens)

    matches = dict_entries_matching_lookup(
        entries,
        target_case=target_case,
        number=number,
        ud_gender=ud_gender,
        upos=upos,
        inflection=inflection,
    )
    if not matches:
        return None

    forms, form_counts = merged_form_counts(matches)
    if not forms:
        return None

    selected = preferred_dictionary_form(forms, form_counts)
    return apply_gold_initial_capitalization(orig_form, selected)


def env_debug_seed_indices(var_name: str) -> Optional[Set[int]]:
    raw = os.environ.get(var_name, "").strip()
    if not raw:
        return None
    out: Set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            _logger.warning("%s: ignore bad token index %r", var_name, part)
    return out or None


# ---------------------------------------------------------------------------
# Dative subtype (from dative_spans/german_datives_all.csv / UD_German.ipynb taxonomy)
# ---------------------------------------------------------------------------

DativeTypeKey = Tuple[str, str, str, str, str]  # split, sent_id, head_lemma, head_form, relation

# Dative-only prepositions (always dative): one ``dative_type`` label per lemma.
DATIVE_ONLY_PREPS: frozenset[str] = frozenset({
    "aus", "bei", "mit", "nach", "seit", "von", "zu",
})
DATIVE_ONLY_PREP_TYPES: Dict[str, str] = {
    "aus": "dative_prep_aus",
    "bei": "dative_prep_bei",
    "mit": "dative_prep_mit",
    "nach": "dative_prep_nach",
    "seit": "dative_prep_seit",
    "von": "dative_prep_von",
    "zu": "dative_prep_zu",
}

# Wechselpräpositionen (dative or accusative): one ``dative_type`` label per lemma.
TWO_WAY_PREPS: frozenset[str] = frozenset({
    "an", "auf", "hinter", "in", "neben", "unter", "vor", "zwischen", "über",
})
TWO_WAY_PREP_TYPES: Dict[str, str] = {
    "an": "dative_prep_an",
    "auf": "dative_prep_auf",
    "hinter": "dative_prep_hinter",
    "in": "dative_prep_in",
    "neben": "dative_prep_neben",
    "unter": "dative_prep_unter",
    "vor": "dative_prep_vor",
    "zwischen": "dative_prep_zwischen",
    "über": "dative_prep_über",
}

TARGET_DATIVE_PREPS: frozenset[str] = DATIVE_ONLY_PREPS | TWO_WAY_PREPS
TARGET_DATIVE_PREP_TYPES: Dict[str, str] = {
    **DATIVE_ONLY_PREP_TYPES,
    **TWO_WAY_PREP_TYPES,
}

DATIVE_TYPE_LABELS: Tuple[str, ...] = (
    "adverbial_case_dative",
    "comparative_case_dative",
    "core_dative_argument",
    "dative_prep_an",
    "dative_prep_auf",
    "dative_prep_aus",
    "dative_prep_bei",
    "dative_prep_hinter",
    "dative_prep_in",
    "dative_prep_mit",
    "dative_prep_nach",
    "dative_prep_neben",
    "dative_prep_seit",
    "dative_prep_unter",
    "dative_prep_von",
    "dative_prep_vor",
    "dative_prep_zu",
    "dative_prep_zwischen",
    "dative_prep_über",
    "nominal_dative_modifier",
    "oblique_dative",
    "other_dative",
    "other_prepositional_dative",
)

# UD ``case`` comparators (GSD: ADP + KOKOM), e.g. *höher als in dem Wasser*.
COMPARATIVE_CASE_MARKERS: frozenset[str] = frozenset({"als", "wie"})
COMPARATIVE_CASE_DATIVE_TYPE = "comparative_case_dative"

# UD ``case`` lemmas that are not prepositions (directional adverbs, …).
ADVERBIAL_CASE_MARKERS: frozenset[str] = frozenset({
    "abseits", "her", "hin", "hinaus", "je", "u.a.", "vorbei", "voll",
})

# Contracted ``case`` lemmas in treebanks → base preposition for classification.
_CASE_LEMMA_TO_PREP: Dict[str, str] = {
    "am": "an",
    "ans": "an",
    "aufs": "auf",
    "beim": "bei",
    "hinterm": "hinter",
    "hinters": "hinter",
    "im": "in",
    "ins": "in",
    "vom": "von",
    "vorm": "vor",
    "vors": "vor",
    "zum": "zu",
    "zur": "zu",
    "überm": "über",
    "übers": "über",
    "unterm": "unter",
    "unters": "unter",
}

# Locative/directional *-lich (nordwestlich, nördlich, …).
_DIRECTIONAL_CASE_RE = re.compile(
    r"^(?:nord|süd|ost|west|nordost|nordwest|südost|südwest)?"
    r"(?:östlich|westlich|nördlich|südlich)$",
    re.IGNORECASE,
)


def normalize_case_prep_lemma(lemma: Optional[str]) -> Optional[str]:
    if not lemma:
        return None
    low = lemma.lower().strip()
    return _CASE_LEMMA_TO_PREP.get(low, low)


def is_comparative_case_marker(lemma: Optional[str]) -> bool:
    if not lemma:
        return False
    low = normalize_case_prep_lemma(lemma) or ""
    return low in COMPARATIVE_CASE_MARKERS


def is_adverbial_case_marker(lemma: Optional[str]) -> bool:
    if not lemma:
        return False
    low = normalize_case_prep_lemma(lemma) or ""
    return low in ADVERBIAL_CASE_MARKERS or bool(_DIRECTIONAL_CASE_RE.match(low))


def is_target_dative_preposition(lemma: Optional[str]) -> bool:
    """True when ``lemma`` is one of the 16 study prepositions (after normalization)."""
    base = normalize_case_prep_lemma(lemma)
    return base in TARGET_DATIVE_PREPS if base else False


def is_proper_preposition(lemma: Optional[str]) -> bool:
    """Alias for :func:`is_target_dative_preposition` (notebook / legacy callers)."""
    return is_target_dative_preposition(lemma)


_COORD_CC_LEMMAS = frozenset({"und", "oder", "sowie", "beziehungsweise"})


def _deprel_base(deprel: Any) -> str:
    if not deprel or deprel == "_":
        return ""
    s = str(deprel).lower()
    return s.split(":")[0] if ":" in s else s


def _token_id_key(tid: Any) -> str:
    return str(tid)


def _coordination_partner_indices(
    tokens: List[Dict[str, Any]], phrase_head_idx: int
) -> Set[int]:
    """Token indices in the same coordinated NP as *phrase_head_idx* (``conj`` / shared parent)."""
    id_to_idx = {_token_id_key(t.get("id")): i for i, t in enumerate(tokens)}
    partners: Set[int] = set()

    def add_conj_component(idx: int) -> None:
        cur = idx
        seen: Set[int] = set()
        while 0 <= cur < len(tokens) and cur not in seen:
            seen.add(cur)
            partners.add(cur)
            tok = tokens[cur]
            if _deprel_base(tok.get("deprel")) != "conj":
                break
            parent_id = tok.get("head")
            if not parent_id:
                break
            pidx = id_to_idx.get(_token_id_key(parent_id))
            if pidx is not None:
                partners.add(pidx)
            for j, t in enumerate(tokens):
                if _deprel_base(t.get("deprel")) == "conj" and t.get("head") == parent_id:
                    partners.add(j)
            if pidx is None:
                break
            cur = pidx

    add_conj_component(phrase_head_idx)

    parent_id = tokens[phrase_head_idx].get("head")
    window = 12
    for cc_i, cc_tok in enumerate(tokens):
        if _deprel_base(cc_tok.get("deprel")) != "cc":
            continue
        if (cc_tok.get("lemma") or "").lower() not in _COORD_CC_LEMMAS:
            continue
        if cc_i < phrase_head_idx - window or cc_i > phrase_head_idx + window:
            continue
        for j, t in enumerate(tokens):
            if parent_id and t.get("head") != parent_id:
                continue
            if _deprel_base(t.get("deprel")) not in {"conj", "nmod"}:
                continue
            if j in partners:
                continue
            if j < cc_i < phrase_head_idx or phrase_head_idx < cc_i < j:
                partners.add(j)

    return partners


def coordination_case_head_ids(
    tokens: List[Dict[str, Any]],
    span_start: int,
    span_end: int,
    phrase_head_idx: int,
) -> Set[Any]:
    """Token ids whose ``case`` dependents govern this span (incl. coordinated conjuncts)."""
    _ = span_start, span_end
    partners = _coordination_partner_indices(tokens, phrase_head_idx)
    return {
        tokens[i].get("id")
        for i in partners
        if 0 <= i < len(tokens) and tokens[i].get("id")
    }


def _case_lemmas_for_head_ids(
    tokens: List[Dict[str, Any]], head_ids: Set[Any]
) -> List[str]:
    """``case``-dependent lemmas for any of the given head token ids."""
    case_lemmas: List[str] = []
    for tok in tokens:
        if tok.get("head") in head_ids and tok.get("deprel") == "case":
            lemma = tok.get("lemma")
            if lemma:
                case_lemmas.append(lemma)
    return case_lemmas


def _pick_governing_case_lemma(case_lemmas: List[str]) -> Optional[str]:
    """Choose one ``case`` lemma from several dependents of the same NP head."""
    if not case_lemmas:
        return None

    last_target: Optional[str] = None
    for lemma in case_lemmas:
        if is_target_dative_preposition(lemma):
            last_target = lemma
    if last_target is not None:
        return last_target

    last_non_adverbial: Optional[str] = None
    for lemma in case_lemmas:
        if not is_adverbial_case_marker(lemma):
            last_non_adverbial = lemma
    if last_non_adverbial is not None:
        return last_non_adverbial

    return case_lemmas[-1]


def case_marker_lemma_for_head_id(
    tokens: List[Dict[str, Any]], head_id: Any
) -> Optional[str]:
    """Pick the ``case`` lemma governing the NP headed by *head_id*."""
    return _pick_governing_case_lemma(_case_lemmas_for_head_ids(tokens, {head_id}))


def _case_marker_mit_und_list(
    tokens: List[Dict[str, Any]], span_start: int, span_end: int
) -> Optional[str]:
    """``mit X und Y`` — span is *Y*; inherit ``mit`` from the left of ``und``."""
    for cc_i, cc_tok in enumerate(tokens):
        if (cc_tok.get("lemma") or "").lower() != "und":
            continue
        if _deprel_base(cc_tok.get("deprel")) != "cc":
            continue
        if cc_i >= span_start or span_end < cc_i:
            continue
        for j in range(cc_i - 1, -1, -1):
            t = tokens[j]
            if t.get("deprel") == "case" and t.get("upos") == "ADP":
                prep = t.get("lemma")
                if prep and is_target_dative_preposition(prep):
                    return prep
                return None
    return None


def _zu_marker_wegen_degree(
    tokens: List[Dict[str, Any]], span_start: int, span_end: int
) -> Optional[str]:
    """``wegen zu geringer Nachfrage`` — ``zu`` is ``advmod`` on the adjective, not ``case``."""
    span_ids = {
        tokens[i].get("id") for i in range(span_start, span_end + 1) if tokens[i].get("id")
    }
    has_wegen = any(
        t.get("lemma") == "wegen"
        and t.get("deprel") == "case"
        and t.get("head") in span_ids
        for t in tokens
    )
    if not has_wegen:
        return None
    for i in range(span_start, span_end + 1):
        head_id = tokens[i].get("id")
        for t in tokens:
            if t.get("lemma") != "zu":
                continue
            if _deprel_base(t.get("deprel")) not in ("advmod", "mark"):
                continue
            if t.get("head") == head_id:
                return "zu"
    return None


def case_marker_lemma_for_span(
    tokens: List[Dict[str, Any]],
    span_start: int,
    span_end: int,
    *,
    phrase_head_idx: Optional[int] = None,
) -> Optional[str]:
    """``case`` lemma for the NP span, including coordinated conjuncts."""
    head_ids = {
        tokens[i].get("id")
        for i in range(span_start, span_end + 1)
        if tokens[i].get("id")
    }
    if phrase_head_idx is not None:
        head_ids |= coordination_case_head_ids(
            tokens, span_start, span_end, phrase_head_idx
        )
    picked = _pick_governing_case_lemma(_case_lemmas_for_head_ids(tokens, head_ids))
    if picked is not None:
        return picked
    return _case_marker_mit_und_list(tokens, span_start, span_end)


def case_marker_lemma_for_head(
    tokens: List[Dict[str, Any]], head_idx: int
) -> Optional[str]:
    return case_marker_lemma_for_head_id(tokens, tokens[head_idx].get("id"))


def preposition_lemma_for_head(
    tokens: List[Dict[str, Any]], head_idx: int
) -> Optional[str]:
    """Base lemma of a target preposition (None for other ``case`` markers)."""
    case_lemma = case_marker_lemma_for_head(tokens, head_idx)
    if not is_target_dative_preposition(case_lemma):
        return None
    return normalize_case_prep_lemma(case_lemma)


def classify_dative_type(token: Dict[str, Any], case_lemma: Optional[str]) -> str:
    """Classify a dative head token (same rules as UD_German.ipynb)."""
    if case_lemma is not None:
        if is_comparative_case_marker(case_lemma):
            return COMPARATIVE_CASE_DATIVE_TYPE
        if is_adverbial_case_marker(case_lemma):
            return "adverbial_case_dative"
        base = normalize_case_prep_lemma(case_lemma)
        if base in TARGET_DATIVE_PREP_TYPES:
            return TARGET_DATIVE_PREP_TYPES[base]
        return "other_prepositional_dative"
    rel = token.get("deprel") or ""
    if rel in {"obj", "obl:arg"}:
        return "core_dative_argument"
    if rel == "obl":
        return "oblique_dative"
    if rel == "nmod":
        return "nominal_dative_modifier"
    return "other_dative"


def load_dative_type_lookup(csv_path: str) -> Dict[DativeTypeKey, str]:
    """Build (split, sent_id, head_lemma, head_form, relation) → dative_type from CSV."""
    lookup: Dict[DativeTypeKey, str] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key: DativeTypeKey = (
                row["split"],
                row["sent_id"],
                row["head_lemma"],
                row["head_form"],
                row.get("relation") or "",
            )
            lookup[key] = row["dative_type"]
    return lookup


_DAT_HEAD_UPOS = frozenset({"NOUN", "PROPN", "PRON"})


def dative_nominal_head_index(
    tokens: List[Dict[str, Any]],
    span_start: int,
    span_end: int,
    *,
    fallback_idx: int,
) -> int:
    """Index of the UD phrase head inside the span (e.g. *Dezember* for *dem 31. Dezember*).

    Walk up from the seed and from each span token via ``det`` / ``amod`` / … so the
    head is the noun/proper noun that dependents attach to, even when it is tagged
    ``Case=Nom`` in isolation (dates after *bis zum …*) or the seed is ``dem``.
    """
    origins = [fallback_idx] + [
        i for i in range(span_start, span_end + 1) if i != fallback_idx
    ]
    for origin in origins:
        if not 0 <= origin < len(tokens):
            continue
        phrase_head = walk_np_head_index(tokens, origin)
        if span_start <= phrase_head <= span_end:
            return phrase_head
    for i in range(span_start, span_end + 1):
        tok = tokens[i]
        feats = tok.get("feats", {}) or {}
        if tok.get("upos") in _DAT_HEAD_UPOS and feats.get("Case") == "Dat":
            return i
    return fallback_idx


def resolve_dative_type(
    *,
    split: str,
    sent_id: str,
    tokens: List[Dict[str, Any]],
    span_start: int,
    span_end: int,
    seed_idx: int,
    lookup: Optional[Dict[DativeTypeKey, str]] = None,
) -> str:
    """Classify the dative NP span from UD structure (live; ``lookup`` ignored)."""
    _ = lookup, split, sent_id
    head_idx = dative_nominal_head_index(
        tokens, span_start, span_end, fallback_idx=seed_idx
    )
    token = tokens[head_idx]
    case_lemma = case_marker_lemma_for_head(tokens, head_idx)
    if case_lemma is None:
        case_lemma = case_marker_lemma_for_span(
            tokens, span_start, span_end, phrase_head_idx=head_idx
        )
    if case_lemma is None:
        case_lemma = _case_marker_mit_und_list(tokens, span_start, span_end)
    zu_wegen = _zu_marker_wegen_degree(tokens, span_start, span_end)
    if zu_wegen is not None:
        case_lemma = zu_wegen
    return classify_dative_type(token, case_lemma)


# ---------------------------------------------------------------------------
# Unified case-conversion engine  (Acc↔Dat, parameterised by direction)
# ---------------------------------------------------------------------------


@dataclass
class CaseConversionPair:
    """One gold → corrupted example produced by converting *source_case* tokens
    in an NP span to *target_case* surface forms."""

    direction: str  # e.g. "Acc→Dat" or "Dat→Acc"
    split: str
    sent_id: str
    gold_text: str
    corrupt_group: str
    token_index: int
    form: str
    lemma: str
    upos: str
    number: str
    gender: str
    group_start_index: Optional[int]
    group_end_index: Optional[int]
    group_span_text: str
    group_span_char_start: Optional[int]
    group_span_char_end: Optional[int]
    dative_type: Optional[str] = None
    corrupted_tokens: Tuple[Dict[str, Any], ...] = ()
    target_span_tokens: Tuple[Dict[str, Any], ...] = ()
    gold_span_tokens: Tuple[Dict[str, Any], ...] = ()


def is_source_case_token(token: Dict[str, Any], source_case: str) -> bool:
    """True when *token* carries ``Case=<source_case>``."""
    feats = token.get("feats", {}) or {}
    return feats.get("Case") == source_case


def token_to_target_surface(
    form: str,
    lemma: str,
    upos: str,
    feats: Dict[str, Any],
    case_dict: dict,
    target_case: str,
    *,
    tokens: Optional[List[Mapping[str, Any]]] = None,
    token: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Look up the *target_case* surface form for a token via the case dictionary."""
    number = feats.get("Number")
    gender = ud_gender_from_feats(feats)
    return select_dict_surface_form(
        case_dict=case_dict,
        lemma=lemma,
        target_case=target_case,
        number=number,
        ud_gender=gender,
        upos=upos,
        orig_form=form,
        tokens=tokens,
        token=token,
    )


def dropout_reason_for_token(
    lemma: str,
    upos: str,
    feats: Dict[str, Any],
    case_dict: dict,
    target_case: str,
) -> str:
    """Return a short code explaining why *target_case* lookup failed."""
    lookup_lemma = dictionary_lookup_lemma(lemma, upos)
    if lookup_lemma not in case_dict:
        if lemma_skipped_by_dictionary_extract(lemma):
            return "lemma_skipped_non_german_initial_identity_form"
        return "lemma_not_in_dictionary"
    number = feats.get("Number")
    gender = ud_gender_from_feats(feats)
    entries_list = case_dict[lookup_lemma]
    if isinstance(entries_list, dict):
        entries_list = [entries_list]

    has_target_any = any(e.get("Case") == target_case for e in entries_list)
    if not has_target_any:
        return f"lemma_has_no_{target_case}_rows"

    cand = dict_entries_matching_lookup(
        entries_list,
        target_case=target_case,
        number=number,
        ud_gender=gender,
        upos=upos,
    )
    if cand:
        forms_union: Set[str] = set()
        for e in cand:
            forms_union.update(e.get("forms") or set())
        if not forms_union:
            return f"{target_case}_rows_match_feats_but_empty_Forms"
        return "lookup_failed_unexpected_matching_rows_have_forms"

    n_match = sum(
        1
        for e in entries_list
        if e.get("Case") == target_case
        and number_matches_dictionary(number, e.get("Number"))
    )
    if n_match == 0:
        return f"no_{target_case}_row_matching_number"

    g_match = sum(
        1
        for e in entries_list
        if e.get("Case") == target_case
        and number_matches_dictionary(number, e.get("Number"))
        and gender_matches_dictionary(gender, e.get("Gender"))
    )
    if g_match == 0 and not _is_plural_number(number):
        return f"no_{target_case}_row_matching_number_gender"
    if _is_plural_number(number):
        plur = _dict_entries_filter(
            entries_list,
            target_case=target_case,
            number=number,
            ud_gender=gender,
            upos=upos,
            require_gender=False,
            require_no_gender=True,
        )
        if not plur:
            return f"no_{target_case}_row_matching_number_plur"
        return f"no_{target_case}_row_matching_number_gender_upos"
    return f"no_{target_case}_row_matching_number_gender_upos"


def first_unconverted_token(
    tokens: List[Dict[str, Any]],
    case_dict: dict,
    source_case: str,
    target_case: str,
    *,
    span_indices: Optional[Tuple[int, ...]] = None,
    start: int = 0,
    end: Optional[int] = None,
    allow_missing_for_upos: Optional[Set[str]] = None,
) -> Optional[Tuple[int, str, str, str, str, str, str]]:
    """Return info about the first *source_case* token in the span that has no
    *target_case* dictionary match, or ``None`` if every such token is convertible."""
    if span_indices is None:
        if end is None:
            end = len(tokens) - 1
        indices: Tuple[int, ...] = tuple(range(start, end + 1))
    else:
        indices = span_indices
    for i in indices:
        tok = tokens[i]
        if not is_source_case_token(tok, source_case):
            continue
        feats = tok.get("feats", {}) or {}
        form = tok.get("form", "")
        lemma = tok.get("lemma", "")
        upos = tok.get("upos", "")
        if token_to_target_surface(form, lemma, upos, feats, case_dict, target_case) is None:
            if allow_missing_for_upos and upos in allow_missing_for_upos:
                continue
            reason = dropout_reason_for_token(lemma, upos, feats, case_dict, target_case)
            g = feats.get("Gender") or ""
            n = feats.get("Number") or ""
            return (i, form, lemma, upos, str(n), str(g), reason)
    return None


def make_group_corrupt(
    tokens: List[Dict[str, Any]],
    head_idx: int,
    case_dict: dict,
    gold_text: str,
    source_case: str,
    target_case: str,
    *,
    allow_missing_for_upos: Optional[Set[str]] = None,
) -> Tuple[
    Optional[str],
    int,
    int,
    Tuple[Dict[str, Any], ...],
    Tuple[Dict[str, Any], ...],
]:
    """Convert all *source_case* tokens inside the NP span of *head_idx* to
    *target_case* forms.

    Returns ``(corrupted_text | None, span_start, span_end, changed_tokens,
    target_span_tokens)`` where *target_span_tokens* holds every converted token's
    new surface (and *changed_tokens* is the subset whose form actually changed).
    """
    span_indices = find_nominal_group_indices(tokens, head_idx)
    start, end = span_indices[0], span_indices[-1]
    base_forms = [t.get("form", "") for t in tokens]

    idx_to_new_form: Dict[int, str] = {}
    for idx in span_indices:
        tok = tokens[idx]
        feats = tok.get("feats", {}) or {}
        if feats.get("Case") != source_case:
            continue
        upos = tok.get("upos", "")
        lemma = tok.get("lemma", "")
        form = base_forms[idx]
        new_form = token_to_target_surface(
            form,
            lemma,
            upos,
            feats,
            case_dict,
            target_case,
            tokens=tokens,
            token=tok,
        )
        if new_form is None:
            if allow_missing_for_upos and upos in allow_missing_for_upos:
                continue
            return None, start, end, (), ()
        idx_to_new_form[idx] = new_form

    target_span_tokens = tuple(
        _token_morph_snapshot(tokens[idx], form=new_form)
        for idx, new_form in sorted(idx_to_new_form.items())
    )
    changed_tokens = tuple(
        _token_morph_snapshot(tokens[idx], form=new_form)
        for idx, new_form in sorted(idx_to_new_form.items())
        if new_form.lower() != base_forms[idx].lower()
    )

    # Contraction-aware replacement
    if gold_text:
        result = replace_tokens_in_gold_text(gold_text, tokens, idx_to_new_form)
        if result is not None:
            return result, start, end, changed_tokens, target_span_tokens

    # Fallback: reconstruct from tokens
    new_forms = base_forms.copy()
    for idx, nf in idx_to_new_form.items():
        new_forms[idx] = nf
    tmp_tokens = [dict(t, form=f) for t, f in zip(tokens, new_forms)]
    # _logger.info(f"\n### New forms: {new_forms}\n")
    return tokens_to_text(tmp_tokens), start, end, changed_tokens, target_span_tokens


def iter_case_conversion_pairs(
    dataset: Dict[str, List[Dict[str, Any]]],
    case_dict: dict,
    source_case: str,
    target_case: str,
    direction_label: str,
    splits: Iterable[str] = ("train", "dev", "test"),
    report: Optional[SpanEliminationReport] = None,
    allow_missing_for_upos_fn: Optional[Callable[[str], Optional[Set[str]]]] = None,
    unconverted_records: Optional[List[Dict[str, Any]]] = None,
    dative_type_lookup: Optional[Dict[DativeTypeKey, str]] = None,
) -> Iterable[CaseConversionPair]:
    """Yield :class:`CaseConversionPair` objects for every *source_case* NP span
    in *dataset* that can be corrupted to *target_case*.

    *allow_missing_for_upos_fn*, when provided, receives the split name and
    returns a set of UPOS tags whose missing conversions are tolerated (or None).

    *unconverted_records*, when provided, is a list that will be populated with
    dicts describing each NP span that could not be fully converted.
    """
    for split in splits:
        if split not in dataset:
            if report is not None:
                report.splits_requested_but_missing_dataset.append(split)
            _logger.warning("Split %r absent from dataset — skipping.", split)
            continue

        allow_missing = (
            allow_missing_for_upos_fn(split) if allow_missing_for_upos_fn else None
        )

        for sent in dataset[split]:
            sent = dict(sent)
            raw_tok = sent.get("tokens")
            if raw_tok is None:
                tokens: List[Dict[str, Any]] = []
            elif isinstance(raw_tok, list):
                tokens = raw_tok
            else:
                tokens = list(raw_tok)

            gold_text = sent.get("text") or ""
            sent_id = str(sent.get("sent_id", "") or "")
            preview = gold_text.replace("\n", " ")[:120]

            if not tokens:
                if report is not None:
                    report.sentences_empty_tokens += 1
                    report._push_sample(
                        report.sample_sentence_dropout,
                        f"EMPTY_TOKENS split={split!r} sent_id={sent_id!r} "
                        f"text_preview={preview!r}",
                    )
                continue

            if not gold_text:
                gold_text = tokens_to_text(tokens)

            span_unconverted_seen: Set[Tuple[int, int]] = set()
            emitted_spans: List[Tuple[int, int]] = []

            def _span_is_proper_subset(
                start: int, end: int, other_start: int, other_end: int
            ) -> bool:
                return (
                    other_start <= start
                    and end <= other_end
                    and (other_start, other_end) != (start, end)
                )

            def _span_dominated_by_emitted(start: int, end: int) -> bool:
                for os, oe in emitted_spans:
                    if (os, oe) == (start, end):
                        return True
                    if _span_is_proper_subset(start, end, os, oe):
                        return True
                return False

            # --- per-token iteration ---
            for idx, tok in enumerate(tokens):
                if not is_source_case_token(tok, source_case):
                    continue

                if report is not None:
                    report.accusative_seed_tokens_seen += 1

                span_indices = find_nominal_group_indices(tokens, idx)
                group_start, group_end = span_indices[0], span_indices[-1]

                (
                    corrupt_group_raw,
                    group_start,
                    group_end,
                    changed_tokens,
                    target_span_tokens,
                ) = make_group_corrupt(
                    tokens,
                    idx,
                    case_dict,
                    gold_text,
                    source_case,
                    target_case,
                    allow_missing_for_upos=allow_missing,
                )
                if corrupt_group_raw is None:
                    span_key = (group_start, group_end)
                    bad = first_unconverted_token(
                        tokens,
                        case_dict,
                        source_case,
                        target_case,
                        span_indices=span_indices,
                        allow_missing_for_upos=allow_missing,
                    )
                    if bad is not None and span_key not in span_unconverted_seen:
                        span_unconverted_seen.add(span_key)
                        _ti, form, lemma, upos, num, gender, reason = bad
                        if report is not None:
                            report.group_dropped_span_unconverted += 1
                            report.dropout_by_reason[reason] += 1
                            report._push_sample(
                                report.sample_group_np_substituted_gold,
                                f"SPAN_SKIP_UNCONVERTED split={split!r} "
                                f"sent_id={sent_id!r} seed_idx={idx} "
                                f"np_span=[{group_start},{group_end}] "
                                f"first_fail_idx={_ti} form={form!r} "
                                f"lemma={lemma!r} reason={reason!r} "
                                f"preview={preview!r}",
                            )
                        if unconverted_records is not None:
                            unconverted: Dict[str, Any] = {
                                "split": split,
                                "sent_id": sent_id,
                                "gold_text": gold_text,
                                "seed_token_index": idx,
                                "group_start": group_start,
                                "group_end": group_end,
                                "group_span_text": group_span_surface_text(
                                    gold_text, tokens, span_indices
                                ),
                                "first_fail_token_index": _ti,
                                "first_fail_form": form,
                                "first_fail_lemma": lemma,
                                "first_fail_upos": upos,
                                "first_fail_number": num,
                                "first_fail_gender": gender,
                                "reason": reason,
                            }
                            if source_case == "Dat":
                                unconverted["dative_type"] = resolve_dative_type(
                                    split=split,
                                    sent_id=sent_id,
                                    tokens=tokens,
                                    span_start=group_start,
                                    span_end=group_end,
                                    seed_idx=idx,
                                    lookup=dative_type_lookup,
                                )
                            unconverted_records.append(unconverted)
                    continue
                else:
                    corrupt_group = fix_unmerged_contractions(
                        gold_text, corrupt_group_raw
                    )

                if _span_dominated_by_emitted(group_start, group_end):
                    if report is not None:
                        report.group_dropped_duplicate_span_key += 1
                        report._push_sample(
                            report.sample_group_dedup,
                            f"SUBSET_SPAN split={split!r} sent_id={sent_id!r} "
                            f"seed_idx={idx} np_span=[{group_start},{group_end}] "
                            f"preview={preview!r}",
                        )
                    continue

                group_span_text = group_span_surface_text(
                    gold_text, tokens, span_indices
                )
                gch0, gch1 = gold_char_span_for_tokens(
                    gold_text, tokens, group_start, group_end
                )

                feats_out = tok.get("feats", {}) or {}
                if report is not None:
                    report.pairs_yielded_from_iterator += 1

                dative_type: Optional[str] = None
                if source_case == "Dat":
                    dative_type = resolve_dative_type(
                        split=split,
                        sent_id=sent_id,
                        tokens=tokens,
                        span_start=group_start,
                        span_end=group_end,
                        seed_idx=idx,
                        lookup=dative_type_lookup,
                    )

                emitted_spans.append((group_start, group_end))

                yield CaseConversionPair(
                    direction=direction_label,
                    split=split,
                    sent_id=sent_id,
                    gold_text=gold_text,
                    corrupt_group=corrupt_group,
                    token_index=idx,
                    form=tok.get("form", ""),
                    lemma=tok.get("lemma", ""),
                    upos=tok.get("upos", ""),
                    number=feats_out.get("Number", ""),
                    gender=feats_out.get("Gender") or "",
                    group_start_index=group_start,
                    group_end_index=group_end,
                    group_span_text=group_span_text,
                    group_span_char_start=gch0,
                    group_span_char_end=gch1,
                    dative_type=dative_type,
                    corrupted_tokens=changed_tokens,
                    target_span_tokens=target_span_tokens,
                    gold_span_tokens=tuple(
                        _token_morph_snapshot(tokens[i]) for i in span_indices
                    ),
                )


# ---------------------------------------------------------------------------
# Shared record / dedup / main helpers
# ---------------------------------------------------------------------------


def pair_to_group_record(pair: CaseConversionPair) -> Optional[Dict[str, Any]]:
    """Serialise a :class:`CaseConversionPair` as a group-level dict (or ``None``)."""
    gs = pair.group_start_index
    ge = pair.group_end_index
    if gs is None or ge is None or ge < gs:
        return None
    record: Dict[str, Any] = {
        "split": pair.split,
        "sent_id": pair.sent_id,
        "group_start": gs,
        "group_end": ge,
        "group_span_text": pair.group_span_text,
        "gold_text": pair.gold_text,
        "corrupted": pair.corrupt_group,
    }
    if pair.dative_type is not None:
        record["dative_type"] = pair.dative_type
    return record


def pair_to_group_duplicates_record(
    pair: CaseConversionPair,
    case_dict: dict,
    target_case: str,
) -> Optional[Dict[str, Any]]:
    """Group record for the duplicates JSON when the span has multi-form lookup rows."""
    record = pair_to_group_record(pair)
    if record is None:
        return None
    record["gold_span_duplicate_forms"] = gold_span_duplicate_surface_forms(
        pair.gold_span_tokens,
        pair.target_span_tokens,
        case_dict,
        target_case,
    )
    if not record["gold_span_duplicate_forms"]:
        return None
    return record


def group_record_dedup_key(
    pair: CaseConversionPair,
) -> Tuple[str, str, int, int, str]:
    """Tuple key for deduplicating group-level records."""
    sid = pair.sent_id or pair.gold_text
    gs = pair.group_start_index if pair.group_start_index is not None else -1
    ge = pair.group_end_index if pair.group_end_index is not None else -1
    return (pair.split, sid, gs, ge, pair.corrupt_group)


def write_dative_category_pair_files(
    group_records: List[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, int]:
    """Write one minimal-pair JSON per :data:`DATIVE_TYPE_LABELS` entry."""
    by_type: Dict[str, List[Dict[str, Any]]] = {
        label: [] for label in DATIVE_TYPE_LABELS
    }
    for record in group_records:
        dt = record.get("dative_type")
        if dt in by_type:
            by_type[dt].append(record)

    os.makedirs(output_dir, exist_ok=True)
    counts: Dict[str, int] = {}
    for label in DATIVE_TYPE_LABELS:
        path = os.path.join(output_dir, f"{label}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(by_type[label], f, ensure_ascii=False, indent=2)
        counts[label] = len(by_type[label])
    return counts


def run_pair_generation_main(
    *,
    ud_base_path: str,
    splits: Optional[Mapping[str, str]] = None,
    output_group_json: str,
    dict_csv_path: str,
    source_case: str,
    target_case: str,
    direction_label: str,
    log_level: str = "INFO",
    allow_missing_for_upos_fn: Optional[Callable[[str], Optional[Set[str]]]] = None,
    output_unconverted_json: Optional[str] = None,
    dative_types_csv_path: Optional[str] = None,
    duplicates_dict_csv_path: Optional[str] = None,
    output_group_duplicates_json: Optional[str] = None,
    output_dative_categories_dir: Optional[str] = None,
) -> SpanEliminationReport:
    """Shared main-loop: load data → iterate pairs → deduplicate → write JSON.

    Both ``corrupt_acc_dat_pairs`` and ``corrupt_dat_acc_pairs``
    delegate here after resolving their direction-specific defaults.

    When *output_unconverted_json* is provided, NP spans that could not be fully
    converted are written to that file for inspection.

    When *output_group_duplicates_json* is set, pairs where some token in the NP span
    (gold or corrupted) has multiple forms in the main lookup dictionary are also
    written there with ``gold_span_duplicate_forms`` listing those surfaces.

    For Dat→Acc runs with dative typing, minimal pairs are also split into one JSON
    per :data:`DATIVE_TYPE_LABELS` label under *output_dative_categories_dir*
    (default: ``<parent of output_group_json>/dative_categories``).
    """
    from load_ud_dataset import load_ud_dataset, UD_GERMAN_GSD_SPLITS
    from german_ud_lookup_dict_load import load_case_dictionary

    lvl_name = log_level.strip().upper() or "INFO"
    try:
        log_level_int = getattr(logging, lvl_name)
    except AttributeError:
        log_level_int = logging.INFO
    logging.basicConfig(
        level=log_level_int,
        format="%(levelname)s %(name)s %(message)s",
    )
    if splits is None:
        splits = UD_GERMAN_GSD_SPLITS
    print(f"📁 Loading UD German-GSD from: {ud_base_path}")
    dataset = load_ud_dataset(ud_base_path, splits_filemap=splits)

    print(f"📁 Loading dictionary from: {dict_csv_path}")
    if not os.path.exists(dict_csv_path):
        raise FileNotFoundError(f"Case dictionary required: {dict_csv_path}")
    case_dict = load_case_dictionary(dict_csv_path)
    attach_preferred_forms(case_dict)

    dative_type_lookup: Optional[Dict[DativeTypeKey, str]] = None
    if dative_types_csv_path:
        if not os.path.exists(dative_types_csv_path):
            raise FileNotFoundError(
                f"Dative types CSV required: {dative_types_csv_path}"
            )
        print(f"📁 Loading dative types from: {dative_types_csv_path}")
        dative_type_lookup = load_dative_type_lookup(dative_types_csv_path)

    parent = os.path.dirname(output_group_json)
    if parent:
        os.makedirs(parent, exist_ok=True)

    report = SpanEliminationReport(direction=direction_label)
    group_records: List[Dict[str, Any]] = []
    group_seen: Set[Tuple[str, str, int, int, str]] = set()
    group_records_duplicates: List[Dict[str, Any]] = []
    group_seen_duplicates: Set[Tuple[str, str, int, int, str]] = set()
    unconverted_records: List[Dict[str, Any]] = []
    dative_type_counts: Counter[str] = Counter()

    for pair in iter_case_conversion_pairs(
        dataset,
        case_dict,
        source_case=source_case,
        target_case=target_case,
        direction_label=direction_label,
        splits=("train", "dev", "test", "deneme"),
        report=report,
        allow_missing_for_upos_fn=allow_missing_for_upos_fn,
        unconverted_records=unconverted_records,
        dative_type_lookup=dative_type_lookup,
    ):
        g_norm_gold = normalize_german_for_pair_compare(pair.gold_text)
        g_norm_corr = normalize_german_for_pair_compare(pair.corrupt_group)

        if g_norm_corr != g_norm_gold:
            gr = pair_to_group_record(pair)
            if gr is None:
                gs = pair.group_start_index
                ge = pair.group_end_index
                if gs is None or ge is None or ge < gs:
                    report.dropout_by_reason["group_record_invalid_gs_ge"] += 1
                continue
            gkey = group_record_dedup_key(pair)
            if gkey in group_seen:
                report.group_dropped_duplicate_span_key += 1
                report._push_sample(
                    report.sample_group_dedup,
                    f"DUP_KEY split={pair.split!r} sent_id={pair.sent_id!r} "
                    f"NP_span=[{pair.group_start_index},{pair.group_end_index}] "
                    f"key_fragment={pair.corrupt_group[:80]!r}",
                )
                continue
            group_seen.add(gkey)
            group_records.append(gr)
            report.group_records_written += 1
            if output_group_duplicates_json:
                if gkey not in group_seen_duplicates:
                    dup_gr = pair_to_group_duplicates_record(
                        pair, case_dict, target_case
                    )
                    if dup_gr is not None:
                        group_seen_duplicates.add(gkey)
                        group_records_duplicates.append(dup_gr)
            dt = gr.get("dative_type")
            if dt:
                dative_type_counts[dt] += 1
        else:
            report.group_dropped_normalized_equals_gold += 1

    with open(output_group_json, "w", encoding="utf-8") as f:
        json.dump(group_records, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Wrote {len(group_records)} group-level {direction_label} pairs "
        f"to: {output_group_json}"
    )

    if output_group_duplicates_json:
        dup_parent = os.path.dirname(output_group_duplicates_json)
        if dup_parent:
            os.makedirs(dup_parent, exist_ok=True)
        with open(output_group_duplicates_json, "w", encoding="utf-8") as f:
            json.dump(group_records_duplicates, f, ensure_ascii=False, indent=2)
        print(
            f"✅ Wrote {len(group_records_duplicates)} group-level {direction_label} "
            f"pairs with multi-form lookup entries to: "
            f"{output_group_duplicates_json}"
        )

    if source_case == "Dat" and dative_type_lookup is not None:
        categories_dir = output_dative_categories_dir
        if categories_dir is None:
            categories_dir = os.path.join(parent or ".", "dative_categories")
        category_counts = write_dative_category_pair_files(
            group_records, categories_dir
        )
        written = sum(category_counts.values())
        print(
            f"✅ Wrote {written} Dat→Acc minimal pairs across "
            f"{len(DATIVE_TYPE_LABELS)} dative_type files under: {categories_dir}"
        )

    if dative_type_counts:
        total_dt = sum(dative_type_counts.values())
        print(f"--- dative_type counts in corrupted testset ({total_dt} spans) ---")
        for label in DATIVE_TYPE_LABELS:
            n = dative_type_counts.get(label, 0)
            pct = 100.0 * n / total_dt if total_dt else 0.0
            print(f"  {label}: {n} ({pct:.1f}%)")
        other_labels = sorted(
            k for k in dative_type_counts if k not in DATIVE_TYPE_LABELS
        )
        for label in other_labels:
            n = dative_type_counts[label]
            pct = 100.0 * n / total_dt
            print(f"  {label}: {n} ({pct:.1f}%)")

    # --- write unconverted sentences ---
    if output_unconverted_json and unconverted_records:
        u_parent = os.path.dirname(output_unconverted_json)
        if u_parent:
            os.makedirs(u_parent, exist_ok=True)
        with open(output_unconverted_json, "w", encoding="utf-8") as f:
            json.dump(unconverted_records, f, ensure_ascii=False, indent=2)
        print(
            f"⚠️  Wrote {len(unconverted_records)} unconverted spans "
            f"to: {output_unconverted_json}"
        )

    summary_text = "\n".join(report.summary_lines())
    print("\n--- elimination summary ---\n" + summary_text + "\n")
    return report
