"""
Nominal phrase span detection for UD German token lists.

Used by the Accusative/Dative pair builders: a ``Case=Acc`` / ``Case=Dat`` seed
token is expanded to the full NP by (1) walking *up* along ``det`` / ``amod`` /
``flat`` / … edges to the phrase head, then (2) collecting *downward*
dependents of that head while skipping prepositional ``case`` markers,
finite/infinitival ``acl`` / ``advcl`` / … branches, and ``nmod`` phrases whose
head noun is not genitive (so dative ``nmod`` like *von einem …* stays out).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

# Edges that stay inside the same nominal phrase when walking token -> head.
_WALK_UP_DEPS = frozenset(
    {"det", "nummod", "amod", "compound", "flat", "fixed", "goeswith", "appos"}
)

# Dependents we never attach to the NP head (clauses, coordination hooks, …).
_BLOCKED_DOWN_DEPS = frozenset(
    {
        "acl",
        "advcl",
        "ccomp",
        "xcomp",
        "parataxis",
        "conj",
        "list",
        "dislocated",
        "orphan",
        "vocative",
        "discourse",
        "dep",
    }
)


def _deprel_base(deprel: Any) -> str:
    if not deprel or deprel == "_":
        return ""
    s = str(deprel).lower()
    if ":" in s:
        return s.split(":")[0]
    return s


def _token_id_key(tid: Any) -> str:
    return str(tid)


def _id_to_index(tokens: List[Dict[str, Any]]) -> Dict[str, int]:
    return {_token_id_key(t.get("id")): i for i, t in enumerate(tokens)}


def _parent_index(
    tokens: List[Dict[str, Any]], tok: Dict[str, Any], id_to_idx: Dict[str, int]
) -> Optional[int]:
    hid = tok.get("head")
    if hid in (None, "_", 0, "0"):
        return None
    return id_to_idx.get(_token_id_key(hid))


def walk_np_head_index(tokens: List[Dict[str, Any]], seed_idx: int) -> int:
    """
    From *seed_idx*, follow ``head`` while ``deprel`` ∈ internal nominal deps
    (``det``, ``amod``, ``flat``, …).  Stop at the phrase head (first edge that
    is not internal, or at the root).
    """
    id_to_idx = _id_to_index(tokens)
    idx = seed_idx
    seen: Set[int] = set()
    while idx not in seen:
        seen.add(idx)
        tok = tokens[idx]
        drel = _deprel_base(tok.get("deprel"))
        pidx = _parent_index(tokens, tok, id_to_idx)
        if pidx is None:
            return idx
        if drel not in _WALK_UP_DEPS:
            return idx
        idx = pidx
    return idx


def _nominal_head_case(
    tokens: List[Dict[str, Any]], start_idx: int, id_to_idx: Dict[str, int]
) -> Optional[str]:
    """Return ``feats['Case']`` of the first ``NOUN``/``PROPN`` when walking up with *_WALK_UP_DEPS*."""
    idx = start_idx
    seen: Set[int] = set()
    while idx not in seen:
        seen.add(idx)
        upos = tokens[idx].get("upos", "")
        feats = tokens[idx].get("feats") or {}
        if upos in {"NOUN", "PROPN"}:
            return feats.get("Case")
        tok = tokens[idx]
        drel = _deprel_base(tok.get("deprel"))
        pidx = _parent_index(tokens, tok, id_to_idx)
        if pidx is None or drel not in _WALK_UP_DEPS:
            return feats.get("Case")
        idx = pidx
    return None


def _down_edge_ok(
    tokens: List[Dict[str, Any]],
    child_idx: int,
    parent_idx: int,
    id_to_idx: Dict[str, int],
) -> bool:
    child = tokens[child_idx]
    if child.get("upos") == "PUNCT":
        return False
    if _token_id_key(child.get("head")) != _token_id_key(tokens[parent_idx].get("id")):
        return False
    drel = _deprel_base(child.get("deprel"))
    upos = child.get("upos", "")
    if drel == "case" and upos == "ADP":
        return False
    if drel in _BLOCKED_DOWN_DEPS:
        return False
    if drel == "nmod":
        return _nominal_head_case(tokens, child_idx, id_to_idx) == "Gen"
    return drel in _WALK_UP_DEPS


def _collect_np_indices(tokens: List[Dict[str, Any]], head_idx: int) -> Set[int]:
    id_to_idx = _id_to_index(tokens)
    collected: Set[int] = {head_idx}
    stack = [head_idx]
    while stack:
        p = stack.pop()
        for j, t in enumerate(tokens):
            if j in collected:
                continue
            if not _down_edge_ok(tokens, j, p, id_to_idx):
                continue
            collected.add(j)
            stack.append(j)
    return collected


def find_nominal_group_span(tokens: List[Dict[str, Any]], seed_idx: int) -> Tuple[int, int]:
    """
    Return inclusive token indices ``(start, end)`` for the nominal phrase
    containing the token at *seed_idx* (typically a ``Case=Acc`` or ``Case=Dat``
    word).
    """
    if not tokens or not (0 <= seed_idx < len(tokens)):
        return seed_idx, seed_idx
    h = walk_np_head_index(tokens, seed_idx)
    cluster = _collect_np_indices(tokens, h)
    return min(cluster), max(cluster)
