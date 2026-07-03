"""
Nominal phrase span detection for UD German token lists.

Used by the Accusative/Dative pair builders: a ``Case=Acc`` / ``Case=Dat`` seed
token is expanded to the full NP by (1) walking *up* along ``det`` / ``amod`` /
``flat`` / … edges to the phrase head (including dative ``nmod`` articles on a
noun tagged with another case, e.g. *der Alpen*), then (2) collecting *downward*
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

# Coordinated adjectives (*freundlicher und ruhiger Umgebung*): walk ``conj`` up to the noun.
_WALK_UP_CONJ_CHILD_UPOS = frozenset({"ADJ", "ADV"})

# Only ascend through ``det`` / ``amod`` / … when the parent can head a nominal phrase.
# Stops e.g. ``dem`` (``det`` of ``mangelt`` in a parataxis) from climbing to the verb.
_NOMINAL_WALK_UP_PARENT_UPOS = frozenset(
    {"NOUN", "PROPN", "PRON", "DET", "ADJ", "NUM"}
)

# ``case`` / ``parataxis`` ADP edges (e.g. *im Gegenteil*, *im Laden*).
_PREP_ADP_DEPRELS = frozenset({"case", "parataxis"})

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


def _is_dative_determiner(tok: Dict[str, Any]) -> bool:
    return tok.get("upos") in {"DET", "PRON"} and (
        tok.get("feats") or {}
    ).get("Case") == "Dat"


def _dative_nmod_article_parent_idx(
    tokens: List[Dict[str, Any]], idx: int, id_to_idx: Dict[str, int]
) -> Optional[int]:
    """GSD ``nmod`` dative article on a noun tagged with another case (*der Alpen*).

    Skip when the determiner heads its own NP (*auf der einen*), i.e. it already
    has ``case`` / ``det`` dependents.
    """
    tok = tokens[idx]
    if _deprel_base(tok.get("deprel")) != "nmod":
        return None
    if not _is_dative_determiner(tok):
        return None
    head_id = tok.get("id")
    for child in tokens:
        if child.get("head") != head_id:
            continue
        if _deprel_base(child.get("deprel")) in {"case", "det"}:
            return None
    pidx = _parent_index(tokens, tok, id_to_idx)
    if pidx is None:
        return None
    if tokens[pidx].get("upos") not in {"NOUN", "PROPN"}:
        return None
    return pidx


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
        if drel == "conj" and tok.get("upos") in _WALK_UP_CONJ_CHILD_UPOS:
            parent_upos = tokens[pidx].get("upos", "")
            if parent_upos not in _NOMINAL_WALK_UP_PARENT_UPOS:
                return idx
            idx = pidx
            continue
        article_parent = _dative_nmod_article_parent_idx(tokens, idx, id_to_idx)
        if article_parent is not None:
            return article_parent
        if drel not in _WALK_UP_DEPS:
            return idx
        parent_upos = tokens[pidx].get("upos", "")
        if parent_upos not in _NOMINAL_WALK_UP_PARENT_UPOS:
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
    if drel == "conj":
        parent_upos = tokens[parent_idx].get("upos", "")
        child_upos = child.get("upos", "")
        return (
            parent_upos in _WALK_UP_CONJ_CHILD_UPOS
            and child_upos in _WALK_UP_CONJ_CHILD_UPOS
        )
    if drel in _BLOCKED_DOWN_DEPS:
        return False
    if drel == "nmod":
        if (
            _is_dative_determiner(child)
            and tokens[parent_idx].get("upos") in {"NOUN", "PROPN"}
        ):
            return True
        return _nominal_head_case(tokens, child_idx, id_to_idx) == "Gen"
    return drel in _WALK_UP_DEPS


def _is_misattached_dative_det(
    tokens: List[Dict[str, Any]], idx: int, id_to_idx: Dict[str, int]
) -> bool:
    """True when a dative ``DET`` depends on a verb instead of the local noun."""
    tok = tokens[idx]
    if tok.get("upos") != "DET":
        return False
    if (tok.get("feats") or {}).get("Case") != "Dat":
        return False
    pidx = _parent_index(tokens, tok, id_to_idx)
    if pidx is None:
        return False
    return tokens[pidx].get("upos") not in _NOMINAL_WALK_UP_PARENT_UPOS


_NOMINAL_GAP_UPOS = frozenset({"NOUN", "PROPN", "NUM", "ADJ", "DET"})


def _misattached_dative_det_pp_indices(
    tokens: List[Dict[str, Any]], det_idx: int
) -> Optional[Tuple[int, ...]]:
    """Rejoin *[ADP][DET]…[NOUN]* when GSD attaches ``dem`` to the matrix verb."""
    id_to_idx = _id_to_index(tokens)
    if not _is_misattached_dative_det(tokens, det_idx, id_to_idx):
        return None

    adp_idx: Optional[int] = None
    for j in range(det_idx - 1, max(det_idx - 3, -1), -1):
        t = tokens[j]
        if t.get("upos") == "ADP" and _deprel_base(t.get("deprel")) in _PREP_ADP_DEPRELS:
            adp_idx = j
            break
    if adp_idx is None:
        return None

    adp_id = tokens[adp_idx].get("id")
    noun_idx: Optional[int] = None
    governed_idx = id_to_idx.get(_token_id_key(tokens[adp_idx].get("head")))
    if governed_idx is not None:
        gtok = tokens[governed_idx]
        if gtok.get("upos") in {"NOUN", "PROPN", "PRON"}:
            noun_idx = governed_idx

    if noun_idx is None:
        for j in range(det_idx + 1, min(len(tokens), det_idx + 4)):
            t = tokens[j]
            if t.get("upos") not in {"NOUN", "PROPN"}:
                continue
            if (t.get("feats") or {}).get("Case") != "Dat":
                continue
            if _token_id_key(t.get("head")) == _token_id_key(adp_id):
                noun_idx = j
                break
            if noun_idx is None:
                noun_idx = j
    if noun_idx is None:
        return None

    h = walk_np_head_index(tokens, noun_idx)
    cluster = _collect_np_indices(tokens, h)
    for j in range(det_idx + 1, noun_idx):
        if tokens[j].get("upos") in _NOMINAL_GAP_UPOS:
            cluster.add(j)
    cluster.update({adp_idx, det_idx})
    return tuple(sorted(cluster))


def _prepend_misattached_pp_prefix(
    tokens: List[Dict[str, Any]], span_indices: Tuple[int, ...]
) -> Tuple[int, ...]:
    """If *span_indices* sits inside a misattached *[ADP][DET]…* PP, include the prefix."""
    if not span_indices:
        return span_indices
    span_set = set(span_indices)
    lo, hi = span_indices[0], span_indices[-1]
    id_to_idx = _id_to_index(tokens)
    for det_idx in range(max(0, lo - 12), hi + 1):
        if not _is_misattached_dative_det(tokens, det_idx, id_to_idx):
            continue
        pp = _misattached_dative_det_pp_indices(tokens, det_idx)
        if pp is None:
            continue
        if span_set & set(pp):
            return tuple(sorted(span_set | set(pp)))
    return span_indices


def _collect_np_indices(tokens: List[Dict[str, Any]], head_idx: int) -> Set[int]:
    id_to_idx = _id_to_index(tokens)
    collected: Set[int] = {head_idx}
    # ``appos_depth``: 0 at phrase head; 1 under a direct ``appos`` dependent.
    # Do not recurse into nested ``appos`` (e.g. *Film Die Commitments (1991, …)*).
    stack: List[Tuple[int, int]] = [(head_idx, 0)]
    while stack:
        p, appos_depth = stack.pop()
        for j, t in enumerate(tokens):
            if j in collected:
                continue
            if not _down_edge_ok(tokens, j, p, id_to_idx):
                continue
            drel = _deprel_base(t.get("deprel"))
            if drel == "appos" and appos_depth >= 1:
                continue
            collected.add(j)
            child_depth = appos_depth + 1 if drel == "appos" else appos_depth
            stack.append((j, child_depth))
    return collected


def find_nominal_group_indices(
    tokens: List[Dict[str, Any]], seed_idx: int
) -> Tuple[int, ...]:
    """
    Sorted token indices in the nominal phrase containing *seed_idx*.

    Only indices that are linked to the phrase head via internal nominal deps
    are included (not every token between min and max in linear order).
    """
    if not tokens or not (0 <= seed_idx < len(tokens)):
        return (seed_idx,)
    misattached_pp = _misattached_dative_det_pp_indices(tokens, seed_idx)
    if misattached_pp is not None:
        return misattached_pp
    h = walk_np_head_index(tokens, seed_idx)
    cluster = _collect_np_indices(tokens, h)
    return _prepend_misattached_pp_prefix(tokens, tuple(sorted(cluster)))


def find_nominal_group_span(tokens: List[Dict[str, Any]], seed_idx: int) -> Tuple[int, int]:
    """
    Return inclusive ``(start, end)`` bounds for :func:`find_nominal_group_indices`.

    When the phrase is a single contiguous chunk, ``start``/``end`` match the
    usual slice ``tokens[start : end + 1]``. Callers that convert or display the
    span should iterate :func:`find_nominal_group_indices`, not that range.
    """
    inds = find_nominal_group_indices(tokens, seed_idx)
    return inds[0], inds[-1]
