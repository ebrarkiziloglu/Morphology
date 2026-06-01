"""Classify German attributive adjectives as strong or weak declension."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

_DET_RELATIONS = frozenset({"det", "det:poss"})

# Determiners that trigger weak adjective endings (after definite article pattern).
_WEAK_DETERMINER_LEMMAS = frozenset(
    {
        "alle",
        "beide",
        "der",
        "die",
        "das",
        "den",
        "dem",
        "des",
        "dieser",
        "jener",
        "jeder",
        "mancher",
        "welcher",
        "solcher",
        "derjenige",
        "derselbe",
        "jeglicher",
    }
)

_WEAK_DETERMINER_PREFIXES = (
    "dies",
    "jen",
    "jeglich",
    "jed",
    "manch",
    "solch",
    "welch",
    "derselb",
    "derjenig",
)


def _determiner_triggers_weak(det: Mapping[str, Any]) -> Optional[bool]:
    """Return True/False for weak/strong; None if this determiner is not recognized."""
    if det.get("upos") != "DET":
        return None

    lemma = (det.get("lemma") or "").lower()
    feats = det.get("feats") or {}

    if lemma in _WEAK_DETERMINER_LEMMAS:
        return True
    if any(lemma.startswith(prefix) for prefix in _WEAK_DETERMINER_PREFIXES):
        return True

    if feats.get("PronType") == "Art" and feats.get("Definite") == "Def":
        return True

    # Indefinite article, negative article, possessives → strong (mixed) endings.
    if feats.get("PronType") == "Art" and feats.get("Definite") == "Ind":
        return False
    if feats.get("Poss") == "Yes":
        return False
    if lemma.startswith("kein"):
        return False

    return None


def adj_inflection_from_context(
    adj_token: Mapping[str, Any],
    tokens: List[Mapping[str, Any]],
) -> Optional[str]:
    """Return ``Strong``, ``Weak``, or ``None`` for an attributive adjective token."""
    if adj_token.get("upos") != "ADJ":
        return None
    feats = adj_token.get("feats") or {}
    if not feats.get("Case"):
        return None

    head_id = str(adj_token.get("head") or "")
    if not head_id or head_id == "0":
        return "Strong"

    determiners = [
        t
        for t in tokens
        if str(t.get("head")) == head_id and t.get("deprel") in _DET_RELATIONS
    ]
    if not determiners:
        return "Strong"

    signals: List[bool] = []
    for det in determiners:
        trigger = _determiner_triggers_weak(det)
        if trigger is not None:
            signals.append(trigger)

    if not signals:
        return None
    return "Weak" if any(signals) else "Strong"
