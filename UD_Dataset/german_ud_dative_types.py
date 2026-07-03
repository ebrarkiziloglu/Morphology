from load_ud_dataset import load_ud_dataset, UD_GERMAN_SPLITS

UD_GERMAN_GSD_SPLITS = {
    'train': 'de_gsd-ud-train.conllu',
    'dev': 'de_gsd-ud-dev.conllu',
    'test': 'de_gsd-ud-test.conllu',
}
ud_path = "../data/UD_German-GSD"

ud = load_ud_dataset(ud_path, splits_filemap=UD_GERMAN_SPLITS[0])

from conllu import parse_incr
import pandas as pd

from ud_corruption_utils import (
    case_marker_lemma_for_head_id,
    classify_dative_type,
    is_target_dative_preposition,
    normalize_case_prep_lemma,
)

########################################################################
# CONFIG
########################################################################

DAT_POS = {"NOUN", "PROPN", "PRON"}

########################################################################
# HELPERS
########################################################################

def has_dative(feats):
    if feats is None:
        return False
    return feats.get("Case") == "Dat"


def get_token_by_id(sent, idx):
    for tok in sent:
        if tok["id"] == idx:
            return tok
    return None


def get_children(sent, head_id):
    return [tok for tok in sent if tok["head"] == head_id]


def collect_span(sent, head_id):
    """
    Collect a simple NP span around the dative head (no preposition).
    Case markers / prepositions are stored via get_case_marker() and get_preposition().
    """

    allowed = {
        "det",
        "amod",
        "nummod",
        "compound",
        "fixed",
        "flat",
    }

    span_tokens = []

    for tok in sent:
        if tok["id"] == head_id:
            span_tokens.append(tok)

        elif tok["head"] == head_id and tok["deprel"] in allowed:
            span_tokens.append(tok)

    span_tokens = sorted(span_tokens, key=lambda x: x["id"])

    return span_tokens


def get_case_marker(sent, head_id):
    """UD ``case`` dependent lemma (preposition or directional adverb, etc.)."""
    return case_marker_lemma_for_head_id(sent, head_id)


def get_preposition(sent, head_id):
    """Target preposition (aus/bei/…/in); None for nordwestlich, dank, …"""
    case_lemma = get_case_marker(sent, head_id)
    if not is_target_dative_preposition(case_lemma):
        return None
    return normalize_case_prep_lemma(case_lemma)


def classify_dative(token, case_lemma, sent=None, tok_idx=None):
    return classify_dative_type(
        token,
        case_lemma,
        tokens=sent,
        head_idx=tok_idx,
        span_start=tok_idx,
        span_end=tok_idx,
    )



########################################################################
# MAIN
########################################################################

splits = {
    'train': '../data/UD_German-GSD/de_gsd-ud-train.conllu',
    'dev': '../data/UD_German-GSD/de_gsd-ud-dev.conllu',
    'test': '../data/UD_German-GSD/de_gsd-ud-test.conllu',
}

all_dfs = []

for split in splits:
    rows = []

    with open(splits[split], "r", encoding="utf-8") as f:

        for sent in parse_incr(f):

            sent_id = sent.metadata.get("sent_id", "")
            text = sent.metadata.get("text", "")

            for tok in sent:

                if tok["upos"] not in DAT_POS:
                    continue

                if not has_dative(tok["feats"]):
                    continue

                span = collect_span(sent, tok["id"])

                span_text = " ".join(t["form"] for t in span)

                case_marker = get_case_marker(sent, tok["id"])
                prep = get_preposition(sent, tok["id"])

                gov = get_token_by_id(sent, tok["head"])

                if gov is None:
                    gov_form = "ROOT"
                    gov_upos = "ROOT"
                else:
                    gov_form = gov["form"]
                    gov_upos = gov["upos"]

                tok_idx = next(i for i, t in enumerate(sent) if t["id"] == tok["id"])
                subtype = classify_dative(tok, case_marker, sent=sent, tok_idx=tok_idx)

                rows.append({
                    "sent_id": sent_id,
                    "sentence": text,

                    "span": span_text,
                    "head_form": tok["form"],
                    "head_lemma": tok["lemma"],
                    "head_upos": tok["upos"],

                    "case_marker": case_marker,
                    "preposition": prep,
                    "dative_type": subtype,

                    "relation": tok["deprel"],

                    "governor": gov_form,
                    "governor_upos": gov_upos,
                    "split": split,  # add split information for provenance
                })

    df = pd.DataFrame(rows)
    all_dfs.append(df)

# Combine into unified DataFrame
unified_df = pd.concat(all_dfs, ignore_index=True)

print(unified_df.head())

unified_df.to_csv("dative_spans/german_datives_all.csv", index=False)

print(f"Dative types: {set(unified_df.dative_type)}")