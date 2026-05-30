from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SpanEliminationReport:
    """
    Aggregates where candidate pairs/spans disappear before (or instead of)
    appearing in JSON. Used for logging + debugging elimination points.
    """

    direction: str = ""
    splits_requested_but_missing_dataset: List[str] = field(default_factory=list)
    sentences_empty_tokens: int = 0
    group_dropped_span_unconverted: int = 0
    accusative_seed_tokens_seen: int = 0
    pairs_yielded_from_iterator: int = 0
    group_span_too_short_for_json_row: int = 0  # legacy: should stay 0 after allowing single-token spans
    group_dropped_normalized_equals_gold: int = 0
    group_records_written: int = 0
    group_dropped_duplicate_span_key: int = 0

    # Reasons for span-level dropout (first failing token in NP span) + anomaly tags.
    # This is a defaultdict at runtime; annotated as Dict for simplicity.
    dropout_by_reason: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Optional capped sample lines per category for manual inspection
    sample_sentence_dropout: List[str] = field(default_factory=list)
    sample_group_below_min_span: List[str] = field(default_factory=list)
    sample_group_np_substituted_gold: List[str] = field(default_factory=list)
    sample_group_dedup: List[str] = field(default_factory=list)
    SAMPLE_CAP: int = 40

    def _push_sample(self, bucket: List[str], line: str) -> None:
        if len(bucket) < self.SAMPLE_CAP:
            bucket.append(line)

    def summary_lines(self) -> List[str]:
        dr = sorted(self.dropout_by_reason.items(), key=lambda kv: (-kv[1], kv[0]))
        dr_txt = "\n".join(f"{k}={v}" for k, v in dr) if dr else "(none)"
        dir_txt = f" ({self.direction})" if self.direction else ""
        return [
            f"--- elimination summary{dir_txt} ---",
            f"splits_requested_but_missing_dataset: {self.splits_requested_but_missing_dataset or '(none)'}",
            f"sentences_empty_tokens: {self.sentences_empty_tokens}",
            f"group_dropped_span_unconverted: {self.group_dropped_span_unconverted}",
            f"span_dropout_reason_counts: {dr_txt}",
            f"accusative_seed_tokens_seen: {self.accusative_seed_tokens_seen}",
            f"pairs_yielded_from_iterator: {self.pairs_yielded_from_iterator}",
            "--- group-level ---",
            f"group_span_too_short_for_json_row (legacy): {self.group_span_too_short_for_json_row}",
            f"group_dropped_normalized_equals_gold: {self.group_dropped_normalized_equals_gold}",
            f"group_dropped_duplicate_span_key: {self.group_dropped_duplicate_span_key}",
            f"group_records_written: {self.group_records_written}",
        ]
