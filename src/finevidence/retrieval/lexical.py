"""Within-record lexical evidence retrieval (BM25) and a replay of saved retriever rankings.

Both retrievers rank the segments of one FinQA record for a query and read only allowlisted
fields. ``Bm25Retriever`` needs no trained model, so the agent runs offline; it fits its term
statistics on the record's own segments, never on other records or annotations.
``RankingFileRetriever`` replays a Notebook 10 cross-encoder ranking file, so the agent can
reuse the trained retriever's output without a GPU.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .pairs import segment_retrieval_text
from .segments import EvidenceSegment, build_evidence_segments

_TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
STOPWORDS = frozenset(
    """a an and are as at be been by during for from had has have in into is it its of on or
    over than that the their them then there these this those to was were what when which
    while who will with within how much many did does do""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word and number tokens without stopwords; commas inside numbers are dropped."""
    return [
        token for token in _TOKEN.findall(text.lower().replace(",", "")) if token not in STOPWORDS
    ]


@dataclass(frozen=True)
class ScoredSegment:
    """One segment with its retrieval score and 1-based rank for a query."""

    segment: EvidenceSegment
    score: float
    rank: int

    @property
    def key(self) -> str:
        return self.segment.key


class SegmentRetriever(Protocol):
    """Ranks one record's evidence segments for a query."""

    name: str

    def rank(self, record: Mapping[str, Any], query: str) -> list[ScoredSegment]: ...


def record_segments(record: Mapping[str, Any]) -> list[EvidenceSegment]:
    """All table rows (header included) and sentences of a record, in segment order."""
    return build_evidence_segments(record, include_header_row=True)


class Bm25Retriever:
    """Okapi BM25 over one record's segments, rendered with column headers."""

    name = "bm25"

    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def rank(self, record: Mapping[str, Any], query: str) -> list[ScoredSegment]:
        segments = record_segments(record)
        documents = [tokenize(segment_retrieval_text(segment)) for segment in segments]
        query_terms = set(tokenize(query))
        count = len(documents)
        average_length = sum(len(document) for document in documents) / max(count, 1)
        document_frequency = Counter(term for document in documents for term in set(document))
        scores = []
        for document in documents:
            frequencies = Counter(document)
            score = 0.0
            for term in query_terms & frequencies.keys():
                idf = math.log(
                    1 + (count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
                )
                tf = frequencies[term]
                norm = tf + self.k1 * (1 - self.b + self.b * len(document) / (average_length or 1))
                score += idf * tf * (self.k1 + 1) / norm
            scores.append(score)
        return _ranked(segments, scores)


class RankingFileRetriever:
    """Replays saved per-record rankings, e.g. Notebook 10's cross-encoder output.

    The saved ranking is query-independent: sub-queries return the same order. Segments
    missing from a ranking are appended in segment order with score 0.
    """

    name = "ranking-file"

    def __init__(self, rankings: Mapping[str, Sequence[str]]) -> None:
        self.rankings = rankings

    def rank(self, record: Mapping[str, Any], query: str) -> list[ScoredSegment]:
        segments = record_segments(record)
        ranked = list(self.rankings[str(record["id"])])
        position = {key: index for index, key in enumerate(ranked)}
        scores = [
            float(len(ranked) - position[segment.key]) if segment.key in position else 0.0
            for segment in segments
        ]
        return _ranked(segments, scores)


def _ranked(segments: Sequence[EvidenceSegment], scores: Sequence[float]) -> list[ScoredSegment]:
    order = sorted(range(len(segments)), key=lambda index: -scores[index])
    return [
        ScoredSegment(segments[index], scores[index], rank)
        for rank, index in enumerate(order, start=1)
    ]
