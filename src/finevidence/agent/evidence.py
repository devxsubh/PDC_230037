"""Turn evidence segments into labelled value mentions the proposer can bind to plan slots.

A table cell's mention takes its row label and column header; a year in either one becomes
the mention's period, so both year-per-column and year-per-row tables work. A sentence's
mentions take the nearby words and the nearest year. Values are parsed with the existing
``parse_numeric_candidate`` (financial-value normalization), never with ``eval``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from decimal import Decimal

from finevidence.reasoning.arithmetic import parse_numeric_candidate
from finevidence.retrieval.lexical import tokenize
from finevidence.retrieval.segments import TEXT_NUMBER_SPAN, EvidenceSegment

from .planner import YEAR
from .schema import ValueMention

_TEXT_WINDOW = 8  # words on each side of a number that describe it
# FinQA cells often repeat a value in parentheses: "$ -4063 ( 4063 )", "32.26% ( 32.26 % )".
_ECHOED_VALUE = re.compile(r"(?<=\d|%)\s*\(\s*[\d.,]+\s*%?\s*\)\s*$")


def clean_cell(raw: str) -> str:
    """Drop a trailing parenthesised echo of the cell's own value; other text is unchanged."""
    return _ECHOED_VALUE.sub("", raw).strip()


def literal_for(value: Decimal) -> str:
    """Render a parsed value as a FinQA program literal, e.g. ``-1970`` or ``14.7``."""
    text = format(value.normalize(), "f")
    return "0" if text in {"-0", ""} else text


def table_mentions(segment: EvidenceSegment) -> list[ValueMention]:
    if segment.source != "table" or len(segment.cells) < 2:
        return []
    label = segment.cells[0]
    # FinQA's first row is usually a header, but in key-value tables it holds data
    # (["balance at may 31 2014", "$ -7199"]); its cells then have no column header.
    columns = segment.header[1:] if segment.key != "table_0" else ("",) * (len(segment.cells) - 1)
    mentions = []
    for column, raw in zip(columns, segment.cells[1:], strict=False):
        candidate = parse_numeric_candidate(
            clean_cell(raw), source_id=segment.key, context=segment.text
        )
        if candidate.value is None:
            continue
        periods = tuple(dict.fromkeys(YEAR.findall(f"{column} {label}")))
        mentions.append(
            ValueMention(
                segment_key=segment.key,
                literal=literal_for(candidate.value),
                value=candidate.value,
                label=label,
                column=column,
                periods=periods,
                source="table",
                is_percent=candidate.is_percent,
            )
        )
    return mentions


def text_mentions(segment: EvidenceSegment) -> list[ValueMention]:
    if segment.source != "text":
        return []
    text = segment.text
    years = [(match.start(), match.group()) for match in YEAR.finditer(text)]
    mentions = []
    for match in TEXT_NUMBER_SPAN.finditer(text):
        raw = match.group()
        if YEAR.fullmatch(raw.rstrip("%")):
            continue
        candidate = parse_numeric_candidate(raw, source_id=segment.key, context=text)
        if candidate.value is None:
            continue
        before = text[: match.start()].split()[-_TEXT_WINDOW:]
        after = text[match.end() :].split()[:_TEXT_WINDOW]
        nearest = sorted(years, key=lambda year: abs(year[0] - match.start()))
        mentions.append(
            ValueMention(
                segment_key=segment.key,
                literal=literal_for(candidate.value),
                value=candidate.value,
                label=" ".join([*before, *after]),
                column="",
                periods=(nearest[0][1],) if nearest else (),
                source="text",
                is_percent=raw.endswith("%") or text[match.end() :].lstrip().startswith("%"),
            )
        )
    return mentions


def extract_mentions(segments: Iterable[EvidenceSegment]) -> list[ValueMention]:
    mentions: list[ValueMention] = []
    for segment in segments:
        mentions.extend(table_mentions(segment) if segment.source == "table" else [])
        mentions.extend(text_mentions(segment))
    return mentions


def label_match(entity_terms: Sequence[str], mention: ValueMention) -> float:
    """Share of entity terms found in the mention's label, row label weighted over column.

    A score in [0, 1]; a table row whose own label covers the terms scores highest.
    """
    if not entity_terms:
        return 0.0
    label_tokens = set(tokenize(mention.label))
    column_tokens = set(tokenize(mention.column)) - label_tokens
    hits = sum(
        1.0 if term in label_tokens else 0.5 if term in column_tokens else 0.0
        for term in entity_terms
    )
    coverage = hits / len(entity_terms)
    # Penalise long labels that match by accident (a whole sentence vs a row label).
    precision = hits / max(len(label_tokens), 1)
    return round(0.75 * coverage + 0.25 * min(precision, 1.0), 6)


def matches_period(mention: ValueMention, period: str) -> bool:
    return period in mention.periods


def conflicting_mentions(mentions: Sequence[ValueMention]) -> list[tuple[ValueMention, ...]]:
    """Groups of mentions with the same label and period but different values.

    Two different numbers for "revenue, 2017" in the retrieved evidence are a conflict the
    verifier must surface instead of silently picking one.
    """
    groups: dict[tuple[str, str, tuple[str, ...]], list[ValueMention]] = {}
    for mention in mentions:
        if mention.source != "table" or not mention.periods:
            continue
        key = (mention.label.strip().lower(), mention.column.strip().lower(), mention.periods)
        groups.setdefault(key, []).append(mention)
    return [
        tuple(group)
        for group in groups.values()
        if len({abs(mention.value) for mention in group}) > 1
        and len({mention.segment_key for mention in group}) > 1
    ]
