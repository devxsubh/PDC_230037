"""Program proposers: turn a plan plus evidence into candidate FinQA programs.

``TemplateProposer`` is the offline default: it binds the plan's slots (old/new, part/whole,
one value per period) to value mentions and writes the program for the plan's task.
``CandidateFileProposer`` replays saved generator beams (Notebook 14's T5 candidates), so the
agent can verify and select among the existing model's outputs without a GPU.

Both return candidates sorted best first; neither executes anything. The controller sends
every candidate through the verifier.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from finevidence.data.annotations import parse_finqa_program

from .evidence import label_match
from .planner import YEAR
from .schema import CandidateProgram, Operand, ResearchPlan, TaskType, ValueMention

_CHANGE_TASKS = {TaskType.PERCENT_CHANGE, TaskType.CHANGE, TaskType.COMPARISON}
_TOTAL = re.compile(r"\btotal\b", re.I)
_UNSAFE_LABEL = re.compile(r"[(),]")


@dataclass(frozen=True)
class ProposalContext:
    """Everything a proposer may read: the plan, evidence and the record's raw inputs."""

    record_id: str
    plan: ResearchPlan
    mentions: Sequence[ValueMention]
    segment_ranks: Mapping[str, int]
    segment_texts: Mapping[str, str]
    table: Sequence[Sequence[object]]


class ProgramProposer(Protocol):
    name: str

    def supports(self, plan: ResearchPlan) -> bool: ...

    def propose(self, context: ProposalContext) -> list[CandidateProgram]: ...


class TemplateProposer:
    """Plan-driven slot filling over labelled value mentions."""

    name = "template"

    def __init__(self, *, max_groups: int = 6, max_candidates: int = 12) -> None:
        self.max_groups = max_groups
        self.max_candidates = max_candidates

    def supports(self, plan: ResearchPlan) -> bool:
        return plan.task is not TaskType.UNSUPPORTED

    def propose(self, context: ProposalContext) -> list[CandidateProgram]:
        task = context.plan.task
        if task is TaskType.UNSUPPORTED or not context.mentions:
            return []
        scored = [(mention, self._score(mention, context)) for mention in context.mentions]
        if task in _CHANGE_TASKS:
            candidates = self._two_period(context, scored)
        elif task is TaskType.RATIO:
            candidates = self._ratio(context, scored)
        elif task in {TaskType.SUM, TaskType.AVERAGE}:
            candidates = self._aggregate(context, scored)
        else:
            candidates = self._row_extreme(context, scored)
        unique: dict[str, CandidateProgram] = {}
        for candidate in sorted(candidates, key=lambda item: -item.score):
            unique.setdefault(candidate.program, candidate)
        return list(unique.values())[: self.max_candidates]

    # Scoring -------------------------------------------------------------------------------

    def _score(self, mention: ValueMention, context: ProposalContext) -> float:
        rank = context.segment_ranks.get(mention.segment_key, 99)
        return label_match(context.plan.entity_terms, mention) + 0.3 / rank

    def _groups(
        self, scored: Sequence[tuple[ValueMention, float]]
    ) -> list[list[tuple[ValueMention, float]]]:
        """Mentions grouped by segment (one table row or sentence), best group first."""
        groups: dict[str, list[tuple[ValueMention, float]]] = {}
        for mention, score in scored:
            groups.setdefault(mention.segment_key, []).append((mention, score))
        ordered = sorted(groups.values(), key=lambda group: -max(score for _, score in group))
        return ordered[: self.max_groups]

    # Task templates ------------------------------------------------------------------------

    def _two_period(self, context, scored) -> list[CandidateProgram]:
        plan = context.plan
        question_order = list(dict.fromkeys(YEAR.findall(plan.question)))
        candidates = []
        pair_sources = [(_period_pairs(group, plan.periods), 0.0) for group in self._groups(scored)]
        # Values for the two periods may sit in different rows ("2015 net revenue" and
        # "2016 net revenue"), so also pair the best match for each period across segments.
        pair_sources.append((_cross_segment_pairs(scored, plan.periods), -0.1))
        for pairs, adjustment in pair_sources:
            for (old, old_score), (new, new_score) in pairs:
                if plan.task is TaskType.COMPARISON:
                    first, second = (new, old)
                    if question_order and old.periods and old.periods[0] == question_order[0]:
                        first, second = (old, new)
                    program = f"greater({first.literal}, {second.literal})"
                elif plan.task is TaskType.CHANGE:
                    program = f"subtract({new.literal}, {old.literal})"
                else:
                    program = f"subtract({new.literal}, {old.literal}), divide(#0, {old.literal})"
                period_bonus = 0.2 if plan.periods else 0.0
                candidates.append(
                    _candidate(
                        program,
                        [("old", old, old_score), ("new", new, new_score)],
                        (old_score + new_score) / 2 + period_bonus + adjustment,
                        self.name,
                        f"{plan.task.value} of '{new.label}' {_period(old)} -> {_period(new)}",
                    )
                )
        return candidates

    def _ratio(self, context, scored) -> list[CandidateProgram]:
        plan = context.plan
        period = plan.periods[-1] if plan.periods else None
        pool = [
            (mention, score)
            for mention, score in scored
            if period is None or period in mention.periods or not mention.periods
        ]
        if plan.part_terms:
            rank_bonus = {id(m): s - label_match(plan.entity_terms, m) for m, s in pool}
            part_pool = [(m, label_match(plan.part_terms, m) + rank_bonus[id(m)]) for m, _ in pool]
            whole_pool = [
                (m, label_match(plan.whole_terms, m) + rank_bonus[id(m)]) for m, _ in pool
            ]
        else:
            part_pool = whole_pool = pool
        parts = sorted(part_pool, key=lambda item: -item[1])[: self.max_groups]
        candidates = []
        for part, part_score in parts:
            for whole, whole_score in whole_pool:
                if whole is part or whole.value == 0:
                    continue
                same_column = whole.column == part.column and whole.source == part.source
                is_total = bool(_TOTAL.search(whole.label)) and not _TOTAL.search(part.label)
                if not (same_column or whole.segment_key == part.segment_key):
                    continue
                structure = 0.35 if is_total and same_column else 0.0
                if whole.segment_key == part.segment_key and not same_column:
                    structure -= 0.15
                weight = 1.0 if plan.whole_terms else 0.5
                score = (part_score + weight * whole_score + structure) / (1 + weight)
                candidates.append(
                    _candidate(
                        f"divide({part.literal}, {whole.literal})",
                        [("part", part, part_score), ("whole", whole, whole_score)],
                        score,
                        self.name,
                        f"ratio of '{part.label}' to '{whole.label}' {_period(part)}",
                    )
                )
        return candidates

    def _aggregate(self, context, scored) -> list[CandidateProgram]:
        plan = context.plan
        candidates = []
        groups = [(group, 0.0) for group in self._groups(scored)]
        if len(plan.periods) >= 2:
            # Year-per-row tables put each period's value in a different segment.
            groups.append((_best_per_period(scored, plan.periods), -0.1))
        for group, adjustment in groups:
            values = _values_for_periods(group, plan.periods)
            if len(values) < 2:
                continue
            program = _add_chain([mention.literal for mention, _ in values])
            if plan.task is TaskType.AVERAGE:
                program += f", divide(#{len(values) - 2}, const_{len(values)})"
            score = sum(score for _, score in values) / len(values)
            candidates.append(
                _candidate(
                    program,
                    [(f"value_{index}", mention, s) for index, (mention, s) in enumerate(values)],
                    score + (0.2 if plan.periods else 0.0) + adjustment,
                    self.name,
                    f"{plan.task.value} of '{values[0][0].label}' over "
                    f"{', '.join(_period(mention) for mention, _ in values)}",
                )
            )
            if plan.task is TaskType.AVERAGE and not plan.periods and _is_period_series(group):
                candidates.extend(self._row_extreme(context, group, operation="table_average"))
        return candidates

    def _row_extreme(self, context, scored, operation: str | None = None):
        operation = operation or (
            "table_max" if context.plan.task is TaskType.ROW_MAX else "table_min"
        )
        candidates = []
        rows = {str(row[0]) for row in context.table if row}
        for group in self._groups(scored):
            mention, score = max(group, key=lambda item: item[1])
            if mention.source != "table" or _UNSAFE_LABEL.search(mention.label):
                continue
            if mention.label not in rows:
                continue
            operands = [(f"cell_{index}", m, s) for index, (m, s) in enumerate(group)]
            candidates.append(
                _candidate(
                    f"{operation}({mention.label}, none)",
                    operands,
                    score - (0.05 if operation == "table_average" else 0.0),
                    self.name,
                    f"{operation} over table row '{mention.label}'",
                )
            )
        return candidates


class CandidateFileProposer:
    """Replays saved generator beams: ``{record_id: [program, ...]}`` in beam order."""

    name = "generator-beams"

    def __init__(self, beams: Mapping[str, Sequence[str]]) -> None:
        self.beams = beams

    def supports(self, plan: ResearchPlan) -> bool:
        return True  # the generator writes programs for any question shape

    def propose(self, context: ProposalContext) -> list[CandidateProgram]:
        programs = self.beams.get(context.record_id, ())
        candidates = []
        for index, program in enumerate(programs):
            literals = _program_literals(program)
            cited = tuple(
                key
                for key, text in context.segment_texts.items()
                if any(_contains_number(text, literal) for literal in literals)
                and context.segment_ranks.get(key, 99) <= 10
            )
            candidates.append(
                CandidateProgram(
                    program=program,
                    operands=(),
                    cited_keys=cited,
                    score=1.0 - 0.1 * index,
                    proposer=self.name,
                    rationale=f"generator beam {index + 1} of {len(programs)}",
                )
            )
        return candidates


# Helpers ---------------------------------------------------------------------------------


def _candidate(program, operands, score, proposer, rationale) -> CandidateProgram:
    bound = tuple(Operand(slot, mention, round(s, 6)) for slot, mention, s in operands)
    cited = tuple(dict.fromkeys(operand.mention.segment_key for operand in bound))
    return CandidateProgram(program, bound, cited, round(score, 6), proposer, rationale)


def _period(mention: ValueMention) -> str:
    return mention.periods[0] if mention.periods else (mention.column or "unlabelled period")


def _period_pairs(group, periods: Sequence[str]):
    """(old, new) mention pairs from one row: named periods first, else adjacent columns."""
    if len(periods) >= 2:
        old = [item for item in group if periods[0] in item[0].periods]
        new = [item for item in group if periods[-1] in item[0].periods]
        return [(o, n) for o, n in itertools.product(old[:1], new[:1])]
    dated = sorted((item for item in group if item[0].periods), key=lambda item: item[0].periods[0])
    if len(periods) == 1:
        named = [item for item in dated if item[0].periods[0] == periods[0]]
        earlier = [item for item in dated if item[0].periods[0] < periods[0]]
        if named and earlier:
            return [(earlier[-1], named[0])]
        return []
    if len(dated) >= 2 and dated[-1][0].periods[0] != dated[-2][0].periods[0]:
        return [(dated[-2], dated[-1])]
    if len(group) >= 2 and group[0][0].source == "table":
        # Undated columns: FinQA tables list the most recent period first.
        return [(group[1], group[0])]
    return []


def _cross_segment_pairs(scored, periods: Sequence[str]):
    """Best-matching old and new value from any segments, for the two named periods."""
    if len(periods) >= 2:
        old_period, new_period = periods[0], periods[-1]
    elif len(periods) == 1:
        new_period = periods[0]
        old_period = str(int(new_period) - 1)
    else:
        return []

    def best(period: str):
        matches = [item for item in scored if period in item[0].periods]
        return max(matches, key=lambda item: item[1]) if matches else None

    old, new = best(old_period), best(new_period)
    if old is None or new is None or old[0].segment_key == new[0].segment_key:
        return []
    return [(old, new)]


def _best_per_period(scored, periods: Sequence[str]):
    """The best-matching mention for each period, from any segment."""
    chosen = []
    for period in periods:
        matches = [item for item in scored if period in item[0].periods]
        if matches:
            chosen.append(max(matches, key=lambda item: item[1]))
    return chosen


def _values_for_periods(group, periods: Sequence[str]):
    if len(periods) >= 2:
        chosen = []
        for period in periods:
            matches = [item for item in group if period in item[0].periods]
            if not matches:
                return []
            chosen.append(matches[0])
        return chosen
    return list(group) if _is_period_series(group) else []


def _is_period_series(group) -> bool:
    """A table row whose values are one measure over distinct periods (like-for-like).

    Rows such as "payments volume | total volume | transactions | cards" mix units and must
    not be summed or averaged.
    """
    if len(group) < 2 or group[0][0].source != "table":
        return False
    periods = [item[0].periods[0] if item[0].periods else None for item in group]
    return None not in periods and len(set(periods)) == len(periods)


def _add_chain(literals: Sequence[str]) -> str:
    steps = [f"add({literals[0]}, {literals[1]})"]
    steps += [f"add(#{index}, {literal})" for index, literal in enumerate(literals[2:])]
    return ", ".join(steps)


def _program_literals(program: str) -> list[str]:
    parsed = parse_finqa_program(program)
    if not parsed.is_valid:
        return []
    return [
        argument
        for step in parsed.steps
        for argument, kind in zip(step.arguments, step.argument_kinds, strict=False)
        if kind == "numeric_literal"
    ]


def _contains_number(text: str, literal: str) -> bool:
    target = abs(float(literal.rstrip("%")))
    for match in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?|\.\d+", text):
        if abs(float(match.replace(",", "")) - target) < 1e-9:
            return True
    return False
