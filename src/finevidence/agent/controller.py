"""The research agent: plan, retrieve, propose, verify, and answer or abstain.

One question runs as a bounded loop. Round 1 searches with the question. Each later round
widens the search and issues sub-queries for the entity and each named period, so evidence
the first search missed can still be found. After every round the candidates are verified;
the agent stops as soon as one passes with enough confidence and no close rival. When the
budget runs out without that, it abstains and says why.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping
from typing import Any

from finevidence.data.preprocessing import build_structured_input
from finevidence.retrieval.lexical import SegmentRetriever

from .config import AgentConfig
from .planner import plan_question
from .proposers import ProgramProposer, ProposalContext
from .schema import (
    AgentAnswer,
    AnswerStatus,
    Citation,
    ResearchPlan,
    TaskType,
    VerifiedCandidate,
)
from .tools import ToolBox
from .verifier import verify_candidate


class FinancialResearchAgent:
    """Answers one FinQA-style question over one report with citations and a trace."""

    def __init__(
        self, retriever: SegmentRetriever, proposer: ProgramProposer, config: AgentConfig
    ) -> None:
        self.retriever = retriever
        self.proposer = proposer
        self.config = config
        self.name = config.name

    def run(self, record: Mapping[str, Any]) -> AgentAnswer:
        started = time.perf_counter()
        source = build_structured_input(record)
        plan = plan_question(source["question"])
        tools = ToolBox(record, self.retriever)
        config = self.config

        verified: dict[str, VerifiedCandidate] = {}
        best: VerifiedCandidate | None = None
        rounds = 0
        k = config.initial_k
        decision: str | None = "the proposer does not support this question shape"
        # A question shape no proposer can calculate is abstained on without spending budget.
        max_rounds = config.max_rounds if self.proposer.supports(plan) else 0
        for round_number in range(1, max_rounds + 1):
            rounds = round_number
            tools.round = round_number
            for query in self._queries(plan, round_number):
                tools.search_evidence(query, k)
            mentions = tools.extract_values(list(tools.retrieved))
            context = ProposalContext(
                record_id=source["id"],
                plan=plan,
                mentions=mentions,
                segment_ranks={key: item.rank for key, item in tools.retrieved.items()},
                segment_texts={key: tools.segment_text(key) for key in tools.retrieved},
                table=tools.table,
            )
            for candidate in self.proposer.propose(context)[: config.max_candidates_verified]:
                if candidate.program in verified:
                    continue
                if not config.verify:
                    result, trace = tools.calculate(candidate.program)
                    verified[candidate.program] = VerifiedCandidate(
                        candidate, (), result.value, trace
                    )
                    continue
                checks, value, trace = verify_candidate(candidate, plan, tools, mentions)
                verified[candidate.program] = VerifiedCandidate(candidate, checks, value, trace)
            best, decision = self._decide(list(verified.values()))
            if decision is None:
                break
            k += config.k_step

        answer_value = best.value if best is not None else None
        reason = None if best is not None and decision is None else decision
        if not config.abstain and best is not None:
            reason = None
        if best is None:
            reason = self._no_candidate_reason(plan, verified)
        status = AnswerStatus.ANSWERED if reason is None else AnswerStatus.ABSTAINED
        alternatives = tuple(
            f"{item.candidate.program} = {item.value}"
            for item in self._ranked(list(verified.values()))[1:4]
        )
        return AgentAnswer(
            record_id=source["id"],
            question=source["question"],
            status=status,
            value=answer_value if status is AnswerStatus.ANSWERED else None,
            display_answer=display_value(plan, answer_value)
            if status is AnswerStatus.ANSWERED
            else None,
            program=best.candidate.program if best else None,
            citations=tuple(
                Citation(key, tools.segment_text(key))
                for key in best.candidate.cited_keys
                if key in tools.retrieved
            )
            if best
            else (),
            calculation=best.trace if best else (),
            checks=best.checks if best else (),
            plan=plan,
            tool_calls=tuple(tools.calls),
            rounds=rounds,
            confidence=self._confidence(best) if best else 0.0,
            abstention_reason=reason,
            alternatives=alternatives,
            withheld_value=answer_value if status is AnswerStatus.ABSTAINED else None,
            milliseconds=round((time.perf_counter() - started) * 1000, 3),
        )

    # Policy --------------------------------------------------------------------------------

    def _queries(self, plan: ResearchPlan, round_number: int) -> list[str]:
        if round_number == 1:
            return [plan.question]
        entity = " ".join(plan.entity_terms)
        queries = [f"{entity} {period}" for period in plan.periods] or [entity]
        if round_number > 2 and plan.task is TaskType.RATIO:
            queries.append(f"total {entity}")
        return [query for query in queries if query.strip()]

    def _support(self, item: VerifiedCandidate) -> float:
        """Unclipped evidence support used for ranking, thresholds and ties.

        A period mismatch discounts support. A conflict does not: a conflicted candidate keeps
        its rank, so the agent reports the conflict instead of quietly answering with a
        weaker, unrelated candidate.
        """
        support = max(0.0, item.candidate.score)
        for check in item.checks:
            if check.name == "periods_match" and not check.passed:
                support *= self.config.period_mismatch_penalty
        return round(support, 6)

    def _confidence(self, item: VerifiedCandidate) -> float:
        """Support on a 0-1 scale, as reported with the answer."""
        return min(1.0, self._support(item))

    def _ranked(self, items: list[VerifiedCandidate]) -> list[VerifiedCandidate]:
        usable = [item for item in items if item.passed and item.value is not None]
        if not self.config.verify:
            usable = [item for item in items if item.value is not None] or items
        return sorted(usable, key=lambda item: -self._support(item))

    def _decide(self, items: list[VerifiedCandidate]):
        """Return ``(best, None)`` to answer now, or ``(best_or_None, reason)`` to keep going."""
        ranked = self._ranked(items)
        if not ranked:
            return None, "no candidate passed verification"
        best = ranked[0]
        if not self.config.verify or not self.config.abstain:
            return best, None
        conflict = next(
            (c for c in best.checks if c.name == "no_conflicting_evidence" and not c.passed), None
        )
        if conflict and self.config.abstain_on_conflict:
            return best, f"conflicting evidence: {conflict.detail}"
        support = self._support(best)
        if support < self.config.min_confidence:
            return best, f"low confidence {support:.2f} < {self.config.min_confidence:.2f}"
        rival = next((item for item in ranked[1:] if item.value != best.value), None)
        if rival and support - self._support(rival) < self.config.ambiguity_margin:
            return best, (
                f"ambiguous: '{best.candidate.program}' = {best.value} and "
                f"'{rival.candidate.program}' = {rival.value} are equally supported"
            )
        return best, None

    def _no_candidate_reason(self, plan: ResearchPlan, verified) -> str:
        if plan.task is TaskType.UNSUPPORTED and not verified:
            return "unsupported question: no supported calculation shape was recognised"
        if not verified:
            return "insufficient evidence: no values matching the plan were found"
        failures = Counter(
            check.name
            for item in verified.values()
            for check in item.checks
            if check.hard and not check.passed
        )
        top = ", ".join(f"{name} ({count})" for name, count in failures.most_common(2))
        return f"no candidate passed verification; failed checks: {top}"


def display_value(plan: ResearchPlan, value: float | str | None) -> str | None:
    """Human-readable answer; FinQA stores percentages as ratios (0.098 -> 9.8%)."""
    if value is None or isinstance(value, str):
        return value
    question = plan.question.lower()
    if plan.task is TaskType.PERCENT_CHANGE or (
        plan.task is TaskType.RATIO and ("percent" in question or "%" in question)
    ):
        return f"{value * 100:.2f}%"
    return f"{value:,.4f}".rstrip("0").rstrip(".")
