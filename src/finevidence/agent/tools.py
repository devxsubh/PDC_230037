"""Controlled tools the agent may call, each logged with its arguments, result and duration.

The agent can only search one record's evidence, read value mentions from segments it has
retrieved, and run the safe FinQA calculator. There is no free-form code execution and no
access to annotation fields: the toolbox is built from the allowlisted inference input.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from finevidence.data.annotations import parse_finqa_program
from finevidence.data.preprocessing import assert_inference_safe, build_structured_input
from finevidence.reasoning.executor import ExecutionResult, execute_program
from finevidence.retrieval.lexical import ScoredSegment, SegmentRetriever
from finevidence.retrieval.pairs import segment_retrieval_text

from .evidence import extract_mentions
from .schema import CalculationStep, ToolCall, ValueMention


class ToolBox:
    """Tools bound to one record. ``calls`` is the ordered tool log for the answer."""

    def __init__(self, record: Mapping[str, Any], retriever: SegmentRetriever) -> None:
        self._source = build_structured_input(record)
        assert_inference_safe(self._source)
        # Retrievers and the executor get the allowlisted fields plus the FinQA ``qa.question``
        # wrapper they expect; no annotation field is passed on.
        self._record = {**self._source, "qa": {"question": self._source["question"]}}
        self._retriever = retriever
        self.calls: list[ToolCall] = []
        self.round = 0
        self.retrieved: dict[str, ScoredSegment] = {}

    @property
    def table(self) -> Sequence[Sequence[object]]:
        return self._source["table"]

    def segment_text(self, key: str) -> str:
        return segment_retrieval_text(self.retrieved[key].segment)

    # Tools ---------------------------------------------------------------------------------

    def search_evidence(self, query: str, k: int) -> list[ScoredSegment]:
        """Top-k segments for a query; adds them to the retrieved evidence pool."""
        started = time.perf_counter()
        ranked = self._retriever.rank(self._record, query)[:k]
        new = [item.key for item in ranked if item.key not in self.retrieved]
        for item in ranked:
            best = self.retrieved.get(item.key)
            if best is None or item.rank < best.rank:
                self.retrieved[item.key] = item
        self._log(
            "search_evidence",
            {"query": query, "k": k, "retriever": self._retriever.name},
            f"top {[item.key for item in ranked]}; new {new}",
            started,
        )
        return ranked

    def extract_values(self, keys: Sequence[str]) -> list[ValueMention]:
        """Labelled numeric values from already retrieved segments."""
        started = time.perf_counter()
        unknown = [key for key in keys if key not in self.retrieved]
        if unknown:
            raise KeyError(f"segments not retrieved yet: {unknown}")
        mentions = extract_mentions(self.retrieved[key].segment for key in keys)
        self._log("extract_values", {"keys": list(keys)}, f"{len(mentions)} values", started)
        return mentions

    def calculate(self, program: str) -> tuple[ExecutionResult, tuple[CalculationStep, ...]]:
        """Run a FinQA program on the safe executor and return a per-step trace."""
        started = time.perf_counter()
        result = execute_program(program, self.table)
        trace = calculation_trace(program, self.table) if result.is_valid else ()
        summary = f"value {result.value}" if result.is_valid else f"error: {result.error}"
        self._log("calculate", {"program": program}, summary, started)
        return result, trace

    def _log(self, tool: str, arguments: dict[str, Any], summary: str, started: float) -> None:
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        self.calls.append(ToolCall(self.round, tool, arguments, summary, elapsed))


def calculation_trace(
    program: str, table: Sequence[Sequence[object]]
) -> tuple[CalculationStep, ...]:
    """Intermediate values, by executing each program prefix on the same safe executor."""
    parsed = parse_finqa_program(program)
    steps = []
    calls = [f"{step.operation}({', '.join(step.arguments)})" for step in parsed.steps]
    for index, step in enumerate(parsed.steps):
        value = execute_program(", ".join(calls[: index + 1]), table).value
        steps.append(CalculationStep(index, step.operation, step.arguments, value))
    return tuple(steps)
