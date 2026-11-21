"""Data records produced by the agent: plan, evidence values, candidates, checks and answer.

Every record is a frozen dataclass so a finished answer is an immutable audit trail.
``to_dict`` on ``AgentAnswer`` gives a JSON-safe view for reports and saved predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any


class TaskType(StrEnum):
    """The calculation shape a question asks for, as decided by the planner."""

    PERCENT_CHANGE = "percent_change"
    CHANGE = "change"
    RATIO = "ratio"
    SUM = "sum"
    AVERAGE = "average"
    ROW_MAX = "row_max"
    ROW_MIN = "row_min"
    COMPARISON = "comparison"
    UNSUPPORTED = "unsupported"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"


@dataclass(frozen=True)
class PlanStep:
    """One step of the research plan, e.g. "find revenue for 2017"."""

    action: str
    description: str


@dataclass(frozen=True)
class ResearchPlan:
    """The planner's reading of a question: what to find and how to combine it."""

    question: str
    task: TaskType
    entity_terms: tuple[str, ...]
    periods: tuple[str, ...]
    steps: tuple[PlanStep, ...]
    rationale: str
    # Ratio questions only: the numerator and denominator phrases, when the wording names both.
    part_terms: tuple[str, ...] = ()
    whole_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValueMention:
    """One number found in a segment, with the labels that say what it measures.

    ``literal`` is the value as it will appear in a program; ``label`` is the row label of a
    table cell or the words around a number in a sentence.
    """

    segment_key: str
    literal: str
    value: Decimal
    label: str
    column: str
    periods: tuple[str, ...]
    source: str
    is_percent: bool = False


@dataclass(frozen=True)
class Operand:
    """A value bound to a slot of the plan, e.g. ``new`` or ``whole``."""

    slot: str
    mention: ValueMention
    match_score: float


@dataclass(frozen=True)
class CandidateProgram:
    """A proposed FinQA program with the operands and evidence it relies on."""

    program: str
    operands: tuple[Operand, ...]
    cited_keys: tuple[str, ...]
    score: float
    proposer: str
    rationale: str


@dataclass(frozen=True)
class CheckResult:
    """One verification check. ``hard`` failures reject the candidate."""

    name: str
    passed: bool
    detail: str
    hard: bool = True


@dataclass(frozen=True)
class CalculationStep:
    """One executed program step, for the calculation trace."""

    index: int
    operation: str
    arguments: tuple[str, ...]
    result: float | str


@dataclass(frozen=True)
class VerifiedCandidate:
    candidate: CandidateProgram
    checks: tuple[CheckResult, ...]
    value: float | str | None
    trace: tuple[CalculationStep, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks if check.hard)


@dataclass(frozen=True)
class ToolCall:
    """One logged tool invocation: name, arguments, a short result summary, and duration."""

    round: int
    tool: str
    arguments: dict[str, Any]
    summary: str
    milliseconds: float


@dataclass(frozen=True)
class Citation:
    key: str
    text: str


@dataclass(frozen=True)
class AgentAnswer:
    """The final output: an answer with citations and calculation trace, or an abstention."""

    record_id: str
    question: str
    status: AnswerStatus
    value: float | str | None
    display_answer: str | None
    program: str | None
    citations: tuple[Citation, ...]
    calculation: tuple[CalculationStep, ...]
    checks: tuple[CheckResult, ...]
    plan: ResearchPlan
    tool_calls: tuple[ToolCall, ...]
    rounds: int
    confidence: float
    abstention_reason: str | None = None
    alternatives: tuple[str, ...] = field(default_factory=tuple)
    # The best candidate's value when the agent abstains, kept for risk-coverage analysis.
    withheld_value: float | str | None = None
    milliseconds: float = 0.0

    @property
    def answered(self) -> bool:
        return self.status is AnswerStatus.ANSWERED

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    return value
