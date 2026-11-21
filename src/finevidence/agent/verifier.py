"""Answer verification: every candidate is checked before the agent may answer with it.

Hard checks reject a candidate outright: the program must execute, every literal must occur
in the cited evidence, every cited segment must have been retrieved, and the result must be
plausible for the task. Soft checks lower confidence: operands should come from the periods
the question names, and the evidence should not contain conflicting values for them.
"""

from __future__ import annotations

from collections.abc import Sequence

from finevidence.evaluation.grounding import operand_grounding

from .evidence import conflicting_mentions
from .schema import CandidateProgram, CheckResult, ResearchPlan, TaskType, ValueMention
from .tools import ToolBox

# A percent change beyond +/-1000% is far more often a wrong operand than a real movement.
MAX_ABS_PERCENT_CHANGE = 10.0
_SHARE_WORDS = ("portion", "proportion", "percentage of", "percent of", "share of", "% of")


def verify_candidate(
    candidate: CandidateProgram,
    plan: ResearchPlan,
    tools: ToolBox,
    mentions: Sequence[ValueMention],
):
    """Execute and check one candidate; returns ``(checks, value, trace)``."""
    result, trace = tools.calculate(candidate.program)
    checks = [
        CheckResult(
            "executes",
            result.is_valid,
            "program executed" if result.is_valid else f"execution failed: {result.error}",
        )
    ]
    checks.append(_retrieved_check(candidate, tools))
    checks.append(_grounding_check(candidate, tools))
    if result.is_valid:
        checks.append(_plausibility_check(plan, candidate, result.value))
    checks.append(_period_check(plan, candidate))
    checks.append(_conflict_check(candidate, mentions))
    return tuple(checks), result.value, trace


def _retrieved_check(candidate: CandidateProgram, tools: ToolBox) -> CheckResult:
    missing = [key for key in candidate.cited_keys if key not in tools.retrieved]
    if not candidate.cited_keys:
        return CheckResult("cites_evidence", False, "no evidence segment supports the program")
    if missing:
        return CheckResult("cites_evidence", False, f"cites unretrieved segments {missing}")
    return CheckResult("cites_evidence", True, f"cites {list(candidate.cited_keys)}")


def _grounding_check(candidate: CandidateProgram, tools: ToolBox) -> CheckResult:
    texts = [tools.segment_text(key) for key in candidate.cited_keys if key in tools.retrieved]
    grounding = operand_grounding(candidate.program, texts)
    if grounding is None:
        return CheckResult("operands_grounded", False, "program does not parse")
    if grounding.ungrounded:
        return CheckResult(
            "operands_grounded",
            False,
            f"literals not in cited evidence: {list(grounding.ungrounded)}",
        )
    return CheckResult(
        "operands_grounded", True, f"{grounding.literals} literal(s) found in cited evidence"
    )


def _plausibility_check(plan: ResearchPlan, candidate, value) -> CheckResult:
    if isinstance(value, str):
        ok = plan.task is TaskType.COMPARISON
        return CheckResult("plausible", ok, f"yes/no result for task {plan.task.value}")
    if plan.task is TaskType.COMPARISON:
        return CheckResult("plausible", False, "comparison task produced a number")
    if plan.task is TaskType.PERCENT_CHANGE and abs(value) > MAX_ABS_PERCENT_CHANGE:
        return CheckResult("plausible", False, f"percent change {value:.4g} exceeds +/-1000%")
    question = plan.question.lower()
    if plan.task is TaskType.RATIO and any(word in question for word in _SHARE_WORDS):
        if not 0 <= abs(value) <= 1:
            return CheckResult("plausible", False, f"a share of a whole cannot be {value:.4g}")
    if plan.task is TaskType.CHANGE and value == 0:
        return CheckResult("plausible", False, "a change of exactly zero: same value twice")
    return CheckResult("plausible", True, f"{value:.6g} is plausible for {plan.task.value}")


def _period_check(plan: ResearchPlan, candidate: CandidateProgram) -> CheckResult:
    if not plan.periods or not candidate.operands:
        return CheckResult("periods_match", True, "no period constraint to check", hard=False)
    operand_periods = {period for op in candidate.operands for period in op.mention.periods}
    missing = [period for period in plan.periods if period not in operand_periods]
    if plan.task is TaskType.RATIO:
        missing = [] if plan.periods[-1] in operand_periods or not operand_periods else missing
    if missing:
        return CheckResult(
            "periods_match", False, f"no operand from period(s) {missing}", hard=False
        )
    return CheckResult("periods_match", True, "operands cover the named periods", hard=False)


def _conflict_check(candidate: CandidateProgram, mentions: Sequence[ValueMention]) -> CheckResult:
    used = {(op.mention.label.lower(), op.mention.periods) for op in candidate.operands}
    conflicts = [
        group
        for group in conflicting_mentions(mentions)
        if (group[0].label.lower(), group[0].periods) in used
    ]
    if conflicts:
        detail = "; ".join(
            f"'{group[0].label}' {group[0].periods}: "
            + ", ".join(f"{m.literal} ({m.segment_key})" for m in group)
            for group in conflicts
        )
        return CheckResult("no_conflicting_evidence", False, detail, hard=False)
    return CheckResult("no_conflicting_evidence", True, "no conflicting values", hard=False)
