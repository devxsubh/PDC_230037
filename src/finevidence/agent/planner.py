"""Rule-based task planner: decide what a question asks for and which values to look up.

The planner reads only the question. It picks one calculation shape (``TaskType``), the
reporting periods named in the question, and the entity terms to search for, then writes the
ordered research steps the controller will carry out. Rules were written from train-split
questions; dev and test were not used to design them.
"""

from __future__ import annotations

import re

from finevidence.retrieval.lexical import tokenize

from .schema import PlanStep, ResearchPlan, TaskType

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

# Checked in order; the first matching rule decides the task. On train, words such as
# "maximum" or "average" usually name an entity ("minimum lease payments", "average price"),
# so the row and average rules come after the change and ratio rules.
_TASK_RULES: tuple[tuple[TaskType, re.Pattern[str]], ...] = (
    (
        TaskType.PERCENT_CHANGE,
        re.compile(
            r"\b(growth rate|rate of growth|percent(age)? (change|increase|decrease|decline|growth)"
            r"|% (change|increase|decrease)|change in percent(age)?|(increase|decrease|decline|"
            r"change|growth) in percent(age)?|percentage growth|cagr|grow by what percent"
            r"|by what percent(age)?|by how much (did|has|have|was|were|is|does))\b"
        ),
    ),
    (
        TaskType.COMPARISON,
        re.compile(r"^(is|was|were|are|did|does|do)\b.*\b(greater|more|higher|less|lower|exceed)"),
    ),
    (
        TaskType.CHANGE,
        re.compile(
            r"\b(change|difference|variation|increase|decrease|decline|growth|grew|"
            r"reduction|fluctuation|how much (more|less|higher|lower))\b"
        ),
    ),
    (
        TaskType.RATIO,
        re.compile(
            r"\b(percent(age)?|portion|proportion|ratio|share|fraction|rate|margin|%|"
            r"what part)\b"
        ),
    ),
    (TaskType.AVERAGE, re.compile(r"\b(average|mean)\b")),
    (TaskType.ROW_MAX, re.compile(r"\b(maximum|highest|largest|greatest|peak)\b")),
    (TaskType.ROW_MIN, re.compile(r"\b(minimum|lowest|smallest)\b")),
    (TaskType.SUM, re.compile(r"\b(total|combined|sum|together|aggregate|in all)\b")),
)

# Words that describe the calculation, not the entity being measured.
_TASK_WORDS = frozenset(
    """percent percentage change changes difference variation increase increased decrease
    decreased decline growth grew rate ratio portion proportion share fraction total combined
    sum average mean maximum minimum highest lowest largest smallest greatest between
    compared value amount year years period annual approximate approximately based
    respectively millions million billions billion thousands thousands dollars usd
    """.split()
)


_RATIO_SPLITS = (
    re.compile(r"\bratio of (?P<part>.+?) to (?P<whole>.+)"),
    re.compile(
        r"(?P<part>.+?)\b(?:as|is|was) an? (?:percent(?:age)?|portion|share|proportion|fraction)"
        r" of (?P<whole>.+)"
    ),
    re.compile(
        r"\b(?:percent(?:age)?|portion|proportion|share|fraction|%) of (?:the )?(?P<whole>.+?)"
        r" (?:is|was|were|are|came|comes|is from|was from|related to|relates to|attributable to"
        r"|consisted of|made up of|represented by|due to) (?P<part>.+)"
    ),
)


def ratio_terms(question: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(part terms, whole terms) when the question names both sides of a ratio."""
    text = question.lower()
    for pattern in _RATIO_SPLITS:
        match = pattern.search(text)
        if match:
            part, whole = entity_terms(match["part"]), entity_terms(match["whole"])
            if part and whole:
                return part, whole
    return (), ()


def classify_task(question: str) -> TaskType:
    text = question.lower()
    for task, pattern in _TASK_RULES:
        if pattern.search(text):
            return task
    return TaskType.UNSUPPORTED


def question_periods(question: str) -> tuple[str, ...]:
    """Distinct years in the question, oldest first."""
    return tuple(sorted(set(YEAR.findall(question))))


def entity_terms(question: str) -> tuple[str, ...]:
    """Content tokens of the question without task words, years or bare numbers."""
    terms = [
        token
        for token in tokenize(question)
        if token not in _TASK_WORDS and not YEAR.fullmatch(token) and not token[0].isdigit()
    ]
    return tuple(dict.fromkeys(terms))


def plan_question(question: str) -> ResearchPlan:
    """Build the research plan for one question."""
    task = classify_task(question)
    periods = question_periods(question)
    terms = entity_terms(question)
    entity = " ".join(terms) or "the asked quantity"
    steps = [PlanStep("search_evidence", f"retrieve table rows and sentences about {entity}")]
    if task in {TaskType.PERCENT_CHANGE, TaskType.CHANGE, TaskType.COMPARISON}:
        if len(periods) >= 2:
            old, new = periods[0], periods[-1]
        elif periods:
            old, new = f"the period before {periods[0]}", periods[0]
        else:
            old, new = "the earlier period", "the later period"
        steps += [
            PlanStep("extract_value", f"find {entity} for {old} (old value)"),
            PlanStep("extract_value", f"find {entity} for {new} (new value)"),
        ]
        calculation = {
            TaskType.PERCENT_CHANGE: "(new - old) / old",
            TaskType.CHANGE: "new - old",
            TaskType.COMPARISON: "is the first value greater than the second",
        }[task]
    elif task is TaskType.RATIO:
        part, whole = ratio_terms(question)
        steps += [
            PlanStep("extract_value", f"find the part: {' '.join(part) or entity}"),
            PlanStep(
                "extract_value",
                f"find the whole: {' '.join(whole)}"
                if whole
                else "find the whole or denominator in the same period",
            ),
        ]
        calculation = "part / whole"
    elif task in {TaskType.SUM, TaskType.AVERAGE}:
        span = ", ".join(periods) if periods else "each listed period"
        steps.append(PlanStep("extract_value", f"find {entity} for {span}"))
        calculation = "sum of values" if task is TaskType.SUM else "sum of values / count"
    elif task in {TaskType.ROW_MAX, TaskType.ROW_MIN}:
        steps.append(PlanStep("extract_value", f"find the table row for {entity}"))
        calculation = "maximum of row" if task is TaskType.ROW_MAX else "minimum of row"
    else:
        calculation = "none: the question shape is outside the supported tasks"
    if task is not TaskType.UNSUPPORTED:
        steps += [
            PlanStep("calculate", f"run the controlled calculator: {calculation}"),
            PlanStep("verify", "check grounding, periods, plausibility and conflicts"),
        ]
    part_terms, whole_terms = ratio_terms(question) if task is TaskType.RATIO else ((), ())
    return ResearchPlan(
        question=question,
        part_terms=part_terms,
        whole_terms=whole_terms,
        task=task,
        entity_terms=terms,
        periods=periods,
        steps=tuple(steps),
        rationale=f"task={task.value}; periods={list(periods) or 'none named'}; "
        f"calculation: {calculation}",
    )
