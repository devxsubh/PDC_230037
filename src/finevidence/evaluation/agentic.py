"""Metrics for comparing the agent with the single-pass pipeline.

Annotations (``exe_ans``, ``gold_inds``) are read here only, after an answer is fixed.

- **Answer accuracy**: official execution accuracy over all records (an abstention counts as
  incorrect), plus selective accuracy over answered records.
- **Reliability**: coverage, wrong-answer rate (answered and wrong, over all records) and the
  risk-coverage curve obtained by sweeping the confidence threshold.
- **Evidence completeness**: for answered records, whether the cited segments include every
  gold segment, the share of gold segments cited, and citation precision.
- **Efficiency**: tool calls, retrieval rounds and wall-clock latency per question.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .execution import compare_execution


@dataclass(frozen=True)
class AgentRecordScore:
    record_id: str
    task: str
    answered: bool
    correct: bool
    best_effort_correct: bool
    confidence: float
    cited_keys: tuple[str, ...]
    gold_keys: tuple[str, ...]
    tool_calls: int
    rounds: int
    milliseconds: float
    abstention_reason: str | None

    @property
    def cites_all_gold(self) -> bool:
        return bool(self.gold_keys) and set(self.gold_keys) <= set(self.cited_keys)

    @property
    def gold_recall(self) -> float:
        if not self.gold_keys:
            return 0.0
        return len(set(self.gold_keys) & set(self.cited_keys)) / len(set(self.gold_keys))

    @property
    def citation_precision(self) -> float:
        if not self.cited_keys:
            return 0.0
        return len(set(self.gold_keys) & set(self.cited_keys)) / len(set(self.cited_keys))


def score_agent_answer(record: Mapping[str, Any], answer: Any) -> AgentRecordScore:
    """Score one ``AgentAnswer`` (or any object with the same fields) against annotations."""
    reference = record["qa"]["exe_ans"]
    value = answer.value if answer.answered else None
    best_effort = value if answer.answered else answer.withheld_value
    return AgentRecordScore(
        record_id=str(record["id"]),
        task=answer.plan.task.value if answer.plan else "n/a",
        answered=answer.answered,
        correct=answer.answered and compare_execution(value, reference).is_correct,
        best_effort_correct=best_effort is not None
        and compare_execution(best_effort, reference).is_correct,
        confidence=answer.confidence,
        cited_keys=tuple(citation.key for citation in answer.citations),
        gold_keys=tuple(str(key) for key in record["qa"]["gold_inds"]),
        tool_calls=len(answer.tool_calls),
        rounds=answer.rounds,
        milliseconds=answer.milliseconds,
        abstention_reason=answer.abstention_reason,
    )


def summarize_agent_scores(
    scores: Sequence[AgentRecordScore], *, bootstrap_samples: int = 1000, seed: int = 42
) -> dict[str, float | int]:
    if not scores:
        raise ValueError("no scores to summarize")
    total = len(scores)
    answered = [score for score in scores if score.answered]
    correct = sum(score.correct for score in scores)
    low, high = bootstrap_interval(
        [float(score.correct) for score in scores], samples=bootstrap_samples, seed=seed
    )
    latencies = sorted(score.milliseconds for score in scores)
    return {
        "records": total,
        "execution_accuracy": round(correct / total, 6),
        "execution_accuracy_ci_low": round(low, 6),
        "execution_accuracy_ci_high": round(high, 6),
        "coverage": round(len(answered) / total, 6),
        "selective_accuracy": round(correct / len(answered), 6) if answered else 0.0,
        "wrong_answer_rate": round(sum(not s.correct for s in answered) / total, 6),
        "abstention_rate": round(1 - len(answered) / total, 6),
        "best_effort_accuracy": round(sum(s.best_effort_correct for s in scores) / total, 6),
        "answered_cites_all_gold": _mean([float(s.cites_all_gold) for s in answered]),
        "answered_gold_recall": _mean([s.gold_recall for s in answered]),
        "answered_citation_precision": _mean([s.citation_precision for s in answered]),
        "mean_tool_calls": _mean([float(s.tool_calls) for s in scores]),
        "mean_rounds": _mean([float(s.rounds) for s in scores]),
        "mean_ms": _mean(latencies),
        "p95_ms": round(latencies[min(total - 1, int(0.95 * total))], 3),
    }


def risk_coverage(
    scores: Iterable[AgentRecordScore], thresholds: Sequence[float]
) -> list[dict[str, float]]:
    """Coverage and selective accuracy if only answers with confidence >= t were given.

    Uses the best-effort value, so a withheld answer counts once its threshold is reached.
    """
    scores = [score for score in scores if score.confidence > 0]
    total = len(scores) or 1
    rows = []
    for threshold in thresholds:
        kept = [score for score in scores if score.confidence >= threshold]
        hits = sum(score.best_effort_correct for score in kept)
        rows.append(
            {
                "threshold": threshold,
                "coverage_of_scored": round(len(kept) / total, 6),
                "selective_accuracy": round(hits / len(kept), 6) if kept else 0.0,
                "correct": hits,
                "answered": len(kept),
            }
        )
    return rows


def bootstrap_interval(
    values: Sequence[float], *, samples: int = 1000, seed: int = 42, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean, with a fixed seed."""
    if not values:
        raise ValueError("no values")
    rng = random.Random(seed)
    count = len(values)
    means = sorted(sum(rng.choices(values, k=count)) / count for _ in range(samples))
    tail = (1 - level) / 2
    return means[int(tail * samples)], means[min(samples - 1, int((1 - tail) * samples))]


def mcnemar_counts(
    left: Sequence[AgentRecordScore], right: Sequence[AgentRecordScore]
) -> dict[str, int]:
    """Paired wins and losses of ``right`` against ``left`` on the same records."""
    by_id = {score.record_id: score for score in left}
    wins = losses = 0
    for score in right:
        other = by_id[score.record_id]
        wins += score.correct and not other.correct
        losses += other.correct and not score.correct
    return {"right_only_correct": wins, "left_only_correct": losses}


def _mean(values: Sequence[float]) -> float:
    return round(statistics.fmean(values), 6) if values else 0.0
