"""Run several systems over one split under identical conditions and compare them.

Systems are anything with ``name`` and ``run(record) -> AgentAnswer``. Each record's answer
is saved as one JSON line, so every summary number can be re-derived from the artifact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from finevidence.evaluation.agentic import (
    AgentRecordScore,
    mcnemar_counts,
    risk_coverage,
    score_agent_answer,
    summarize_agent_scores,
)

from .schema import AgentAnswer

RISK_THRESHOLDS = (0.0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8)


class System(Protocol):
    name: str

    def run(self, record: Mapping[str, Any]) -> AgentAnswer: ...


def run_system(
    system: System, records: Sequence[Mapping[str, Any]], *, output: Path | None = None
) -> list[AgentRecordScore]:
    scores = []
    handle = output.open("w", encoding="utf-8") if output else None
    try:
        for record in records:
            answer = system.run(record)
            scores.append(score_agent_answer(record, answer))
            if handle:
                handle.write(json.dumps(answer.to_dict()) + "\n")
    finally:
        if handle:
            handle.close()
    return scores


def compare_systems(
    systems: Sequence[System],
    records: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path | None = None,
    split: str = "dev",
) -> dict[str, Any]:
    """Summaries per system, paired wins/losses against the first system, risk-coverage."""
    results: dict[str, Any] = {"split": split, "records": len(records), "systems": {}}
    all_scores: dict[str, list[AgentRecordScore]] = {}
    for system in systems:
        output = output_dir / f"{system.name}_{split}_answers.jsonl" if output_dir else None
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
        scores = run_system(system, records, output=output)
        all_scores[system.name] = scores
        results["systems"][system.name] = summarize_agent_scores(scores)
    reference = systems[0].name
    results["paired_vs_" + reference] = {
        name: mcnemar_counts(all_scores[reference], scores)
        for name, scores in all_scores.items()
        if name != reference
    }
    results["risk_coverage"] = {
        name: risk_coverage(scores, RISK_THRESHOLDS) for name, scores in all_scores.items()
    }
    results["by_task"] = {name: _by_task(scores) for name, scores in all_scores.items()}
    return results


def _by_task(scores: Sequence[AgentRecordScore]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for score in scores:
        row = table.setdefault(score.task, {"records": 0, "answered": 0, "correct": 0})
        row["records"] += 1
        row["answered"] += score.answered
        row["correct"] += score.correct
    return dict(sorted(table.items(), key=lambda item: -item[1]["records"]))


def format_comparison(results: Mapping[str, Any]) -> str:
    """A Markdown table of the headline metrics."""
    columns = [
        ("execution_accuracy", "Exec acc"),
        ("coverage", "Coverage"),
        ("selective_accuracy", "Selective acc"),
        ("wrong_answer_rate", "Wrong-answer rate"),
        ("answered_cites_all_gold", "Cites all gold"),
        ("answered_gold_recall", "Gold recall"),
        ("mean_tool_calls", "Tool calls"),
        ("mean_ms", "ms / q"),
    ]
    lines = [
        "| System | " + " | ".join(label for _, label in columns) + " |",
        "| --- |" + " --- |" * len(columns),
    ]
    for name, summary in results["systems"].items():
        cells = []
        for key, _ in columns:
            value = summary[key]
            if key.startswith("mean"):
                cells.append(f"{value:.1f}")
            else:
                cells.append(f"{value * 100:.2f}%")
        ci = (
            f" [{summary['execution_accuracy_ci_low'] * 100:.2f}, "
            f"{summary['execution_accuracy_ci_high'] * 100:.2f}]"
        )
        cells[0] += ci
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
