import json
from pathlib import Path

import pytest

from agent_fixtures import record
from finevidence.agent import AgentConfig, build_agent, load_agent_config
from finevidence.agent.baselines import ConservativePipeline
from finevidence.agent.benchmark import compare_systems, format_comparison
from finevidence.agent.cli import main
from finevidence.agent.factory import load_beams
from finevidence.evaluation import (
    bootstrap_interval,
    risk_coverage,
    score_agent_answer,
    summarize_agent_scores,
)
from finevidence.reasoning import parse_numeric_candidate, reason_over_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_correct_answer_is_scored_with_evidence_completeness():
    score = score_agent_answer(record(), build_agent(AgentConfig()).run(record()))
    assert score.correct and score.answered and score.best_effort_correct
    assert score.cites_all_gold and score.gold_recall == 1.0 and score.citation_precision == 1.0


def test_abstention_counts_as_incorrect_but_withheld_value_is_best_effort():
    conflicted = record()
    conflicted["table"].append(["net sales", "$ 1100", "$ 1150"])  # conflicting 2016 value
    score = score_agent_answer(conflicted, build_agent(AgentConfig()).run(conflicted))
    assert not score.answered and not score.correct and score.best_effort_correct
    summary = summarize_agent_scores([score], bootstrap_samples=20)
    assert summary["execution_accuracy"] == 0.0
    assert summary["coverage"] == 0.0 and summary["wrong_answer_rate"] == 0.0
    assert summary["best_effort_accuracy"] == 1.0


def test_risk_coverage_trades_coverage_for_accuracy():
    agent = build_agent(AgentConfig())
    scores = [
        score_agent_answer(record(), agent.run(record())),
        score_agent_answer(record(exe_ans=0.5), agent.run(record())),
    ]
    (row,) = risk_coverage(scores, [0.0])
    assert row["answered"] == 2 and row["correct"] == 1


def test_bootstrap_is_deterministic_and_bounded():
    values = [1.0, 0.0, 1.0, 1.0]
    assert bootstrap_interval(values, seed=1) == bootstrap_interval(values, seed=1)
    low, high = bootstrap_interval(values, seed=1)
    assert 0.0 <= low <= 0.75 <= high <= 1.0


def test_compare_systems_reports_paired_counts_and_table(tmp_path):
    records = [record(), record("who audits the company?", exe_ans=1.0)]
    systems = [ConservativePipeline(), build_agent(AgentConfig())]
    results = compare_systems(systems, records, output_dir=tmp_path)
    assert set(results["systems"]) == {"pipeline-conservative-arithmetic", "agent-offline-v1"}
    assert "agent-offline-v1" in results["paired_vs_pipeline-conservative-arithmetic"]
    assert (tmp_path / "agent-offline-v1_dev_answers.jsonl").read_text().count("\n") == 2
    assert format_comparison(results).startswith("| System |")


def test_conservative_arithmetic_no_longer_divides_by_zero_for_other_operations():
    candidates = [
        parse_numeric_candidate("5", source_id="table_1", context=""),
        parse_numeric_candidate("0", source_id="table_1", context=""),
    ]
    trace = reason_over_candidates("what was the change in revenue?", candidates)
    assert trace.result == 5


def test_repository_configs_load(tmp_path):
    for name in ("agent_offline", "agent_single_pass", "agent_no_abstention"):
        config = load_agent_config(PROJECT_ROOT / "configs" / f"{name}.toml")
        assert config.name.startswith("agent-")
    bad = tmp_path / "bad.toml"
    bad.write_text("[retrieval]\nunknown = 1\n")
    with pytest.raises(ValueError, match="unknown setting"):
        load_agent_config(bad)


def test_load_beams_filters_by_fact_count(tmp_path):
    path = tmp_path / "beams.jsonl"
    rows = [
        {"record_id": "a", "facts": 5, "candidates": ["add(1, 2)"]},
        {"record_id": "a", "facts": 3, "candidates": ["add(3, 4)"]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert load_beams(path, facts=5) == {"a": ["add(1, 2)"]}


def test_cli_runs_the_generator_config_on_saved_artifacts(tmp_path, capsys):
    report = record()
    payload = {key: report[key] for key in ("id", "table", "pre_text", "post_text")}
    payload["question"] = report["qa"]["question"]
    (tmp_path / "report.json").write_text(json.dumps(payload))
    (tmp_path / "rankings.jsonl").write_text(
        json.dumps({"id": report["id"], "ranked_keys": ["table_1", "table_2"]}) + "\n"
    )
    beams = ["divide(1100, 1234)", "subtract(1100, 1000), divide(#0, 1000)"]
    (tmp_path / "beams.jsonl").write_text(
        json.dumps({"record_id": report["id"], "facts": 5, "candidates": beams}) + "\n"
    )
    arguments = ["ask", "--record", str(tmp_path / "report.json"), "--json"]
    arguments += ["--config", str(PROJECT_ROOT / "configs" / "agent_generator.toml")]
    arguments += ["--rankings", str(tmp_path / "rankings.jsonl")]
    arguments += ["--beams", str(tmp_path / "beams.jsonl")]
    assert main(arguments) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["program"] == beams[1] and answer["status"] == "answered"
    assert answer["tool_calls"][0]["arguments"]["retriever"] == "ranking-file"


def test_cli_ask_on_a_custom_report(tmp_path, capsys):
    report = record()
    payload = {key: report[key] for key in ("table", "pre_text", "post_text")}
    payload["question"] = report["qa"]["question"]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload))
    assert main(["ask", "--record", str(path)]) == 0
    assert "## Answer: 10.00%" in capsys.readouterr().out
    assert main(["ask", "--record", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["value"] == 0.1
