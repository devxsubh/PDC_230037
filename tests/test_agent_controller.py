from agent_fixtures import inference_only, record
from finevidence.agent import (
    AgentConfig,
    AnswerStatus,
    CandidateFileProposer,
    FinancialResearchAgent,
    TemplateProposer,
    build_agent,
    render_markdown,
)
from finevidence.agent.tools import ToolBox, calculation_trace
from finevidence.retrieval.lexical import Bm25Retriever


def _agent(**overrides):
    return build_agent(AgentConfig().with_overrides(**overrides))


def test_percent_change_is_answered_with_citation_and_trace():
    answer = _agent().run(record())
    assert answer.status is AnswerStatus.ANSWERED
    assert answer.program == "subtract(1100, 1000), divide(#0, 1000)"
    assert answer.value == 0.1
    assert answer.display_answer == "10.00%"
    assert [citation.key for citation in answer.citations] == ["table_1"]
    assert [step.result for step in answer.calculation] == [100.0, 0.1]
    assert all(check.passed for check in answer.checks)
    assert answer.tool_calls[0].tool == "search_evidence"


def test_agent_reads_no_annotation_field():
    full = _agent().run(record())
    stripped = _agent().run(inference_only(record()))
    assert (stripped.program, stripped.value) == (full.program, full.value)


def test_ratio_uses_the_total_row_as_denominator():
    answer = _agent().run(record("what portion of the total expenses in 2017 is cost of sales?"))
    assert answer.program == "divide(600, 2000)"
    assert answer.answered


def test_change_with_negative_accounting_value():
    answer = _agent().run(record("what was the change in operating income from 2016 to 2017?"))
    assert answer.program == "subtract(-50, 120)"
    assert answer.value == -170.0


def test_average_never_mixes_unlike_columns():
    mixed = record("what is the average payment volume per transaction?")
    mixed["table"] = [
        ["", "payments volume", "total transactions", "cards"],
        ["payment volume", "637", "5.0", "86"],
    ]
    answer = _agent().run(mixed)
    assert answer.program is None or "const_3" not in answer.program


def test_average_over_named_periods():
    answer = _agent().run(record("what was the average net sales from 2016 to 2017?"))
    assert answer.program == "add(1000, 1100), divide(#0, const_2)"  # period order
    assert answer.value == 1050.0


def test_unsupported_question_abstains_without_spending_retrieval_budget():
    answer = _agent().run(record("who audits the company?"))
    assert answer.status is AnswerStatus.ABSTAINED
    assert "unsupported" in answer.abstention_reason
    assert answer.tool_calls == () and answer.rounds == 0


def test_conflicting_evidence_triggers_abstention():
    conflicted = record()
    conflicted["table"].append(["net sales", "$ 1150", "$ 1000"])
    answer = _agent().run(conflicted)
    assert answer.status is AnswerStatus.ABSTAINED
    assert answer.abstention_reason.startswith("conflicting evidence")
    assert answer.withheld_value is not None


def test_weak_evidence_abstains_but_keeps_the_withheld_value():
    # "dividends" matches no row label, so any candidate rests on retrieval rank alone.
    answer = _agent().run(record("what was the percentage change in dividends from 2016 to 2017?"))
    assert answer.status is AnswerStatus.ABSTAINED
    assert answer.abstention_reason.startswith(("low confidence", "ambiguous"))
    assert answer.value is None and answer.withheld_value is not None


def test_single_pass_answers_without_verification():
    answer = _agent(verify=False, abstain=False, max_rounds=1).run(record())
    assert answer.answered and answer.checks == () and answer.rounds == 1


def test_generator_beams_are_verified_and_ungrounded_beams_rejected():
    beams = {record()["id"]: ["divide(1100, 9999)", "subtract(1100, 1000), divide(#0, 1000)"]}
    agent = FinancialResearchAgent(Bm25Retriever(), CandidateFileProposer(beams), AgentConfig())
    answer = agent.run(record())
    assert answer.program == "subtract(1100, 1000), divide(#0, 1000)"
    assert answer.answered


def test_missing_beams_mean_insufficient_evidence():
    agent = FinancialResearchAgent(Bm25Retriever(), CandidateFileProposer({}), AgentConfig())
    answer = agent.run(record())
    assert answer.status is AnswerStatus.ABSTAINED
    assert "insufficient evidence" in answer.abstention_reason
    assert answer.rounds == AgentConfig().max_rounds


def test_tools_log_every_call_and_refuse_unretrieved_segments():
    tools = ToolBox(record(), Bm25Retriever())
    tools.search_evidence("net sales", 2)
    try:
        tools.extract_values(["text_0"])
    except KeyError:
        pass
    else:
        raise AssertionError("extract_values must refuse segments that were not retrieved")
    tools.calculate("divide(1, 0)")
    assert [call.tool for call in tools.calls] == ["search_evidence", "calculate"]
    assert tools.calls[-1].summary.startswith("error")


def test_calculation_trace_reports_each_step():
    trace = calculation_trace("add(1, 2), multiply(#0, 3)", [])
    assert [(step.operation, step.result) for step in trace] == [("add", 3.0), ("multiply", 9.0)]


def test_report_and_json_views():
    answer = _agent().run(record())
    report = render_markdown(answer)
    for heading in ("## Answer: 10.00%", "## Evidence", "## Calculation", "## Verification"):
        assert heading in report
    payload = answer.to_dict()
    assert payload["status"] == "answered"
    assert payload["plan"]["task"] == "percent_change"


def test_template_proposer_returns_no_candidates_without_mentions():
    from finevidence.agent import ProposalContext, plan_question

    context = ProposalContext("x", plan_question("what was the change in x?"), [], {}, {}, [])
    assert TemplateProposer().propose(context) == []
