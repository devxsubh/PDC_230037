import pytest

from finevidence.agent import TaskType, classify_task, plan_question
from finevidence.agent.planner import entity_terms, question_periods, ratio_terms


@pytest.mark.parametrize(
    ("question", "task"),
    [
        ("what was the percentage change in net sales from 2016 to 2017?", "percent_change"),
        ("what is the growth rate in brent oil prices from 2016 to 2017?", "percent_change"),
        ("by how much did net undeveloped acres decrease from 2015 to 2016?", "percent_change"),
        ("what was the net change in asset retirement liability?", "change"),
        ("what portion of total expenses is cost of sales?", "ratio"),
        ("what was the effective tax rate in 2004?", "ratio"),
        ("what is the minimum lease payment as a percent of total payments?", "ratio"),
        ("what was the average net sales from 2016 to 2017?", "average"),
        ("what was the maximum interest expense?", "row_max"),
        # "rate" is a ratio cue and ratio rules outrank row cues (ordering set on train).
        ("what was the maximum interest rate?", "ratio"),
        ("what is the total combined royalty fees for 2006 and 2007?", "sum"),
        ("was net sales in 2017 greater than in 2016?", "comparison"),
        ("who audits the company?", "unsupported"),
    ],
)
def test_classify_task(question, task):
    assert classify_task(question).value == task


def test_periods_are_distinct_and_sorted():
    assert question_periods("change between 2017 and 2015 , and 2017 again") == ("2015", "2017")


def test_entity_terms_drop_task_words_years_and_numbers():
    assert entity_terms("what was the percentage change in net sales from 2016 to 2017?") == (
        "net",
        "sales",
    )


@pytest.mark.parametrize(
    ("question", "part", "whole"),
    [
        ("what is the ratio of net sales to total expenses?", ("net", "sales"), ("expenses",)),
        ("cost of sales as a percentage of net sales", ("cost", "sales"), ("net", "sales")),
        ("what portion of the total expenses is cost of sales?", ("cost", "sales"), ("expenses",)),
    ],
)
def test_ratio_terms_split_part_and_whole(question, part, whole):
    assert ratio_terms(question) == (part, whole)


def test_plan_lists_steps_through_verification():
    plan = plan_question("what was the percentage change in net sales from 2016 to 2017?")
    assert plan.task is TaskType.PERCENT_CHANGE
    assert plan.periods == ("2016", "2017")
    assert [step.action for step in plan.steps] == [
        "search_evidence",
        "extract_value",
        "extract_value",
        "calculate",
        "verify",
    ]
    assert "(new - old) / old" in plan.rationale


def test_unsupported_plan_has_no_calculation_step():
    plan = plan_question("who audits the company?")
    assert plan.task is TaskType.UNSUPPORTED
    assert [step.action for step in plan.steps] == ["search_evidence"]
