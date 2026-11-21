"""Small synthetic FinQA-shaped records shared by the agent tests. Not real report data."""

from __future__ import annotations

import copy
from typing import Any

_REPORT: dict[str, Any] = {
    "id": "SYNTH/2017/page_1.pdf-1",
    "pre_text": [
        "the company operates in two segments .",
        "net sales increased due to higher volume .",
    ],
    "post_text": ["interest expense was $ 40 million in 2017 ."],
    "table": [
        ["( in millions )", "2017", "2016"],
        ["net sales", "$ 1100", "$ 1000"],
        ["cost of sales", "600", "550"],
        ["operating income", "( 50 )", "120"],
        ["total expenses", "$ 2000", "$ 1800"],
    ],
    "qa": {
        "question": "what was the percentage change in net sales from 2016 to 2017?",
        "program": "subtract(1100, 1000), divide(#0, 1000)",
        "exe_ans": 0.1,
        "answer": "10%",
        "gold_inds": {"table_1": "the net sales of 2017 is $ 1100 ;"},
        "explanation": "",
    },
}


def record(question: str | None = None, **qa: Any) -> dict[str, Any]:
    """A fresh copy of the synthetic record, optionally with a new question/annotations."""
    result = copy.deepcopy(_REPORT)
    if question is not None:
        result["qa"]["question"] = question
    result["qa"].update(qa)
    return result


def inference_only(full: dict[str, Any]) -> dict[str, Any]:
    """The record with every annotation removed, as a user-supplied report would be."""
    return {
        "id": full["id"],
        "qa": {"question": full["qa"]["question"]},
        "table": full["table"],
        "pre_text": full["pre_text"],
        "post_text": full["post_text"],
    }
