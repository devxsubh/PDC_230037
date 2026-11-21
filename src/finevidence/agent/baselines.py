"""The existing single-pass pipeline, wrapped so it is scored exactly like the agent.

``ConservativePipeline`` reproduces Notebook 08: retrieve the top 3 segments once, parse
their values and apply one cue-selected operation with ``reason_over_candidates``. Notebook 08
used scikit-learn TF-IDF; this variant uses the same BM25 retriever as the agent so the
comparison isolates the agentic components. It has no planner, no verification and no
iterative retrieval; it abstains only where ``reason_over_candidates`` does.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from finevidence.data.preprocessing import build_structured_input
from finevidence.reasoning.arithmetic import reason_over_candidates
from finevidence.retrieval.lexical import Bm25Retriever, SegmentRetriever
from finevidence.retrieval.pairs import segment_retrieval_text
from finevidence.retrieval.segments import segment_numeric_candidates

from .planner import plan_question
from .schema import AgentAnswer, AnswerStatus, Citation, ToolCall


class ConservativePipeline:
    name = "pipeline-conservative-arithmetic"

    def __init__(self, retriever: SegmentRetriever | None = None, *, top_k: int = 3) -> None:
        self.retriever = retriever or Bm25Retriever()
        self.top_k = top_k

    def run(self, record: Mapping[str, Any]) -> AgentAnswer:
        started = time.perf_counter()
        source = build_structured_input(record)
        query_record = {**source, "qa": {"question": source["question"]}}
        selected = self.retriever.rank(query_record, source["question"])[: self.top_k]
        candidates = [
            candidate for item in selected for candidate in segment_numeric_candidates(item.segment)
        ]
        trace = reason_over_candidates(source["question"], candidates)
        value = float(trace.result) if trace.answered else None
        keys = tuple(dict.fromkeys(trace.evidence_references))
        texts = {item.key: segment_retrieval_text(item.segment) for item in selected}
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        return AgentAnswer(
            record_id=source["id"],
            question=source["question"],
            status=AnswerStatus.ANSWERED if trace.answered else AnswerStatus.ABSTAINED,
            value=round(value, 5) if value is not None else None,
            display_answer=None if value is None else f"{value:.5g}",
            program=None,
            citations=tuple(Citation(key, texts[key]) for key in keys if key in texts),
            calculation=(),
            checks=(),
            plan=plan_question(source["question"]),
            tool_calls=(ToolCall(1, "retrieve", {"k": self.top_k}, str(list(texts)), elapsed),),
            rounds=1,
            confidence=1.0 if trace.answered else 0.0,
            abstention_reason=trace.abstention_reason,
            milliseconds=elapsed,
        )
