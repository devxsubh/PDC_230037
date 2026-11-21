"""Assemble an agent from a config and optional saved artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from finevidence.generation.inputs import load_rankings
from finevidence.retrieval.lexical import Bm25Retriever, RankingFileRetriever, SegmentRetriever

from .config import AgentConfig
from .controller import FinancialResearchAgent
from .proposers import CandidateFileProposer, ProgramProposer, TemplateProposer


def build_agent(
    config: AgentConfig,
    *,
    rankings_path: Path | None = None,
    beams_path: Path | None = None,
    beams_facts: int = 5,
) -> FinancialResearchAgent:
    """``retriever = "bm25" | "ranking-file"``; ``proposer = "template" | "generator-beams"``."""
    retriever: SegmentRetriever
    if config.retriever == "bm25":
        retriever = Bm25Retriever()
    elif config.retriever == "ranking-file":
        if rankings_path is None:
            raise ValueError("retriever 'ranking-file' needs a rankings path (Notebook 10)")
        retriever = RankingFileRetriever(load_rankings(rankings_path))
    else:
        raise ValueError(f"unknown retriever {config.retriever!r}")

    proposer: ProgramProposer
    if config.proposer == "template":
        proposer = TemplateProposer()
    elif config.proposer == "generator-beams":
        if beams_path is None:
            raise ValueError("proposer 'generator-beams' needs a candidates path (Notebook 14)")
        proposer = CandidateFileProposer(load_beams(beams_path, facts=beams_facts))
    else:
        raise ValueError(f"unknown proposer {config.proposer!r}")
    return FinancialResearchAgent(retriever, proposer, config)


def load_beams(path: Path, *, facts: int = 5) -> dict[str, list[str]]:
    """Load Notebook 14 candidates (``record_id``, ``facts``, ``candidates``) for one fact count.

    Rows without a ``facts`` field (one setting per file) are always kept.
    """
    beams: dict[str, list[str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        row = json.loads(line)
        if "facts" in row and row["facts"] != facts:
            continue
        if not isinstance(row.get("record_id"), str) or not isinstance(row.get("candidates"), list):
            raise ValueError(f"line {line_number}: expected 'record_id' and 'candidates'")
        beams[row["record_id"]] = [str(program) for program in row["candidates"]]
    return beams
