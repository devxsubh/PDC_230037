"""Agentic financial research assistant built on the FinEvidence pipeline components.

The agent plans a question (``planner``), searches evidence and reads values with controlled
tools (``tools``, ``evidence``), proposes calculation programs (``proposers``), checks them
(``verifier``) and answers with citations and a calculation trace, or abstains
(``controller``). ``factory`` assembles an agent from a versioned config; ``cli`` is the
command-line entry point (``finevidence-agent``).
"""

from .config import AgentConfig, load_agent_config
from .controller import FinancialResearchAgent, display_value
from .factory import build_agent
from .planner import classify_task, plan_question
from .proposers import CandidateFileProposer, ProposalContext, TemplateProposer
from .report import render_markdown
from .schema import (
    AgentAnswer,
    AnswerStatus,
    CandidateProgram,
    CheckResult,
    Citation,
    ResearchPlan,
    TaskType,
    ValueMention,
)

__all__ = [
    "AgentAnswer",
    "AgentConfig",
    "AnswerStatus",
    "CandidateFileProposer",
    "CandidateProgram",
    "CheckResult",
    "Citation",
    "FinancialResearchAgent",
    "ProposalContext",
    "ResearchPlan",
    "TaskType",
    "TemplateProposer",
    "ValueMention",
    "build_agent",
    "classify_task",
    "display_value",
    "load_agent_config",
    "plan_question",
    "render_markdown",
]
