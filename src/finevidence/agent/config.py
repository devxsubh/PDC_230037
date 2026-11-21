"""Versioned agent settings, loaded from ``configs/agent_*.toml``."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentConfig:
    """Retrieval budget, verification and abstention policy for one agent run."""

    name: str = "agent-offline-v1"
    retriever: str = "bm25"
    proposer: str = "template"
    initial_k: int = 5
    k_step: int = 5
    max_rounds: int = 3
    max_candidates_verified: int = 12
    verify: bool = True
    abstain: bool = True
    min_confidence: float = 0.60
    ambiguity_margin: float = 0.01
    abstain_on_conflict: bool = True
    period_mismatch_penalty: float = 0.6

    def with_overrides(self, **overrides: Any) -> AgentConfig:
        return replace(self, **overrides)


def load_agent_config(path: Path) -> AgentConfig:
    """Read ``[agent]``, ``[retrieval]``, ``[verification]`` and ``[abstention]`` tables."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    known = {field.name for field in fields(AgentConfig)}
    values: dict[str, Any] = {}
    if "experiment_name" in payload:
        values["name"] = payload["experiment_name"]
    for table in ("agent", "retrieval", "verification", "abstention"):
        for key, value in payload.get(table, {}).items():
            if key not in known:
                raise ValueError(f"{path.name}: unknown setting [{table}].{key}")
            values[key] = value
    config = AgentConfig(**values)
    if config.initial_k < 1 or config.max_rounds < 1:
        raise ValueError("initial_k and max_rounds must be at least 1")
    if not 0.0 <= config.min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    return config
