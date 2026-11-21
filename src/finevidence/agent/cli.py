"""Command line: ``finevidence-agent ask`` answers one question; ``evaluate`` compares systems.

Examples (from the repository root)::

    finevidence-agent ask --split dev --index 0
    finevidence-agent ask --record my_report.json --json
    finevidence-agent evaluate --split dev

``--record`` takes one JSON object with ``question``, ``table``, ``pre_text`` and
``post_text`` (``id`` optional), so the agent can be run on a report outside FinQA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from finevidence.data.finqa import load_finqa_split

from .baselines import ConservativePipeline
from .benchmark import compare_systems, format_comparison
from .config import load_agent_config
from .factory import build_agent
from .report import render_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "agent_offline.toml"
ABLATION_CONFIGS = (
    PROJECT_ROOT / "configs" / "agent_single_pass.toml",
    PROJECT_ROOT / "configs" / "agent_no_abstention.toml",
)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "agent"
EVALUATION_SPLITS = ("train", "dev")  # the test split is reserved for the final M4 run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finevidence-agent", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="answer one question with citations and a trace")
    source = ask.add_mutually_exclusive_group(required=True)
    source.add_argument("--record", type=Path, help="JSON file with one report and question")
    source.add_argument("--index", type=int, help="record position in --split")
    source.add_argument("--id", help="FinQA record id in --split")
    ask.add_argument("--split", default="dev", choices=EVALUATION_SPLITS)
    ask.add_argument("--question", help="override the record's question")
    ask.add_argument("--json", action="store_true", help="print the full answer as JSON")
    _add_agent_options(ask)

    evaluate = commands.add_parser("evaluate", help="compare pipeline, ablation and agent")
    evaluate.add_argument("--split", default="dev", choices=EVALUATION_SPLITS)
    evaluate.add_argument("--limit", type=int, help="first N records only (smoke runs)")
    evaluate.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    evaluate.add_argument("--results", type=Path, help="also write the summary JSON here")
    _add_agent_options(evaluate)

    args = parser.parse_args(argv)
    if args.command == "ask":
        return _ask(args)
    return _evaluate(args)


def _add_agent_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rankings", type=Path, help="Notebook 10 rankings (ranking-file)")
    parser.add_argument("--beams", type=Path, help="Notebook 14 candidates (generator-beams)")


def _agent(args: argparse.Namespace):
    return build_agent(
        load_agent_config(args.config), rankings_path=args.rankings, beams_path=args.beams
    )


def _ask(args: argparse.Namespace) -> int:
    if args.record:
        record = _custom_record(json.loads(args.record.read_text(encoding="utf-8")))
    else:
        records = load_finqa_split(RAW_DIR / f"{args.split}.json")
        if args.id:
            matches = [record for record in records if record["id"] == args.id]
            if not matches:
                print(f"no record with id {args.id!r} in {args.split}", file=sys.stderr)
                return 2
            record = matches[0]
        else:
            record = records[args.index]
    if args.question:
        record = {**record, "qa": {**record.get("qa", {}), "question": args.question}}
    answer = _agent(args).run(record)
    print(json.dumps(answer.to_dict(), indent=2) if args.json else render_markdown(answer))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    records = load_finqa_split(RAW_DIR / f"{args.split}.json")
    if args.limit:
        records = records[: args.limit]
    systems = [
        ConservativePipeline(),
        *(build_agent(load_agent_config(path)) for path in ABLATION_CONFIGS),
        _agent(args),
    ]
    results = compare_systems(systems, records, output_dir=args.output_dir, split=args.split)
    print(format_comparison(results))
    print(json.dumps(results["paired_vs_" + systems[0].name], indent=2))
    if args.results:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


def _custom_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a user report in the FinQA record shape the agent reads."""
    required = ("question", "table", "pre_text", "post_text")
    missing = [field for field in required if field not in payload]
    if missing:
        raise SystemExit(f"--record is missing fields: {missing}")
    return {
        "id": str(payload.get("id", "custom")),
        "qa": {"question": payload["question"]},
        "table": payload["table"],
        "pre_text": payload["pre_text"],
        "post_text": payload["post_text"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
