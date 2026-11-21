"""Render an ``AgentAnswer`` as a Markdown research report an analyst can audit."""

from __future__ import annotations

from .schema import AgentAnswer


def render_markdown(answer: AgentAnswer, *, show_tools: bool = True) -> str:
    lines = [f"## Question\n\n{answer.question}\n"]
    if answer.answered:
        lines.append(f"## Answer: {answer.display_answer}\n")
        lines.append(
            f"Confidence {answer.confidence:.2f} · {answer.rounds} retrieval round(s) · "
            f"{len(answer.tool_calls)} tool calls · {answer.milliseconds:.0f} ms\n"
        )
    else:
        lines.append("## Abstained\n")
        lines.append(f"Reason: {answer.abstention_reason}\n")
        if answer.withheld_value is not None:
            lines.append(
                f"Best unverified candidate (withheld): `{answer.program}` = "
                f"{answer.withheld_value}\n"
            )

    lines.append("## Plan\n")
    lines.append(f"{answer.plan.rationale}\n")
    lines.extend(
        f"{index}. **{step.action}**: {step.description}"
        for index, step in enumerate(answer.plan.steps, start=1)
    )
    lines.append("")

    if answer.citations:
        lines.append("## Evidence\n")
        lines.extend(f"- `[{citation.key}]` {citation.text}" for citation in answer.citations)
        lines.append("")

    if answer.calculation:
        lines.append("## Calculation\n")
        lines.append(f"Program: `{answer.program}`\n")
        lines.extend(
            f"- `#{step.index}` {step.operation}({', '.join(step.arguments)}) = {step.result}"
            for step in answer.calculation
        )
        lines.append("")

    if answer.checks:
        lines.append("## Verification\n")
        lines.extend(
            f"- {'PASS' if check.passed else 'FAIL'} {check.name}"
            f"{'' if check.hard else ' (soft)'}: {check.detail}"
            for check in answer.checks
        )
        lines.append("")

    if answer.alternatives:
        lines.append("## Other candidates considered\n")
        lines.extend(f"- `{alternative}`" for alternative in answer.alternatives)
        lines.append("")

    if show_tools:
        lines.append("## Tool log\n")
        lines.append("| Round | Tool | Arguments | Result | ms |")
        lines.append("| --- | --- | --- | --- | --- |")
        for call in answer.tool_calls:
            arguments = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
            lines.append(
                f"| {call.round} | {call.tool} | {_cell(arguments)} | {_cell(call.summary)} "
                f"| {call.milliseconds:.1f} |"
            )
        lines.append("")
    lines.append("_Research prototype; not financial or investment advice._")
    return "\n".join(lines)


def _cell(text: str, limit: int = 90) -> str:
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"
