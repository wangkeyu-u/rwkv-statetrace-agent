"""Human-readable Markdown and standalone HTML trace reports."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .models import AgentTask


def _backend_label(task: AgentTask) -> str:
    generations = [step.generation for step in task.steps if step.generation]
    if not generations:
        return "unknown"
    last = generations[-1]
    label = f"{last.backend_name} / {last.model_name}"
    if last.backend_name == "replay":
        label += " (recorded demonstration — not live model inference)"
    return label


def render_markdown(task: AgentTask) -> str:
    report = task.final_report
    lines = [
        f"# StateTrace report: {task.task_id}",
        "",
        f"- Goal: {task.goal}",
        f"- Status: {task.status.value}",
        f"- Backend: {_backend_label(task)}",
        f"- Steps: {task.step}",
        f"- Validation failures: {task.validation_failures}",
        "",
        "## Result",
        "",
    ]
    if report is None:
        lines.append("No validated final report was produced.")
    else:
        lines.extend([report.summary, "", "### Findings", ""])
        for finding in report.findings:
            refs = ", ".join(finding.evidence_ids)
            lines.append(f"- `{finding.file}:{finding.line}` — {finding.claim} (evidence: {refs})")
        lines.extend(["", "### Recommendations", ""])
        lines.extend(f"- {item}" for item in report.recommendations)
        if report.uncertainties:
            lines.extend(["", "### Uncertainties", ""])
            lines.extend(f"- {item}" for item in report.uncertainties)

    lines.extend(["", "## Execution timeline", ""])
    for step in task.steps:
        if step.action is None:
            continue
        action = step.action.as_dict()
        observation = step.observation.as_dict() if step.observation else None
        lines.append(f"### Step {step.number}: `{action.get('tool', action.get('type'))}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps({"action": action, "observation": observation}, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_html(task: AgentTask) -> str:
    markdown = render_markdown(task)
    # A dependency-free renderer: preserve the readable Markdown while wrapping
    # it in a styled, escaped document. JSON remains safe and selectable.
    body = html.escape(markdown)
    mode = "REPLAY" if "replay" in _backend_label(task) else "LIVE"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StateTrace {html.escape(task.task_id)}</title>
<style>
:root{{--bg:#0b1020;--panel:#111a2e;--text:#dbe7ff;--muted:#8fa5c9;--accent:#72e6b1}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace}}
main{{max-width:1000px;margin:auto;padding:48px 24px}} .badge{{display:inline-block;padding:4px 10px;border:1px solid var(--accent);border-radius:999px;color:var(--accent)}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--panel);padding:24px;border-radius:16px;border:1px solid #243354}}
</style></head><body><main><span class="badge">{mode}</span><pre>{body}</pre></main></body></html>"""


def export_report(task: AgentTask, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_html(task) if path.suffix.lower() == ".html" else render_markdown(task)
    path.write_text(content, encoding="utf-8")
    return path
