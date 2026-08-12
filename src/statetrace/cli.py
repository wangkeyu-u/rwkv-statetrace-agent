"""Command-line entry point for StateTrace."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from importlib import resources, util
from pathlib import Path
from typing import Any

from .backends.api import RWKVAPIBackend
from .backends.replay import ReplayBackend
from .checkpoints import CheckpointManager
from .controller import AgentController, ControllerConfig
from .models import AgentTask
from .report import export_report
from .storage import TaskStore
from .tools.registry import default_registry
from .trace import TraceWriter
from .validator import ReportValidator

_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _state_root() -> Path:
    configured = os.getenv("STATETRACE_HOME", "").strip()
    return Path(configured or ".statetrace").expanduser().resolve()


def _task_id(value: str) -> str:
    if not _SAFE_TASK_ID.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "task ID must start with a letter or digit and contain at most 128 "
            "letters, digits, dots, underscores, or hyphens"
        )
    return value


def _require_demo_dependencies() -> None:
    if util.find_spec("pytest") is None:
        raise RuntimeError(
            "The Replay demo executes pytest, but pytest is not installed. "
            "Install with `pip install -e '.[dev]'` from a clone or "
            "`pip install 'rwkv-statetrace-agent[demo]'` from a package."
        )


def _task_payload(task: AgentTask, trace_path: Path, report_paths: list[Path]) -> dict[str, Any]:
    return {
        **task.as_dict(),
        "workspace": str(task.workspace),
        "trace_path": str(trace_path),
        "report_paths": [str(item) for item in report_paths],
    }


def _run_task(*, backend: Any, workspace: Path, goal: str, max_steps: int) -> AgentTask:
    state_root = _state_root()
    task_id = None
    # Construct task first so each task owns a stable trace path.
    bootstrap = AgentController(backend=backend, tools=default_registry())
    task = bootstrap.new_task(goal, workspace, task_id)
    trace_path = state_root / "traces" / f"{task.task_id}.jsonl"
    trace = TraceWriter(trace_path, workspace=workspace)
    controller = AgentController(
        backend=backend,
        tools=default_registry(),
        validator=ReportValidator(workspace),
        trace=trace,
        checkpoint_manager=CheckpointManager(state_root / "checkpoints"),
        config=ControllerConfig(max_steps=max_steps),
    )
    # Record creation on the task's actual durable trace.
    trace.append("task_created", task_id=task.task_id, goal=task.goal, workspace=workspace)
    task = controller.run(task)
    reports = state_root / "reports"
    markdown_path = export_report(task, reports / f"{task.task_id}.md")
    html_path = export_report(task, reports / f"{task.task_id}.html")
    TaskStore(state_root / "tasks").save(
        task.task_id, _task_payload(task, trace_path, [markdown_path, html_path])
    )
    print(json.dumps({
        "task_id": task.task_id,
        "status": task.status.value,
        "mode": getattr(backend, "name", "unknown"),
        "trace": str(trace_path),
        "report": str(html_path),
    }, ensure_ascii=False, indent=2))
    return task


def cmd_demo(args: argparse.Namespace) -> int:
    _require_demo_dependencies()
    data = resources.files("statetrace").joinpath("demo_data")
    demo_parent = _state_root() / "demo-workspaces"
    demo_parent.mkdir(parents=True, exist_ok=True)
    demo_root = Path(tempfile.mkdtemp(prefix="demo-", dir=demo_parent))
    fixture_resource = data.joinpath("fixture")
    with resources.as_file(fixture_resource) as fixture_source:
        shutil.copytree(fixture_source, demo_root, dirs_exist_ok=True)
    replay_resource = data.joinpath("replay_demo.json")
    with resources.as_file(replay_resource) as replay_path:
        backend = ReplayBackend.from_file(replay_path)
    task = _run_task(
        backend=backend,
        workspace=demo_root,
        goal="Run the tests, diagnose the date-boundary failures, cite evidence, and recommend a minimal fix without modifying files.",
        max_steps=args.max_steps,
    )
    return 0 if task.status.value == "COMPLETED" else 1


def cmd_run(args: argparse.Namespace) -> int:
    if args.backend != "rwkv-api":
        print("rwkv-direct is an adapter contract; supply it from Python. CLI currently supports rwkv-api and replay demo.", file=sys.stderr)
        return 2
    base_url = os.getenv("RWKV_API_BASE_URL")
    model = os.getenv("RWKV_API_MODEL")
    if not base_url or not model:
        print("Set RWKV_API_BASE_URL and RWKV_API_MODEL.", file=sys.stderr)
        return 2
    backend = RWKVAPIBackend(base_url, model)
    task = _run_task(backend=backend, workspace=Path(args.workspace), goal=args.goal, max_steps=args.max_steps)
    return 0 if task.status.value == "COMPLETED" else 1


def cmd_status(args: argparse.Namespace) -> int:
    payload = TaskStore(_state_root() / "tasks").load(args.task_id)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    payload = TaskStore(_state_root() / "tasks").load(args.task_id)
    suffix = ".html" if args.format == "html" else ".md"
    candidates = [Path(item) for item in payload.get("report_paths", []) if str(item).endswith(suffix)]
    if not candidates or not candidates[0].exists():
        print(f"No {args.format} report found for {args.task_id}.", file=sys.stderr)
        return 1
    print(candidates[0])
    return 0


def cmd_fork(args: argparse.Namespace) -> int:
    path = CheckpointManager(_state_root() / "checkpoints").clone_checkpoint(
        source_task_id=args.task_id,
        new_task_id=args.new_task_id,
        step=args.from_step,
    )
    print(path)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    payload = TaskStore(_state_root() / "tasks").load(args.task_id)
    if payload.get("status") == "COMPLETED":
        print(f"Task {args.task_id} is already completed; no actions were repeated.")
        return 0
    print(
        "Generic CLI resume needs the original live backend instance. Use CheckpointManager.load "
        "and AgentController from Python; replay/live adapter-specific resume is intentionally explicit.",
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="statetrace", description="Auditable RWKV-oriented agent runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run the recorded, non-live demonstration")
    demo.add_argument("--max-steps", type=int, default=12)
    demo.set_defaults(func=cmd_demo)
    run = sub.add_parser("run", help="run with a live OpenAI-compatible RWKV API")
    run.add_argument("--workspace", required=True)
    run.add_argument("--goal", required=True)
    run.add_argument("--backend", choices=["rwkv-api", "rwkv-direct"], default="rwkv-api")
    run.add_argument("--max-steps", type=int, default=12)
    run.set_defaults(func=cmd_run)
    status = sub.add_parser("status", help="show a persisted task summary")
    status.add_argument("--task-id", required=True, type=_task_id)
    status.set_defaults(func=cmd_status)
    export = sub.add_parser("export", help="locate an exported report")
    export.add_argument("--task-id", required=True, type=_task_id)
    export.add_argument("--format", choices=["html", "markdown"], default="html")
    export.set_defaults(func=cmd_export)
    fork = sub.add_parser("fork", help="clone a task checkpoint")
    fork.add_argument("--task-id", required=True, type=_task_id)
    fork.add_argument("--new-task-id", required=True, type=_task_id)
    fork.add_argument("--from-step", type=int)
    fork.set_defaults(func=cmd_fork)
    resume = sub.add_parser("resume", help="resume or inspect a stopped task")
    resume.add_argument("--task-id", required=True, type=_task_id)
    resume.set_defaults(func=cmd_resume)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"statetrace: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
