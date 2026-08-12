"""Build bounded, explicit prompts for the agent decision core."""

from __future__ import annotations

import json
from typing import Any

from .models import AgentTask
from .protocol import protocol_instruction


def _history_json(task: AgentTask, max_chars: int) -> str:
    """Serialize recent history without ever cutting through JSON syntax.

    Tool output can be very large.  Dropping whole old steps first preserves the
    latest feedback and yields a valid JSON value.  If one step alone exceeds
    the budget, retain its control metadata and a bounded data preview.
    """

    if max_chars < 2:
        return "[]"
    recent: list[dict[str, Any]] = []
    for step in task.steps[-6:]:
        if step.action is None:
            continue
        recent.append(
            {
                "step": step.number,
                "action": step.action.as_dict(),
                "observation": step.observation.as_dict() if step.observation else None,
            }
        )

    def encode(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    history = encode(recent)
    while len(history) > max_chars and len(recent) > 1:
        recent.pop(0)
        history = encode(recent)
    if len(history) <= max_chars:
        return history
    if not recent:
        return "[]"

    latest = recent[-1]
    raw_observation = latest.get("observation")
    observation: dict[str, Any] | None = None
    if isinstance(raw_observation, dict):
        data_text = encode(raw_observation.get("data", {}))
        preview_size = max(0, min(1000, max_chars // 3))
        observation = {
            key: raw_observation[key]
            for key in ("type", "status", "error_code", "evidence_id", "truncated")
            if key in raw_observation
        }
        observation["message"] = str(raw_observation.get("message", ""))[:preview_size]
        observation["data_preview"] = data_text[:preview_size]
        observation["prompt_truncated"] = True
    compact = [{"step": latest["step"], "action": latest["action"], "observation": observation}]
    history = encode(compact)
    if len(history) <= max_chars:
        return history

    # Pathological actions (for example a malformed model output echoed by the
    # protocol tool) are reduced to identifiers rather than slicing JSON text.
    action = latest.get("action")
    summary = [
        {
            "step": latest["step"],
            "action": {
                "type": action.get("type"),
                "tool": action.get("tool"),
            }
            if isinstance(action, dict)
            else None,
            "observation": {
                "status": observation.get("status"),
                "error_code": observation.get("error_code"),
                "evidence_id": observation.get("evidence_id"),
                "prompt_truncated": True,
            }
            if observation
            else None,
        }
    ]
    history = encode(summary)
    return history if len(history) <= max_chars else "[]"


def build_prompt(
    task: AgentTask,
    tool_descriptions: list[dict[str, Any]],
    *,
    max_observation_chars: int = 12_000,
) -> str:
    history = _history_json(task, max_observation_chars)
    return (
        "You are the decision core of a bounded, read-only repository analysis agent.\n"
        f"Goal: {task.goal}\n"
        "Choose the single next action from the current evidence. Tool errors are observations: "
        "correct the call or change route. Finish only when every claim has file, line, and evidence IDs.\n"
        f"Recent execution state:\n{history}\n\n"
        + protocol_instruction(tool_descriptions)
    )


def build_incremental_prompt(task: AgentTask) -> str:
    """Return only information not already encoded in a recurrent state.

    A native RWKV state returned after generation already contains the initial
    contract, prior observations, and the model's own action. The next input is
    therefore just the new environment observation plus a short instruction.
    """

    if not task.steps:
        return "Choose the next action as one JSON object."
    step = task.steps[-1]
    observation = step.observation.as_dict() if step.observation else {
        "type": "observation",
        "status": "success",
        "message": "The previous action was recorded.",
    }
    return (
        "Environment observation for the previous action:\n"
        + json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
        + "\nChoose the single next action using the previously supplied JSON contract."
    )
