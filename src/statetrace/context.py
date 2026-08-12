"""Build bounded, explicit prompts for the agent decision core."""

from __future__ import annotations

import json
from typing import Any

from .models import AgentTask
from .protocol import protocol_instruction


def build_prompt(
    task: AgentTask,
    tool_descriptions: list[dict[str, Any]],
    *,
    max_observation_chars: int = 12_000,
) -> str:
    recent = []
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
    history = json.dumps(recent, ensure_ascii=False, indent=2)
    if len(history) > max_observation_chars:
        history = history[-max_observation_chars:]
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
