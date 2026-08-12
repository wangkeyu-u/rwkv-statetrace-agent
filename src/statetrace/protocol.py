"""Parser for the small, explicit model-to-agent action protocol."""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any

from .models import (
    AgentAction,
    ErrorCode,
    FinalAction,
    FinalReport,
    Finding,
    JSONValue,
    Observation,
    ToolCall,
)


class ProtocolError(ValueError):
    """A model output error which is safe to feed back as an observation."""

    def __init__(self, code: ErrorCode, message: str, *, data: dict[str, JSONValue] | None = None):
        super().__init__(message)
        self.code = code
        self.data = data or {}

    def as_observation(self) -> Observation:
        return Observation(
            status="error", message=str(self), error_code=self.code, data=self.data
        )


def _extract_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        raise ProtocolError(ErrorCode.INVALID_MODEL_OUTPUT, "Model output was empty.")

    decoder = json.JSONDecoder()
    # Scan for a decodable object. This covers fenced JSON and small prose
    # wrappers without accepting arrays or silently merging multiple actions.
    failures = 0
    for position, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(cleaned[position:])
        except JSONDecodeError:
            failures += 1
            continue
        if isinstance(value, dict):
            suffix = cleaned[position + end :]
            for next_position, next_character in enumerate(suffix):
                if next_character != "{":
                    continue
                try:
                    extra, _ = decoder.raw_decode(suffix[next_position:])
                except JSONDecodeError:
                    continue
                if isinstance(extra, dict):
                    raise ProtocolError(
                        ErrorCode.INVALID_MODEL_OUTPUT,
                        "Output contains multiple JSON objects; return exactly one action.",
                    )
            return value
    code = ErrorCode.INVALID_JSON if failures else ErrorCode.INVALID_MODEL_OUTPUT
    raise ProtocolError(code, "Output must contain one valid JSON object.")


def _required_string(data: dict[str, Any], key: str, location: str = "action") -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(
            ErrorCode.INVALID_ARGUMENTS,
            f"{location}.{key} must be a non-empty string.",
        )
    return value.strip()


def _string_list(value: Any, location: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ProtocolError(ErrorCode.INVALID_ARGUMENTS, f"{location} must be a list of strings.")
    return tuple(item.strip() for item in value if item.strip())


def _parse_final(arguments: dict[str, Any]) -> FinalAction:
    findings_value = arguments.get("findings")
    if not isinstance(findings_value, list):
        raise ProtocolError(ErrorCode.INVALID_ARGUMENTS, "arguments.findings must be a list.")
    findings: list[Finding] = []
    for index, raw in enumerate(findings_value):
        location = f"arguments.findings[{index}]"
        if not isinstance(raw, dict):
            raise ProtocolError(ErrorCode.INVALID_ARGUMENTS, f"{location} must be an object.")
        line = raw.get("line")
        if isinstance(line, bool) or not isinstance(line, int) or line < 1:
            raise ProtocolError(ErrorCode.INVALID_ARGUMENTS, f"{location}.line must be >= 1.")
        evidence_ids = _string_list(raw.get("evidence_ids"), f"{location}.evidence_ids")
        findings.append(
            Finding(
                file=_required_string(raw, "file", location),
                line=line,
                claim=_required_string(raw, "claim", location),
                evidence_ids=evidence_ids,
            )
        )
    verification = arguments.get("verification", {})
    if not isinstance(verification, dict):
        raise ProtocolError(
            ErrorCode.INVALID_ARGUMENTS, "arguments.verification must be an object."
        )
    return FinalAction(
        FinalReport(
            summary=_required_string(arguments, "summary", "arguments"),
            findings=tuple(findings),
            verification=verification,
            recommendations=_string_list(
                arguments.get("recommendations"), "arguments.recommendations"
            ),
            uncertainties=_string_list(arguments.get("uncertainties"), "arguments.uncertainties"),
        )
    )


def parse_action(text: str) -> AgentAction:
    """Parse exactly one tool or final action from potentially fenced output."""

    data = _extract_object(text)
    action_type = data.get("type")
    tool = data.get("tool")
    arguments = data.get("arguments")
    if action_type not in {"tool_call", "final"}:
        raise ProtocolError(
            ErrorCode.INVALID_ARGUMENTS,
            "action.type must be either 'tool_call' or 'final'.",
        )
    if not isinstance(tool, str) or not tool.strip():
        raise ProtocolError(ErrorCode.INVALID_ARGUMENTS, "action.tool must be a non-empty string.")
    if not isinstance(arguments, dict):
        raise ProtocolError(ErrorCode.INVALID_ARGUMENTS, "action.arguments must be an object.")
    if action_type == "final":
        if tool != "finish_report":
            raise ProtocolError(
                ErrorCode.INVALID_ARGUMENTS,
                "A final action must use the finish_report tool.",
            )
        return _parse_final(arguments)
    if tool == "finish_report":
        raise ProtocolError(
            ErrorCode.INVALID_ARGUMENTS,
            "finish_report must be emitted with action.type='final'.",
        )
    thought = data.get("thought_summary", "")
    if not isinstance(thought, str):
        raise ProtocolError(
            ErrorCode.INVALID_ARGUMENTS, "action.thought_summary must be a string."
        )
    return ToolCall(tool=tool.strip(), arguments=arguments, thought_summary=thought.strip())


def protocol_instruction(tool_descriptions: list[dict[str, JSONValue]]) -> str:
    """Return a compact contract suitable for inclusion in every model prompt."""

    schemas = json.dumps(tool_descriptions, ensure_ascii=False, indent=2)
    return (
        "Return exactly one JSON action. Do not use prose outside JSON. "
        "For a tool call use {\"type\":\"tool_call\",\"thought_summary\":\"brief reason\","
        "\"tool\":\"name\",\"arguments\":{...}}. For the final answer use "
        "{\"type\":\"final\",\"tool\":\"finish_report\",\"arguments\":{...}}. "
        "Use only observed evidence; never invent files, lines, commands, or results.\n"
        f"Available tools:\n{schemas}"
    )
