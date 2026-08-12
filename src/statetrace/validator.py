"""Deterministic validation for evidence-backed final reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class ValidationError:
    code: str
    message: str
    finding_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.finding_index is not None:
            result["finding_index"] = self.finding_index
        return result


@dataclass(slots=True)
class ValidationResult:
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_observation(self) -> dict[str, Any]:
        if self.passed:
            return {"type": "observation", "status": "validation_passed", "errors": []}
        return {
            "type": "observation",
            "status": "validation_failed",
            "errors": [error.as_dict() for error in self.errors],
            "instruction": "Please collect missing evidence or correct the final report.",
        }


def _resolve_in_workspace(workspace: Path, supplied: Any) -> Path | None:
    if not isinstance(supplied, str) or not supplied.strip():
        return None
    candidate = Path(supplied)
    candidate = candidate if candidate.is_absolute() else workspace / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(workspace)
    except (OSError, ValueError):
        return None
    return resolved


def _evidence_file(evidence: Mapping[str, Any]) -> str | None:
    artifact = evidence.get("artifact")
    if isinstance(artifact, Mapping):
        artifact_file = artifact.get("file")
        if isinstance(artifact_file, str):
            return artifact_file
    arguments = evidence.get("arguments")
    if isinstance(arguments, Mapping):
        for key in ("path", "file"):
            argument_file = arguments.get(key)
            if isinstance(argument_file, str):
                return argument_file
    evidence_file = evidence.get("file")
    if isinstance(evidence_file, str):
        return evidence_file
    return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Normalize dependency-free project models without importing them here."""

    if isinstance(value, Mapping):
        return value
    if hasattr(value, "as_dict") and callable(value.as_dict):
        converted = value.as_dict()
        if isinstance(converted, Mapping):
            return converted
    if is_dataclass(value) and not isinstance(value, type):
        converted = asdict(cast(Any, value))
        if isinstance(converted, Mapping):
            return converted
    raise TypeError(f"Expected a mapping or serializable model, got {type(value).__name__}")


def evidence_from_steps(steps: Iterable[Any]) -> dict[str, dict[str, Any]]:
    """Convert ``AgentStep`` objects into validator evidence records.

    ``AgentTask.evidence`` stores Observations for quick lookup, while tool name
    and arguments live on the corresponding AgentStep. This helper joins them
    so the validator can check provenance rather than merely ID existence.
    """

    result: dict[str, dict[str, Any]] = {}
    for step in steps:
        observation = getattr(step, "observation", None)
        evidence_id = getattr(observation, "evidence_id", None)
        if not evidence_id:
            continue
        action = getattr(step, "action", None)
        record = dict(_as_mapping(observation))
        if action is not None:
            action_record = _as_mapping(action)
            record["tool"] = action_record.get("tool")
            record["arguments"] = action_record.get("arguments", {})
        record["id"] = evidence_id
        record["step"] = getattr(step, "number", None)
        result[str(evidence_id)] = record
    return result


class ReportValidator:
    """Check report structure, source locations, and evidence provenance."""

    def __init__(self, workspace: str | Path, *, require_test_evidence: bool = True) -> None:
        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {self.workspace}")
        self.require_test_evidence = require_test_evidence

    def validate(
        self,
        report: Any,
        evidence: Iterable[Any] | Mapping[str, Any],
    ) -> ValidationResult:
        report = _as_mapping(report)
        # Accept the complete model action as well as finish_report arguments.
        if isinstance(report.get("arguments"), Mapping):
            report = report["arguments"]
        evidence_by_id: dict[str, dict[str, Any]]
        if isinstance(evidence, Mapping):
            evidence_by_id = {str(key): dict(_as_mapping(value)) for key, value in evidence.items()}
        else:
            evidence_by_id = {}
            for raw_item in evidence:
                item = _as_mapping(raw_item)
                evidence_id = item.get("id", item.get("evidence_id"))
                if evidence_id is not None:
                    evidence_by_id[str(evidence_id)] = dict(item)

        errors: list[ValidationError] = []
        if not isinstance(report.get("summary"), str) or not report["summary"].strip():
            errors.append(ValidationError("MISSING_SUMMARY", "Final report requires a non-empty summary."))
        findings = report.get("findings")
        if not isinstance(findings, Sequence) or isinstance(findings, (str, bytes)) or not findings:
            errors.append(ValidationError("MISSING_FINDINGS", "Final report requires at least one finding."))
            findings = []

        for index, finding in enumerate(findings):
            if not isinstance(finding, Mapping):
                errors.append(ValidationError("INVALID_FINDING", "Finding must be an object.", index))
                continue
            if not isinstance(finding.get("claim"), str) or not finding["claim"].strip():
                errors.append(ValidationError("MISSING_CLAIM", "Finding requires a non-empty claim.", index))
            supplied_file = finding.get("file")
            resolved = _resolve_in_workspace(self.workspace, supplied_file)
            if resolved is None:
                errors.append(
                    ValidationError("INVALID_FILE", f"Finding file is missing or outside the workspace: {supplied_file!r}.", index)
                )
            elif not resolved.is_file():
                errors.append(ValidationError("FILE_NOT_FOUND", f"Finding file does not exist: {supplied_file}.", index))

            line = finding.get("line")
            if not isinstance(line, int) or isinstance(line, bool) or line < 1:
                errors.append(ValidationError("INVALID_LINE_NUMBER", "Finding line must be an integer >= 1.", index))
            elif resolved is not None and resolved.is_file():
                try:
                    with resolved.open("r", encoding="utf-8", errors="replace") as handle:
                        line_count = sum(1 for _ in handle)
                except OSError:
                    line_count = 0
                if line > line_count:
                    errors.append(
                        ValidationError(
                            "INVALID_LINE_NUMBER",
                            f"{supplied_file} has {line_count} lines but the finding cites line {line}.",
                            index,
                        )
                    )

            evidence_ids = finding.get("evidence_ids")
            if (
                not isinstance(evidence_ids, Sequence)
                or isinstance(evidence_ids, (str, bytes))
                or not evidence_ids
            ):
                errors.append(ValidationError("MISSING_EVIDENCE", "Finding must cite at least one evidence ID.", index))
                continue
            for evidence_id in evidence_ids:
                evidence_record = evidence_by_id.get(str(evidence_id))
                if evidence_record is None:
                    errors.append(
                        ValidationError("UNKNOWN_EVIDENCE", f"Evidence ID does not exist: {evidence_id}.", index)
                    )
                    continue
                evidence_path = _evidence_file(evidence_record)
                if supplied_file and evidence_path:
                    report_path = _resolve_in_workspace(self.workspace, supplied_file)
                    observed_path = _resolve_in_workspace(self.workspace, evidence_path)
                    if report_path is not None and observed_path is not None and report_path != observed_path:
                        errors.append(
                            ValidationError(
                                "EVIDENCE_FILE_MISMATCH",
                                f"Evidence {evidence_id} concerns {evidence_path}, not {supplied_file}.",
                                index,
                            )
                        )

        verification = report.get("verification")
        tests_run = verification.get("tests_run") if isinstance(verification, Mapping) else None
        test_evidence = [test_item for test_item in evidence_by_id.values() if test_item.get("tool") == "run_tests"]
        if self.require_test_evidence:
            if not isinstance(tests_run, Sequence) or isinstance(tests_run, (str, bytes)) or not tests_run:
                errors.append(ValidationError("MISSING_TEST_VERIFICATION", "Report must list tests that were run."))
            if not test_evidence:
                errors.append(ValidationError("UNVERIFIED_TEST_RESULT", "No run_tests evidence exists for this report."))
            elif isinstance(tests_run, Sequence) and not isinstance(tests_run, (str, bytes)):
                actual_commands = {
                    str(test_item.get("arguments", {}).get("command"))
                    for test_item in test_evidence
                    if isinstance(test_item.get("arguments"), Mapping)
                }
                for command in tests_run:
                    if str(command) not in actual_commands:
                        errors.append(
                            ValidationError(
                                "UNVERIFIED_TEST_COMMAND",
                                f"Report claims test command {command!r}, but no matching evidence exists.",
                            )
                        )
        return ValidationResult(errors)


# Concise compatibility alias for callers that prefer ``Validator``.
Validator = ReportValidator
