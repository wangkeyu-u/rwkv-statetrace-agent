"""Run controller experiments; fake state is never reported as native RWKV inference."""

import json
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "state-resume": ["tests/test_controller.py", "tests/test_checkpoints.py"],
    "fork-behavior": ["tests/test_checkpoints.py", "-k", "clone"],
    "evidence-validation": ["tests/test_validator.py", "tests/test_protocol.py"],
}
report = {
    "created_at": datetime.now(UTC).isoformat(),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "scope": "Replay and fake-adapter controller contracts; live API and native runtime NOT RUN",
    "experiments": [],
}
for name, args in GROUPS.items():
    with tempfile.TemporaryDirectory() as temp:
        xml = Path(temp) / "results.xml"
        cmd = [sys.executable, "-m", "pytest", "-q", *args, f"--junitxml={xml}"]
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        cases = []
        if xml.exists():
            for case in ET.parse(xml).iter("testcase"):
                cases.append(
                    {
                        "name": case.attrib["name"],
                        "class": case.attrib.get("classname"),
                        "status": "failed"
                        if case.find("failure") is not None or case.find("error") is not None
                        else "skipped"
                        if case.find("skipped") is not None
                        else "passed",
                    }
                )
        report["experiments"].append(
            {
                "name": name,
                "command": ["python", "-m", "pytest", "-q", *args],
                "exit_code": result.returncode,
                "cases": cases,
            }
        )
fixture = subprocess.run(
    [sys.executable, "-m", "pytest", "-q"], cwd=ROOT / "examples/fixture_repo", text=True, capture_output=True
)
report["intentional_fixture"] = {
    "exit_code": fixture.returncode,
    "expected_failure_observed": fixture.returncode == 1 and "3 failed, 1 passed" in fixture.stdout,
    "output": fixture.stdout.replace(str(ROOT), "<repo>"),
}
output = ROOT / "docs/experiments/contract-results.json"
output.write_text(json.dumps(report, indent=2) + "\n")
print(output)
if (
    any(e["exit_code"] or not e["cases"] for e in report["experiments"])
    or not report["intentional_fixture"]["expected_failure_observed"]
):
    sys.exit(1)
