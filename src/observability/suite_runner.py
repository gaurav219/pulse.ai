#!/usr/bin/env python
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def _extract_top_bottleneck(report_text: str) -> str | None:
    match = re.search(r"`([^`]+)`\s+with score", report_text)
    if match:
        return match.group(1)
    return None


def _run_case(case_path: Path, reports_dir: Path, project_root: Path) -> dict:
    case = json.loads(case_path.read_text())
    expected_top = case.get("expected_top")
    report_path = reports_dir / f"{case_path.stem}.md"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "observability.case_runner",
            "--payload",
            str(case_path),
            "--output",
            str(report_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    report_text = report_path.read_text() if report_path.exists() else ""
    actual_top = _extract_top_bottleneck(report_text)
    passed = expected_top == actual_top and completed.returncode == 0

    return {
        "name": case_path.stem,
        "expected_top": expected_top,
        "actual_top": actual_top,
        "passed": passed,
        "report_path": str(report_path),
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full CrewAI+Ollama regression suite over JSON metric fixtures."
    )
    parser.add_argument(
        "--cases-dir",
        default="tests/cases",
        help="Directory containing JSON fixture files.",
    )
    parser.add_argument(
        "--reports-dir",
        default="tests/reports",
        help="Directory where markdown reports should be written.",
    )
    parser.add_argument(
        "--summary-file",
        default="tests/reports/summary.txt",
        help="Path to the final pass/fail summary text file.",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    cases_dir = project_root / args.cases_dir
    reports_dir = project_root / args.reports_dir
    summary_file = project_root / args.summary_file

    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    case_paths = sorted(cases_dir.glob("*.json"))
    results = [_run_case(case_path, reports_dir, project_root) for case_path in case_paths]

    passed = sum(1 for result in results if result["passed"])
    failed = len(results) - passed

    lines = [
        f"Total tests: {len(results)}",
        f"Passed: {passed}",
        f"Failed: {failed}",
        "",
    ]
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(
            f"{status} {result['name']}: expected={result['expected_top']} actual={result['actual_top']} returncode={result['returncode']} report={result['report_path']}"
        )
        if result["stderr"]:
            lines.append(f"stderr: {result['stderr']}")

    summary_file.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
