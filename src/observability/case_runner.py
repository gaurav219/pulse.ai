#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from observability.crew import Observability
from observability.pipeline import build_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one CrewAI diagnosis case.")
    parser.add_argument("--payload", required=True, help="Path to the JSON fixture.")
    parser.add_argument("--output", required=True, help="Path to write the markdown report.")
    args = parser.parse_args()

    case = json.loads(Path(args.payload).read_text())
    question = case.get("question", "Why did training performance regress?")
    metrics = case.get("metrics", {})

    crew = Observability().crew()
    crew.verbose = False
    for agent in crew.agents:
        agent.verbose = False

    result = crew.kickoff(inputs=build_inputs(question, metrics))

    report_text = getattr(result, "raw", None)
    if not report_text:
        default_report = Path("report.md")
        report_text = default_report.read_text() if default_report.exists() else ""

    Path(args.output).write_text(report_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
