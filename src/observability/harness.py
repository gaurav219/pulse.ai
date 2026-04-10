#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from observability.tools.custom_tool import DeterministicTriageEngine


def _load_payload(payload_path: str | None) -> tuple[str, dict]:
    if payload_path is None:
        from observability.main import _default_inputs

        inputs = _default_inputs()
        return inputs["question"], json.loads(inputs["metrics_payload"])

    payload = json.loads(Path(payload_path).read_text())
    question = payload.get("question", "Why did training performance regress?")
    metrics = payload.get("metrics", payload)
    return question, metrics


def _build_report(diagnosis: dict) -> str:
    top = diagnosis["top_diagnosis"]
    metrics = diagnosis["normalized_metrics"]

    evidence = []
    for layer_name in ("storage", "data_pipeline", "network", "gpu", "training"):
        for metric in metrics[layer_name]["metrics"]:
            if metric["anomaly"] == "regression":
                evidence.append(
                    f"- `{metric['metric']}`: {metric['summary']}, severity `{metric['severity']}`."
                )

    evidence = evidence[:6]

    return "\n".join(
        [
            "# Incident Note",
            "",
            "## Answer",
            top["summary"],
            "",
            "## Top Bottleneck",
            f"`{top['candidate']}` with score `{top['score']}`",
            "",
            "## Evidence",
            *evidence,
            "",
            "## Root Cause",
            top["causal_chain"],
            "",
            "## Confidence",
            top["confidence"],
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic bottleneck diagnosis on a metrics payload."
    )
    parser.add_argument(
        "--payload",
        help="Path to a JSON file. Supports either {question, metrics} or raw metrics JSON.",
    )
    parser.add_argument(
        "--expect-top",
        help="Expected top diagnosis candidate, such as storage_bottleneck.",
    )
    parser.add_argument(
        "--write-report",
        help="Optional path to write the generated markdown report.",
    )
    args = parser.parse_args()

    question, metrics = _load_payload(args.payload)
    engine = DeterministicTriageEngine()
    normalized = engine.normalize_payload(metrics)
    diagnosis = engine.diagnose(question, normalized)

    print(json.dumps(diagnosis, indent=2))

    if args.write_report:
        Path(args.write_report).write_text(_build_report(diagnosis) + "\n")

    if args.expect_top and diagnosis["top_diagnosis"]["candidate"] != args.expect_top:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "expected_top": args.expect_top,
                    "actual_top": diagnosis["top_diagnosis"]["candidate"],
                },
                indent=2,
            )
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
