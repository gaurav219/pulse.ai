import json

from observability.tools.custom_tool import DeterministicTriageEngine


def _compact_diagnosis_payload(question: str, diagnosis: dict) -> str:
    top = diagnosis["top_diagnosis"]
    evidence: list[str] = []

    for layer_name in ("storage", "data_pipeline", "network", "gpu", "training"):
        for metric in diagnosis["normalized_metrics"][layer_name]["metrics"]:
            if metric["anomaly"] == "regression":
                evidence.append(metric["summary"])

    compact = {
        "question": question,
        "top_bottleneck": top["candidate"],
        "score": top["score"],
        "confidence": top["confidence"],
        "summary": top["summary"],
        "root_cause": top["causal_chain"],
        "supporting_layers": top["supporting_layers"],
        "evidence": evidence[:5],
    }
    return json.dumps(compact, indent=2)


def build_inputs(question: str, metrics: dict) -> dict[str, str]:
    engine = DeterministicTriageEngine()
    normalized = engine.normalize_payload(metrics)
    diagnosis = engine.diagnose(question, normalized)

    return {
        "question": question,
        "metrics_payload": json.dumps(metrics),
        "diagnosis_payload": _compact_diagnosis_payload(question, diagnosis),
    }
