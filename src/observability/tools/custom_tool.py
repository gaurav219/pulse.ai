import json
from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class NormalizeMetricsToolInput(BaseModel):
    """Input schema for normalizing raw metrics."""

    metrics_payload: str = Field(
        ...,
        description="JSON string containing current and baseline metrics across layers.",
    )


class DiagnoseBottleneckToolInput(BaseModel):
    """Input schema for deterministic diagnosis."""

    question: str = Field(
        ...,
        description="Operator question to answer, such as why GPU utilization dropped.",
    )
    normalized_metrics_payload: str = Field(
        ...,
        description="JSON string produced by the normalize_metrics tool.",
    )


class DeterministicTriageEngine:
    METRIC_SPECS: dict[str, dict[str, dict[str, str]]] = {
        "gpu": {
            "utilization_pct": {"direction": "lower_is_bad", "unit": "%"},
            "memory_usage_pct": {"direction": "higher_is_bad", "unit": "%"},
            "power_w": {"direction": "lower_is_bad", "unit": " W"},
            "temperature_c": {"direction": "higher_is_bad", "unit": " C"},
        },
        "data_pipeline": {
            "data_loader_latency_ms": {"direction": "higher_is_bad", "unit": " ms"},
            "batch_prep_time_ms": {"direction": "higher_is_bad", "unit": " ms"},
            "queue_depth": {"direction": "lower_is_bad", "unit": ""},
            "dataset_read_throughput_mb_s": {"direction": "lower_is_bad", "unit": " MB/s"},
        },
        "network": {
            "rdma_latency_us": {"direction": "higher_is_bad", "unit": " us"},
            "packet_drops": {"direction": "higher_is_bad", "unit": ""},
            "retries": {"direction": "higher_is_bad", "unit": ""},
            "bandwidth_gbps": {"direction": "lower_is_bad", "unit": " Gbps"},
        },
        "storage": {
            "read_throughput_mb_s": {"direction": "lower_is_bad", "unit": " MB/s"},
            "iops": {"direction": "lower_is_bad", "unit": ""},
            "file_open_latency_ms": {"direction": "higher_is_bad", "unit": " ms"},
            "cache_hit_rate_pct": {"direction": "lower_is_bad", "unit": "%"},
        },
        "training": {
            "step_time_ms": {"direction": "higher_is_bad", "unit": " ms"},
            "tokens_per_sec": {"direction": "lower_is_bad", "unit": " tokens/s"},
            "batch_completion_time_ms": {"direction": "higher_is_bad", "unit": " ms"},
            "failure_rate_pct": {"direction": "higher_is_bad", "unit": "%"},
        },
    }

    LAYER_LABELS = {
        "gpu": "GPU",
        "data_pipeline": "Data pipeline",
        "network": "Network",
        "storage": "Storage",
        "training": "Training/application",
    }

    def normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized_layers: dict[str, Any] = {}

        for layer_name, metric_specs in self.METRIC_SPECS.items():
            layer_payload = payload.get(layer_name, {})
            normalized_metrics = []
            layer_score = 0.0

            for metric_name, spec in metric_specs.items():
                current = layer_payload.get(f"{metric_name}_current")
                baseline = layer_payload.get(f"{metric_name}_baseline")
                metric_result = self._normalize_metric(
                    metric_name=metric_name,
                    current=current,
                    baseline=baseline,
                    direction=spec["direction"],
                    unit=spec["unit"],
                )
                normalized_metrics.append(metric_result)
                layer_score += metric_result["severity_score"]

            normalized_layers[layer_name] = {
                "label": self.LAYER_LABELS[layer_name],
                "score": round(layer_score, 1),
                "status": self._layer_status(layer_score),
                "metrics": normalized_metrics,
            }

        return {
            "window": payload.get("window"),
            "normalized_metrics": normalized_layers,
        }

    def diagnose(self, question: str, normalized_payload: dict[str, Any]) -> dict[str, Any]:
        normalized_layers = normalized_payload["normalized_metrics"]
        candidate_scores = self._score_candidates(normalized_layers)
        ranked_diagnoses = self._rank_candidates(candidate_scores, normalized_layers)

        return {
            "question": question,
            "window": normalized_payload.get("window"),
            "normalized_metrics": normalized_layers,
            "candidate_scores": candidate_scores,
            "ranked_diagnoses": ranked_diagnoses,
            "top_diagnosis": ranked_diagnoses[0] if ranked_diagnoses else None,
        }

    def _score_candidates(self, normalized_layers: dict[str, Any]) -> dict[str, float]:
        gpu = normalized_layers["gpu"]["score"]
        data_pipeline = normalized_layers["data_pipeline"]["score"]
        network = normalized_layers["network"]["score"]
        storage = normalized_layers["storage"]["score"]
        training = normalized_layers["training"]["score"]

        gpu_idle_signals = self._metric_count(
            normalized_layers["gpu"]["metrics"],
            {"utilization_pct", "power_w"},
            minimum_score=20,
        )
        input_pipeline_pressure = self._metric_count(
            normalized_layers["data_pipeline"]["metrics"],
            {
                "data_loader_latency_ms",
                "batch_prep_time_ms",
                "dataset_read_throughput_mb_s",
                "queue_depth",
            },
            minimum_score=20,
        )
        storage_pressure = self._metric_count(
            normalized_layers["storage"]["metrics"],
            {
                "read_throughput_mb_s",
                "iops",
                "file_open_latency_ms",
                "cache_hit_rate_pct",
            },
            minimum_score=20,
        )
        network_pressure = self._metric_count(
            normalized_layers["network"]["metrics"],
            {"rdma_latency_us", "packet_drops", "retries", "bandwidth_gbps"},
            minimum_score=20,
        )
        training_regression = self._metric_count(
            normalized_layers["training"]["metrics"],
            {"step_time_ms", "tokens_per_sec", "batch_completion_time_ms", "failure_rate_pct"},
            minimum_score=20,
        )
        gpu_compute_pressure = self._metric_count(
            normalized_layers["gpu"]["metrics"],
            {"memory_usage_pct", "temperature_c"},
            minimum_score=20,
        )

        upstream_starvation = (
            storage_pressure >= 2
            and input_pipeline_pressure >= 2
            and gpu_idle_signals >= 2
        )
        network_starvation = (
            network_pressure >= 2
            and input_pipeline_pressure >= 1
            and gpu_idle_signals >= 2
        )
        likely_training_symptoms = upstream_starvation or network_starvation

        scores = {
            "storage_bottleneck": 0.0,
            "network_bottleneck": 0.0,
            "data_pipeline_bottleneck": 0.0,
            "gpu_compute_bottleneck": 0.0,
            "training_instability": 0.0,
            "inconclusive": 10.0,
        }

        scores["storage_bottleneck"] = (
            storage * 1.4 + data_pipeline * 0.9 + training * 0.5 + gpu_idle_signals * 8
        )
        if storage_pressure >= 2:
            scores["storage_bottleneck"] += 20
        if input_pipeline_pressure >= 2:
            scores["storage_bottleneck"] += 12
        if upstream_starvation:
            scores["storage_bottleneck"] += 120

        scores["network_bottleneck"] = (
            network * 1.5 + data_pipeline * 0.7 + training * 0.4 + gpu_idle_signals * 6
        )
        if network_pressure >= 2:
            scores["network_bottleneck"] += 20
        if input_pipeline_pressure >= 1:
            scores["network_bottleneck"] += 8
        if network_starvation:
            scores["network_bottleneck"] += 70

        scores["data_pipeline_bottleneck"] = (
            data_pipeline * 1.5 + storage * 0.5 + training * 0.5 + gpu_idle_signals * 7
        )
        if input_pipeline_pressure >= 2:
            scores["data_pipeline_bottleneck"] += 18
        if storage_pressure == 0 and network_pressure == 0:
            scores["data_pipeline_bottleneck"] += 10
        if upstream_starvation:
            scores["data_pipeline_bottleneck"] += 80

        scores["gpu_compute_bottleneck"] = gpu * 1.2 + training * 0.6
        if normalized_layers["gpu"]["score"] >= 50 and gpu_idle_signals == 0:
            scores["gpu_compute_bottleneck"] += 12
        if gpu_compute_pressure >= 2:
            scores["gpu_compute_bottleneck"] += 25
        if gpu_idle_signals >= 2:
            scores["gpu_compute_bottleneck"] -= 40

        scores["training_instability"] = training * 0.9 + gpu * 0.2
        if training_regression >= 2:
            scores["training_instability"] += 18
        if likely_training_symptoms:
            scores["training_instability"] -= 220
        if (
            storage_pressure == 0
            and input_pipeline_pressure == 0
            and network_pressure == 0
            and training_regression >= 2
        ):
            scores["training_instability"] += 60

        if max(scores.values()) < 45:
            scores["inconclusive"] += 30
        if (
            storage_pressure == 0
            and network_pressure == 0
            and input_pipeline_pressure == 0
            and training_regression == 0
        ):
            scores["inconclusive"] += 20

        return {name: round(score, 1) for name, score in scores.items()}

    def _rank_candidates(
        self, candidate_scores: dict[str, float], normalized_layers: dict[str, Any]
    ) -> list[dict[str, Any]]:
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_score = sorted_candidates[0][1] if sorted_candidates else 0.0
        diagnoses = []

        for candidate, score in sorted_candidates:
            diagnoses.append(
                {
                    "candidate": candidate,
                    "score": score,
                    "confidence": self._confidence(score, top_score),
                    "summary": self._candidate_summary(candidate),
                    "causal_chain": self._candidate_chain(candidate),
                    "supporting_layers": self._supporting_layers(candidate, normalized_layers),
                }
            )

        return diagnoses

    def _normalize_metric(
        self,
        *,
        metric_name: str,
        current: float | int | None,
        baseline: float | int | None,
        direction: str,
        unit: str,
    ) -> dict[str, Any]:
        if current is None or baseline is None:
            return {
                "metric": metric_name,
                "current": current,
                "baseline": baseline,
                "delta_pct": None,
                "ratio_to_baseline": None,
                "anomaly": "missing_data",
                "severity": "unknown",
                "severity_score": 0.0,
                "summary": f"{metric_name}: missing current or baseline data",
            }

        if baseline == 0:
            return {
                "metric": metric_name,
                "current": current,
                "baseline": baseline,
                "delta_pct": None,
                "ratio_to_baseline": None,
                "anomaly": "baseline_zero",
                "severity": "unknown",
                "severity_score": 0.0,
                "summary": f"{metric_name}: baseline is zero, cannot normalize",
            }

        raw_change_pct = ((current - baseline) / baseline) * 100
        anomaly_score = raw_change_pct if direction == "higher_is_bad" else -raw_change_pct
        ratio_to_baseline = round(current / baseline, 2)
        severity_score = round(max(anomaly_score, 0), 1)

        if severity_score >= 50:
            severity = "critical"
        elif severity_score >= 25:
            severity = "high"
        elif severity_score >= 10:
            severity = "medium"
        else:
            severity = "normal"

        anomaly = "regression" if severity_score >= 10 else "normal"
        verb = "increased" if raw_change_pct >= 0 else "dropped"
        summary = (
            f"{metric_name} {verb} from {baseline}{unit} to {current}{unit} "
            f"({round(raw_change_pct, 1)}% vs baseline, {ratio_to_baseline}x baseline)"
        )

        return {
            "metric": metric_name,
            "current": current,
            "baseline": baseline,
            "delta_pct": round(raw_change_pct, 1),
            "ratio_to_baseline": ratio_to_baseline,
            "anomaly": anomaly,
            "severity": severity,
            "severity_score": severity_score,
            "summary": summary,
        }

    @staticmethod
    def _layer_status(score: float) -> str:
        if score >= 120:
            return "critical"
        if score >= 60:
            return "degraded"
        if score >= 20:
            return "watch"
        return "healthy"

    @staticmethod
    def _metric_count(
        metrics: list[dict[str, Any]], metric_names: set[str], minimum_score: float
    ) -> int:
        return sum(
            1
            for metric in metrics
            if metric["metric"] in metric_names and metric["severity_score"] >= minimum_score
        )

    def _supporting_layers(
        self, candidate: str, normalized_layers: dict[str, Any]
    ) -> list[dict[str, Any]]:
        layer_map = {
            "storage_bottleneck": ["storage", "data_pipeline", "gpu", "training"],
            "network_bottleneck": ["network", "data_pipeline", "gpu", "training"],
            "data_pipeline_bottleneck": ["data_pipeline", "gpu", "training"],
            "gpu_compute_bottleneck": ["gpu", "training"],
            "training_instability": ["training", "gpu"],
            "inconclusive": ["gpu", "data_pipeline", "network", "storage", "training"],
        }
        return [
            {
                "layer": layer_name,
                "score": normalized_layers[layer_name]["score"],
                "status": normalized_layers[layer_name]["status"],
            }
            for layer_name in layer_map[candidate]
        ]

    @staticmethod
    def _confidence(score: float, top_score: float) -> str:
        if score >= 90 and (top_score - score) <= 15:
            return "High"
        if score >= 50:
            return "Medium"
        return "Low"

    @staticmethod
    def _candidate_summary(candidate: str) -> str:
        summaries = {
            "storage_bottleneck": "Storage regressions are likely starving the training input path.",
            "network_bottleneck": "Network transport issues likely slowed distributed training or remote reads.",
            "data_pipeline_bottleneck": "The data preparation path is likely not feeding the GPUs fast enough.",
            "gpu_compute_bottleneck": "The slowdown appears closer to GPU-side execution than input delivery.",
            "training_instability": "Application-level training regressions or failures are dominating the slowdown.",
            "inconclusive": "No single layer has enough deterministic evidence to isolate a root cause.",
        }
        return summaries[candidate]

    @staticmethod
    def _candidate_chain(candidate: str) -> str:
        chains = {
            "storage_bottleneck": "Storage degraded -> dataset reads slowed -> data pipeline stalled -> GPU usage fell",
            "network_bottleneck": "Network degraded -> remote transfer latency increased -> training steps waited -> throughput fell",
            "data_pipeline_bottleneck": "Batch prep slowed -> queue depth weakened -> GPUs waited for data -> utilization fell",
            "gpu_compute_bottleneck": "GPU-side execution regressed -> step time increased -> throughput fell",
            "training_instability": "Training/app instability increased -> step completion worsened -> throughput fell",
            "inconclusive": "Signals are mixed -> deterministic rules cannot isolate one root cause",
        }
        return chains[candidate]


class NormalizeMetricsTool(BaseTool):
    name: str = "normalize_metrics"
    description: str = (
        "Normalize raw multi-layer AI infrastructure metrics into a structured JSON "
        "payload with per-metric deltas, per-layer scores, and health status."
    )
    args_schema: Type[BaseModel] = NormalizeMetricsToolInput

    def _run(self, metrics_payload: str) -> str:
        payload = json.loads(metrics_payload)
        normalized = DeterministicTriageEngine().normalize_payload(payload)
        return json.dumps(normalized, indent=2)


class DiagnoseBottleneckTool(BaseTool):
    name: str = "diagnose_bottleneck"
    description: str = (
        "Take normalized metrics JSON from normalize_metrics and return a ranked, "
        "deterministic bottleneck diagnosis with candidate scores and causal chains."
    )
    args_schema: Type[BaseModel] = DiagnoseBottleneckToolInput

    def _run(self, question: str, normalized_metrics_payload: str) -> str:
        normalized_payload = json.loads(normalized_metrics_payload)
        diagnosis = DeterministicTriageEngine().diagnose(question, normalized_payload)
        return json.dumps(diagnosis, indent=2)
