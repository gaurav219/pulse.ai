#!/usr/bin/env python
import json
import sys
import warnings

from observability.crew import Observability
from observability.pipeline import build_inputs

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def _default_inputs() -> dict[str, str]:
    sample_metrics = {
        "window": "last_15m vs trailing_24h_baseline",
        "gpu": {
            "utilization_pct_current": 40,
            "utilization_pct_baseline": 82,
            "memory_usage_pct_current": 54,
            "memory_usage_pct_baseline": 71,
            "power_w_current": 210,
            "power_w_baseline": 315,
            "temperature_c_current": 58,
            "temperature_c_baseline": 71,
        },
        "data_pipeline": {
            "data_loader_latency_ms_current": 120,
            "data_loader_latency_ms_baseline": 60,
            "batch_prep_time_ms_current": 72,
            "batch_prep_time_ms_baseline": 36,
            "queue_depth_current": 2,
            "queue_depth_baseline": 7,
            "dataset_read_throughput_mb_s_current": 440,
            "dataset_read_throughput_mb_s_baseline": 920,
        },
        "network": {
            "rdma_latency_us_current": 28,
            "rdma_latency_us_baseline": 22,
            "packet_drops_current": 1,
            "packet_drops_baseline": 0,
            "retries_current": 3,
            "retries_baseline": 1,
            "bandwidth_gbps_current": 92,
            "bandwidth_gbps_baseline": 100,
        },
        "storage": {
            "read_throughput_mb_s_current": 480,
            "read_throughput_mb_s_baseline": 980,
            "iops_current": 8300,
            "iops_baseline": 14500,
            "file_open_latency_ms_current": 18,
            "file_open_latency_ms_baseline": 7,
            "cache_hit_rate_pct_current": 74,
            "cache_hit_rate_pct_baseline": 93,
        },
        "training": {
            "step_time_ms_current": 1450,
            "step_time_ms_baseline": 920,
            "tokens_per_sec_current": 9100,
            "tokens_per_sec_baseline": 15400,
            "batch_completion_time_ms_current": 1510,
            "batch_completion_time_ms_baseline": 970,
            "failure_rate_pct_current": 1.4,
            "failure_rate_pct_baseline": 0.2,
        },
    }

    return build_inputs("Why is GPU utilization at 40%?", sample_metrics)


def run():
    """Run the crew."""
    try:
        Observability().crew().kickoff(inputs=_default_inputs())
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """Train the crew for a given number of iterations."""
    try:
        Observability().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=_default_inputs(),
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """Replay the crew execution from a specific task."""
    try:
        Observability().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """Test the crew execution and returns the results."""
    try:
        Observability().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=_default_inputs(),
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    """Run the crew with trigger payload."""
    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        raise Exception("Invalid JSON payload provided as argument") from exc

    inputs = {
        **build_inputs(
            trigger_payload.get("question", "Why did training performance regress?"),
            trigger_payload.get("metrics", {}),
        ),
    }

    try:
        return Observability().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
