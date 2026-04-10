# Pulse: An AI Infra Bottleneck Detective for training performance optimization

This project is a CrewAI-based observability assistant built to answer questions like:

- Why is GPU utilization at 40%?
- Why did training throughput drop?
- Is the bottleneck storage, network, data loading, GPU compute, or the training application itself?

Instead of asking the LLM to infer everything from scratch, the system first performs a deterministic diagnosis over multi-layer metrics, then asks the CrewAI agent to turn that diagnosis into a short operator-facing incident note.

## What This Project Does

The project analyzes training slowdowns across five layers:

- GPU
- Data pipeline
- Network
- Storage
- Training/application

It compares current metrics against a baseline, computes anomaly severity, ranks likely bottlenecks, and generates a concise markdown report explaining the most likely cause.

Typical root-cause chains look like:

> Storage degraded -> dataset reads slowed -> data pipeline stalled -> GPU usage fell

## Architecture

The system is intentionally split into two stages.

### 1. Deterministic diagnosis

The deterministic engine:

- normalizes raw metrics against baselines
- computes per-metric severity scores
- assigns per-layer health states
- scores candidate bottlenecks
- ranks the most likely root cause

This logic lives in [custom_tool.py](/home/mellow/Desktop/observability/observability/src/observability/tools/custom_tool.py).

### 2. CrewAI explanation

CrewAI receives a compact diagnosis payload rather than the full raw metric set. The agent is instructed to:

- treat the deterministic diagnosis as final
- restate it in plain language
- cite the strongest evidence lines
- avoid reinterpreting or overriding the computed result

This prompt flow keeps token usage lower and reduces hallucination risk.

## Agent Structure

There is one main agent in this project:

- `bottleneck_detective`

Its job is to explain training slowdowns using the deterministic diagnosis results.

Relevant files:

- Agent config: [agents.yaml](/home/mellow/Desktop/observability/observability/src/observability/config/agents.yaml)
- Task config: [tasks.yaml](/home/mellow/Desktop/observability/observability/src/observability/config/tasks.yaml)
- Crew wiring: [crew.py](/home/mellow/Desktop/observability/observability/src/observability/crew.py)

## Metric Layers

The diagnosis engine currently supports these metrics.

### GPU

- `utilization_pct`
- `memory_usage_pct`
- `power_w`
- `temperature_c`

### Data pipeline

- `data_loader_latency_ms`
- `batch_prep_time_ms`
- `queue_depth`
- `dataset_read_throughput_mb_s`

### Network

- `rdma_latency_us`
- `packet_drops`
- `retries`
- `bandwidth_gbps`

### Storage

- `read_throughput_mb_s`
- `iops`
- `file_open_latency_ms`
- `cache_hit_rate_pct`

### Training/application

- `step_time_ms`
- `tokens_per_sec`
- `batch_completion_time_ms`
- `failure_rate_pct`

## Runtime Flow

The end-to-end flow is:

1. Raw metrics are provided as JSON.
2. The deterministic engine builds a normalized diagnosis.
3. A compact summary payload is generated in [pipeline.py](/home/mellow/Desktop/observability/observability/src/observability/pipeline.py).
4. CrewAI uses that summary to write a markdown incident note.
5. The final report is written to `report.md`.

## Model Provider

The project is currently configured to use OpenRouter with a lightweight default model in [crew.py](/home/mellow/Desktop/observability/observability/src/observability/crew.py):

- Default model: `openrouter/meta-llama/llama-3.2-3b-instruct`
- Base URL: `https://openrouter.ai/api/v1`

Environment variables supported:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`
- `OPENROUTER_BASE_URL`

### Example `.env`

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openrouter/meta-llama/llama-3.2-3b-instruct
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## Installation

Ensure you have Python `>=3.10,<3.14`.

Install dependencies with `uv`:

```bash
pip install uv
cd observability
crewai install
```

## Running The Crew

Run the default sample case:

```bash
crewai run
```

Or run the entrypoint directly:

```bash
./.venv/bin/python -m observability.main
```

The default sample input lives in [main.py](/home/mellow/Desktop/observability/observability/src/observability/main.py).

## Input Format

The runtime accepts:

- `question`
- `metrics_payload`

At the app layer, the project transforms this into:

- `question`
- `metrics_payload`
- `diagnosis_payload`

where `diagnosis_payload` is the compact deterministic summary passed to the CrewAI task.

## Testing And Fixtures

The project includes deterministic regression fixtures under [tests/cases](/home/mellow/Desktop/observability/observability/tests/cases).

These cover:

- storage starvation
- data pipeline bottlenecks
- network bottlenecks
- GPU compute bottlenecks
- training instability
- inconclusive cases

### Deterministic harness

You can run the deterministic harness without CrewAI:

```bash
./.venv/bin/python -m observability.harness \
  --payload tests/cases/01_storage_starvation_a.json \
  --expect-top storage_bottleneck \
  --write-report report.md
```

### Full suite runner

The project also includes a suite runner intended for full CrewAI runs:

```bash
./.venv/bin/python -m observability.suite_runner
```

Script aliases are defined in [pyproject.toml](/home/mellow/Desktop/observability/observability/pyproject.toml).

## Key Files

- Main entrypoint: [main.py](/home/mellow/Desktop/observability/observability/src/observability/main.py)
- Crew definition: [crew.py](/home/mellow/Desktop/observability/observability/src/observability/crew.py)
- Agent config: [agents.yaml](/home/mellow/Desktop/observability/observability/src/observability/config/agents.yaml)
- Task config: [tasks.yaml](/home/mellow/Desktop/observability/observability/src/observability/config/tasks.yaml)
- Deterministic pipeline: [pipeline.py](/home/mellow/Desktop/observability/observability/src/observability/pipeline.py)
- Deterministic tools and scoring logic: [custom_tool.py](/home/mellow/Desktop/observability/observability/src/observability/tools/custom_tool.py)
- Single case runner: [case_runner.py](/home/mellow/Desktop/observability/observability/src/observability/case_runner.py)
- Suite runner: [suite_runner.py](/home/mellow/Desktop/observability/observability/src/observability/suite_runner.py)

## Current Design Intent

This project is optimized around one core principle:

> deterministic diagnosis first, LLM explanation second

That keeps the system:

- cheaper
- easier to test
- easier to trust
- easier to debug when models or providers behave inconsistently
