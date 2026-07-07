# Agentic Decision Benchmark

A LangGraph-based research prototype for comparing three decision-making architectures on the same strategic business case:

- `single`
- `supervisor`
- `self_organizing`

The `self_organizing` mode is implemented as a Delphi-Blackboard Swarm with MCDA Consensus: fixed Delphi-style rounds, an append-only structured blackboard, peer critique, belief updates, scorecards, and deterministic MCDA aggregation.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

## Ollama Setup

```bash
ollama pull llama3.1:8b
ollama serve
```

Copy `.env.example` to `.env` and adjust settings if needed.

## Run With Mock Provider

The mock provider is deterministic and does not require Ollama.

```bash
python -m agentic_decision_benchmark.main run-all --provider mock
python -m agentic_decision_benchmark.main run-single --provider mock
python -m agentic_decision_benchmark.main run-supervisor --provider mock
python -m agentic_decision_benchmark.main run-self-organizing --provider mock
```

Optional resilience/adaptability flags:

```bash
python -m agentic_decision_benchmark.main run-all --provider mock --fault-injection
python -m agentic_decision_benchmark.main run-all --provider mock --new-info
```

## Run With Ollama

```bash
python -m agentic_decision_benchmark.main run-all --provider ollama
python -m agentic_decision_benchmark.main run-self-organizing --provider ollama
```

## Tests

```bash
python -m pytest
```

## Outputs

Runs are saved under:

```text
runs/YYYYMMDD_HHMMSS/
```

Mode folders use exactly:

```text
single/
supervisor/
self_organizing/
```

Each run writes JSON state, recommendations, metrics, evaluation artifacts, and a Markdown comparative report when possible.

## Provider Migration

Graph logic depends only on the `LLMProvider` interface:

```python
def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
    ...
```

To add a paid provider later, implement a new provider class beside `OllamaProvider`, add it to `llm/factory.py`, and configure:

```env
MODEL_PROVIDER=openai
OPENAI_MODEL=gpt-4.1
OPENAI_API_KEY=...
```

Agents, graphs, consensus, evaluator, and storage modules do not import Ollama-specific classes.

## Self-Organization Design

The `self_organizing` graph controls protocol timing but does not decide the strategy. The final recommendation emerges from:

1. independent expert analysis;
2. structured blackboard claims, risks, assumptions, opportunities, questions, critiques, and belief updates;
3. per-agent MCDA scorecards;
4. deterministic Python aggregation in `consensus/mcda.py`.

The consensus engine is not an LLM supervisor. It selects the scorecard winner by mean score and documented tie-breakers while preserving minority concerns and unresolved risks.

