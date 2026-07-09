# Agentic Decision Benchmark

A LangGraph-based research prototype for comparing three decision-making architectures on the same strategic business case:

- `single`
- `supervisor`
- `self_organizing`

The `self_organizing` mode is implemented as a Delphi-Blackboard deliberation loop with MCDA Consensus: four macro-rounds, deterministic blackboard salience, locally selected peer critique, deterministic conflict mapping, belief updates, scorecards, and deterministic MCDA aggregation.

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

Copy `.env.example` to `.env` and adjust settings if needed. Runtime config precedence is:

```text
CLI flags -> environment / .env -> config/model.yaml defaults
```

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
python -m agentic_decision_benchmark.main run-all --provider mock --fault-injection --new-info
```

Both stress tests are off by default in `config/benchmark.yaml`:

```yaml
fault_injection_enabled: false
new_info_injection_enabled: false
```

The CLI flags override those config defaults for a single run. Use `--fault-injection` / `--new-info` to turn them on, or `--no-fault-injection` / `--no-new-info` to force them off.

Both options apply to all three modes. Fault injection tests whether a mode catches an unsupported AI-rollout claim. New information tests whether a mode incorporates the updated 24-month OEM deadline at its architecture-specific decision point.

## Run With Ollama

```bash
python -m agentic_decision_benchmark.main run-all --provider ollama
python -m agentic_decision_benchmark.main run-self-organizing --provider ollama
```

## Run With OpenAI

Set an API key in `.env`:

```env
MODEL_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.4-mini
```

Then run any benchmark mode with the OpenAI-backed provider:

```bash
python -m agentic_decision_benchmark.main run-all --provider openai
python -m agentic_decision_benchmark.main run-self-organizing --provider openai
```

The OpenAI provider calls the Responses API and requests JSON-object output. `OPENAI_STORE=false` is the default for benchmark runs. Override `OPENAI_MODEL` or pass `--model` to compare another model:

```bash
python -m agentic_decision_benchmark.main run-all --provider openai --model gpt-5.5
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

For `self_organizing`, the run folder also includes:

```text
salience_map.json
conflict_map.json
convergence.json
scorecards.json
```

## Evaluation Scores

The evaluator uses decimal scores from `1.0` to `5.0` for decision quality, convergence, resilience, explainability, adaptability, cost efficiency, runtime efficiency, and overall score.

Decimal scores reduce evaluator flattening and allow finer distinctions between architectures that choose the same final strategy but differ in reasoning quality, auditability, resilience, or cost.

## Private Role-Specific Knowledge

The benchmark uses synthetic private role-specific briefs because the EuroTech case is fictional but the challenge requires specialized agents with partial knowledge.

The default information setting is `equal_total`.

In `equal_total`:
- `single` receives all private briefs as one consolidated information pack.
- `supervisor` distributes private briefs by role to isolated domain agents.
- `self_organizing` distributes private briefs by role, and agents share information only through the blackboard.

This isolates the effect of coordination mechanism rather than giving one mode more total information than another.

The private brief file is located at:

```text
data/private_role_briefs.yaml
```

## Provider Backends

Graph logic depends only on the `LLMProvider` interface:

```python
def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
    ...
```

Available providers are `mock`, `ollama`, and `openai`. Configure OpenAI with:

```env
MODEL_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.4-mini
```

Agents, graphs, consensus, evaluator, and storage modules do not import provider-specific classes.

## Self-Organization Design

The `self_organizing` graph controls protocol timing but does not decide the strategy. The final recommendation emerges from:

1. independent expert analysis;
2. deterministic blackboard salience updates;
3. locally selected critiques and deterministic conflict maps;
4. belief updates over the blackboard and current conflicts;
5. per-agent MCDA scorecards;
6. deterministic Python aggregation in `consensus/mcda.py`.

The mode has four macro-rounds:

```text
Round 1 independent analysis
-> blackboard salience update
-> [Round 2 agent-selected critique -> deterministic conflict map -> Round 3 belief update -> convergence check] loop
-> Round 4 MCDA scorecards
-> deterministic consensus
```

Round 2 and Round 3 may repeat through the adaptive deliberation loop. The loop is bounded by `max_deliberation_cycles` in `config/benchmark.yaml`; the default is `2`.

Conflicts are not manually assigned and are not freely generated by a hidden LLM analyst. They are deterministically derived from structured blackboard artifacts:

- `direct_critique`: critiques targeting the same blackboard item.
- `opposing_stances`: support and opposition under the same strategy, criterion, and topic.
- `risk_cluster`: high-severity risks from multiple agents under overlapping topic tags.

Agents must choose topic tags from the controlled vocabulary in `schemas.py`. Invalid tags are discarded; if no valid tags remain, the item is assigned `uncategorized`.

The consensus engine is not an LLM supervisor. It selects the scorecard winner by mean score and documented tie-breakers while preserving minority concerns and unresolved risks.

