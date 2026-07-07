# Agentic Decision Benchmark, Explained Simply

## Short Summary

This project is a research prototype that compares three different ways of using AI assistants to make a difficult business decision.

The business case is fictional. A company called EuroTech Manufacturing must decide how to respond when a major automotive customer threatens to end a contract unless EuroTech proves it can use AI-based predictive quality management. At the same time, EuroTech has a possible new opportunity with a wind turbine manufacturer.

The system does not actually run a factory, connect to SAP or another ERP system, or make a real business decision. It is a benchmark: it runs the same business problem through three AI decision-making setups, saves the results, and compares how well each setup performed.

## The Business Problem

EuroTech Manufacturing GmbH is described as a mid-size German manufacturer with:

- 1,200 employees
- EUR180M in revenue
- customers in automotive and renewable energy
- one major automotive customer representing 30% of revenue

The automotive customer requires EuroTech to demonstrate full AI-enabled predictive quality management within 18 months. If EuroTech cannot do that, it may lose the contract.

At the same time, a wind turbine manufacturer offers a new partnership that could replace the at-risk automotive revenue, but it would require major factory changes and hiring.

So EuroTech must decide how to balance:

- protecting current revenue
- avoiding overdependence on one automotive customer
- investing in AI quality systems
- retooling operations for wind components
- hiring and retraining workers
- staying compliant with contracts, safety, and governance
- avoiding too much financial risk

## The Four Choices Being Compared

The system gives every decision-making setup the same four possible strategies.

| ID | Strategy | Plain-language meaning |
| --- | --- | --- |
| A | OEM-first AI compliance | Focus mainly on satisfying the automotive customer quickly. |
| B | Wind pivot | Shift aggressively toward the wind turbine opportunity. |
| C | Dual-track staged strategy | Do both in a controlled, phased way: protect the automotive customer while piloting the wind opportunity. |
| D | Defensive minimal investment | Spend as little as possible and wait for more certainty. |

In the included mock runs and tests, the system usually selects Strategy C, the staged dual-track approach.

## What The Project Is Testing

The project asks a simple research question:

Which AI decision structure works better for a complex business decision?

It compares three structures:

1. A single generalist AI assistant
2. A supervisor-led group of specialist AI assistants
3. A self-organizing group of specialist AI assistants that critique each other and vote through structured scoring

The purpose is not only to see which answer they choose. The project also measures whether the process is explainable, resilient, adaptable, and efficient.

## The Three Decision-Making Modes

### 1. Single Mode

This is the simplest mode.

One generalist AI assistant reads the case and the four strategy options. It then produces a final recommendation, reasoning, risks, assumptions, and confidence.

Plain-language analogy: one consultant writes a recommendation alone.

Strengths:

- Fast
- Cheap in model calls
- Easy to understand

Weaknesses:

- No peer challenge
- Easier for one mistaken assumption to survive
- Less transparent than a full expert discussion

### 2. Supervisor Mode

This mode has one central supervisor and six specialist assistants.

The supervisor first creates a plan. Then each specialist gives an isolated analysis from its own area. Finally, the supervisor reads those specialist outputs and writes the final recommendation.

The six specialist roles are:

- Finance
- Operations
- Human Resources
- Legal / Compliance
- Strategy
- Technology

Plain-language analogy: a project lead asks six department experts for input, then writes the final decision.

Strengths:

- Broader perspective than the single mode
- Brings in several business functions
- Still has one clear final decision-maker

Weaknesses:

- The supervisor has a lot of power
- Specialists do not directly challenge each other
- The final answer depends heavily on the supervisor's synthesis

### 3. Self-Organizing Mode

This is the main experimental mode.

The same six specialist assistants participate, but there is no AI supervisor deciding the answer. Instead, the system runs a fixed decision process:

1. Each specialist gives an independent first opinion.
2. The specialists read the shared notes and critique each other.
3. Each specialist updates its view after reading the critiques.
4. Each specialist scores all four strategies against shared criteria.
5. A deterministic scoring engine combines the scorecards and selects the result.

Plain-language analogy: six experts independently study the problem, challenge each other's assumptions, revise their views, fill out score sheets, and then the final answer is calculated from the score sheets.

Strengths:

- More transparent
- Better at exposing disagreement
- Better at catching unsupported claims
- Keeps a detailed record of the discussion

Weaknesses:

- Uses more model calls
- Takes more steps
- Produces more artifacts to inspect
- More complex than a supervisor-led process

## What "Self-Organizing" Means Here

In this project, "self-organizing" does not mean the AI agents freely chat without rules.

It means the agents are given a structured process where they can:

- make independent claims
- raise risks and assumptions
- critique other agents
- update their beliefs
- score the options
- preserve minority concerns

The system controls the order of events, but it does not let a supervisor AI choose the final answer. The final self-organizing recommendation is selected by a normal Python scoring algorithm.

This is important because the final choice is auditable. You can inspect the scorecards and see why one strategy won.

## The Shared Blackboard

The self-organizing mode uses a "blackboard."

This is not a visual board. It is a structured shared log where the system stores everything the agents contribute.

The blackboard can contain:

- claims
- risks
- opportunities
- assumptions
- questions
- critiques
- belief updates
- scorecard summaries
- minority concerns
- new information
- final consensus notes

Plain-language analogy: a shared meeting record where every expert's comments are saved instead of overwritten.

The blackboard is append-only. New items are added, but old items are not erased. That makes the decision trail easier to audit.

## How The Final Self-Organizing Decision Is Chosen

The self-organizing mode uses MCDA, which means multi-criteria decision analysis.

In plain language, each expert scores each strategy across several criteria, and the system averages the scores.

The criteria are:

- financial viability
- operational feasibility
- strategic value
- workforce feasibility
- legal, compliance, and safety

Each specialist scores Strategy A, B, C, and D from 1 to 5 on those criteria.

The consensus engine then:

- calculates average scores
- counts which strategy each agent top-scored
- measures agreement
- preserves minority concerns
- records unresolved risks
- applies deterministic tie-breakers if needed

Tie-breakers favor:

1. legal, compliance, and safety
2. financial viability
3. operational feasibility
4. lower score disagreement

This means the final self-organizing result is not decided by a hidden AI judge. It is calculated from visible scorecards.

## Fault Injection And New Information

The project includes two optional stress tests.

### Fault Injection

Fault injection gives the system an intentionally overconfident or unsupported claim:

"Full AI-enabled predictive quality management can be deployed across all plants in 3 months with minimal cost and minimal integration risk."

The goal is to see whether the agents challenge that claim instead of blindly accepting it.

This tests resilience.

### New Information

New information injection tells the system:

"The automotive OEM extends the compliance deadline from 18 months to 24 months."

The goal is to see whether the agents can incorporate the new timing information without restarting the whole process.

This tests adaptability.

## The AI Model Providers

The system can use two provider modes.

### Mock Provider

The mock provider is deterministic. It returns predictable, schema-valid answers without calling a real language model.

It is used for:

- tests
- repeatable demo runs
- development without Ollama

### Ollama Provider

The Ollama provider calls a local Ollama server, usually with the `llama3.1:8b` model.

It is used when you want real model-generated answers while still running locally.

The code is designed so another provider could be added later behind the same provider interface, but OpenAI support is not implemented in this repository.

## What Gets Saved After A Run

Every run is saved under the `runs` folder with a timestamped directory name such as:

```text
runs/YYYYMMDD_HHMMSS/
```

Inside that folder, each mode gets its own folder:

```text
single/
supervisor/
self_organizing/
```

The saved artifacts include:

- full state
- final recommendation
- evaluation result
- metrics
- blackboard, where relevant
- scorecards, for self-organizing mode
- a comparative Markdown report

The comparative report summarizes:

- the scenario
- the four strategy options
- the three architectures
- final recommendations by mode
- model calls and estimated tokens
- evaluation scores
- blackboard item counts
- critique summaries
- belief updates
- MCDA score summaries
- strengths and weaknesses

## How The System Evaluates Results

The system evaluates each final recommendation using both measured metrics and an evaluator prompt.

The evaluation categories are:

- decision quality
- convergence
- resilience
- explainability
- adaptability
- cost efficiency
- runtime efficiency
- overall score

The measured metrics include:

- runtime
- number of model calls
- estimated input and output tokens
- number of rounds
- number of blackboard items
- number of claims
- number of critiques
- number of belief updates
- number of scorecards
- agreement ratio
- score variance
- selected final strategy
- whether fault correction was detected
- whether new information was incorporated

## How The Project Is Organized

| Area | Purpose |
| --- | --- |
| `README.md` | Setup and command examples. |
| `pyproject.toml` | Python package settings and dependencies. |
| `config/agents.yaml` | The six specialist agent definitions. |
| `config/benchmark.yaml` | Benchmark modes, criteria, fault claim, and new information. |
| `config/evaluator_rubric.yaml` | Evaluation categories and scoring scale. |
| `config/model.yaml` | Default model provider settings. |
| `data/eurotech_case.md` | The fictional business case. |
| `data/candidate_strategies.yaml` | The four strategy options. |
| `src/.../main.py` | Command-line entry point. |
| `src/.../graphs/` | The three decision workflows. |
| `src/.../agents/` | Agent loading and agent role objects. |
| `src/.../llm/` | Mock and Ollama model providers. |
| `src/.../consensus/` | MCDA consensus calculation. |
| `src/.../evaluation/` | Metrics and evaluator logic. |
| `src/.../storage/` | Run artifact and report writing. |
| `src/.../schemas.py` | Data validation models. |
| `src/.../state.py` | Shared run state structure. |
| `tests/` | Unit and smoke tests. |
| `runs/` | Saved benchmark outputs. |

## How A Run Works From Start To Finish

1. The command-line program loads settings.
2. It loads the EuroTech scenario.
3. It loads the four candidate strategies.
4. It chooses one mode or all three modes.
5. It creates the selected AI provider.
6. It builds the LangGraph workflow for that mode.
7. It creates the initial run state.
8. The graph executes the decision process.
9. The system calculates metrics.
10. The evaluator scores the result.
11. The run store writes JSON files.
12. The report writer creates a comparative Markdown report.

## The Role Of LangGraph

LangGraph is used to define and execute the workflow steps.

In this project, LangGraph is the process controller. It decides which node runs after which node.

It does not decide the business strategy by itself.

In single mode, the generalist AI decides.

In supervisor mode, the supervisor AI decides after reading specialist input.

In self-organizing mode, the agents produce structured evidence and scorecards, then the Python consensus engine calculates the result.

## Data Quality And Guardrails

The project uses several guardrails to keep outputs structured:

- Prompts ask for JSON only.
- Pydantic schemas validate model responses.
- Strategy IDs are restricted to A, B, C, and D.
- Confidence scores must stay between 0 and 1.
- Scorecard values must stay between 1 and 5.
- The self-organizing scorecard must include all four strategies.
- The blackboard uses controlled item types.

These guardrails make the benchmark easier to test and compare.

## What The Tests Cover

The tests check that:

- blackboard items validate correctly
- initial state is created correctly
- valid modes are accepted and invalid modes are rejected
- blackboard entries append instead of replacing old entries
- the provider factory can create mock and Ollama providers
- the mock provider tracks calls
- the evaluator returns valid scores
- each graph can run with the mock provider
- the self-organizing graph creates blackboard items, scorecards, critiques, and consensus
- MCDA selects the correct top strategy
- MCDA tie-breaking behaves as designed
- minority concerns are preserved

## What This System Is Good For

This project is useful for studying:

- how different AI decision structures behave
- whether structured agent debate improves auditability
- whether peer critique catches weak claims
- how much extra cost a self-organizing process creates
- whether a decision process can preserve disagreement instead of hiding it
- how to compare AI workflows using the same case and metrics

## What This System Is Not

This project is not:

- a production SAP or ERP application
- a real EuroTech decision system
- a live factory planning tool
- a financial forecasting system
- a legal compliance engine
- a replacement for human executives or domain experts

It is a controlled prototype for comparing AI orchestration patterns.

## Main Takeaway

The core idea is straightforward:

A complex business decision is given to three AI decision-making structures. The system records how each structure reasons, what it recommends, how much it costs to run, and how explainable the result is.

The self-organizing mode is the most distinctive part. It does not rely on one AI supervisor to make the final call. Instead, it asks specialists to independently reason, critique each other, update their beliefs, score the options, and let a transparent scoring process select the recommendation.

That makes the system slower and more complex than a single-agent answer, but it gives a clearer audit trail for decisions where disagreement, risk, and accountability matter.
