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

## Information Setting And Private Role Knowledge

The benchmark uses an `equal_total` information setting. This matters because the project is trying to compare coordination mechanisms, not give one architecture more total information than another.

The public case in `data/eurotech_case.md` is supplemented by synthetic private role-specific briefs in `data/private_role_briefs.yaml`. These briefs simulate knowledge that different departments might hold inside EuroTech.

| Mode | Private information access |
| --- | --- |
| `single` | Receives all private role briefs as one consolidated information pack. |
| `supervisor` | Each specialist receives only its own private role brief. The supervisor sees specialist outputs, not the raw private briefs. |
| `self_organizing` | Each specialist receives only its own private role brief. Other agents learn private facts only if they are posted to the blackboard. |

The private briefs are explicitly synthetic because EuroTech is fictional. They include facts such as:

- Finance: EuroTech can safely allocate about EUR22M to EUR28M over 18 months, and losing the OEM puts about 30% of revenue at risk.
- Operations: only two of five production lines have enough sensor coverage for near-term predictive quality deployment, and wind retooling could reduce automotive capacity for 4 to 6 months.
- Human Resources: EuroTech lacks industrial data engineers and quality analytics specialists, and specialized hiring may take 6 to 9 months.
- Legal / Compliance: the OEM requirement is traceable and auditable predictive quality management, and premature wind commitments could create penalty exposure.
- Strategy: wind offers diversification, but demand is not backed by firm long-term purchase commitments; early AI quality capability could improve OEM supplier status.
- Technology: a minimum viable predictive quality system may be possible on pilot lines in 9 to 12 months, but full plant-wide deployment inside 18 months is risky.

## The Three Decision-Making Modes

### 1. Single Mode

This is the simplest mode.

One generalist AI assistant reads the case and the four strategy options. It then produces a final recommendation, reasoning, risks, assumptions, and confidence.

In the implementation, this mode is defined in `src/agentic_decision_benchmark/graphs/single_graph.py`. The generalist prompt receives the public scenario, the four strategies, the consolidated private brief pack, the optional faulty claim when fault injection is enabled, and the optional new-information update when adaptability testing is enabled. The graph then runs the same evaluator node used by the other modes.

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

In the implementation, this mode is defined in `src/agentic_decision_benchmark/graphs/supervisor_graph.py`. The flow is supervisor plan, six isolated domain analyses, supervisor synthesis, then evaluator scoring. The domain agents do not see each other's private briefs or outputs directly, and they do not critique each other in this mode. When new-information testing is enabled, the supervisor receives the update during final synthesis rather than rerunning the isolated expert calls.

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

The same six specialist assistants participate, but there is no AI supervisor deciding the answer. Instead, the system runs four macro-rounds with an adaptive critique/update loop:

1. Each specialist gives an independent first opinion.
2. The blackboard receives deterministic salience scores.
3. Specialists select up to three critique targets locally from high-salience blackboard items.
4. Optional new information can be injected once onto the blackboard during the first critique/update cycle.
5. A deterministic conflict map is built from structured blackboard artifacts.
6. Each specialist updates its view after reading the critiques, visible blackboard content, and conflict map.
7. The system checks convergence.
8. If convergence is not reached and `max_deliberation_cycles` is not exhausted, steps 2 through 7 repeat.
9. Each specialist scores all four strategies against shared criteria.
10. A deterministic scoring engine combines the scorecards and selects the result.

In the implementation, this mode is defined in `src/agentic_decision_benchmark/graphs/self_organizing_graph.py`. The graph controls round order, but the final strategy is selected by deterministic MCDA consensus in `src/agentic_decision_benchmark/consensus/mcda.py`.

Plain-language analogy: six experts independently study the problem, use a shared issue tracker to decide what deserves challenge, revise their views, fill out score sheets, and then the final answer is calculated from the score sheets.

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
- choose critique targets from blackboard salience signals
- critique other agents
- update their beliefs using the deterministic conflict map
- score the options
- preserve minority concerns

The system controls the order of events, but it does not let a supervisor AI choose the final answer. The final self-organizing recommendation is selected by a normal Python scoring algorithm.

The `self_organizing` mode has four macro-rounds, but Round 2 critique and Round 3 belief update may repeat through a bounded adaptive deliberation loop. The default limit is `max_deliberation_cycles: 2` in `config/benchmark.yaml`.

The loop shape is:

```text
Round 1 independent analysis
-> blackboard salience update
-> [Round 2 agent-selected critique -> deterministic conflict map -> Round 3 belief update -> convergence check] loop
-> Round 4 MCDA scorecards
-> deterministic consensus
```

Conflicts are not manually assigned. They are also not freely generated by a hidden LLM analyst. They are deterministically derived from structured blackboard artifacts:

- `direct_critique`: one or more critique items target the same blackboard item.
- `opposing_stances`: agents support and oppose the same strategy, criterion, and topic.
- `risk_cluster`: multiple agents post high-severity risks under overlapping topic tags.

Agents must select topic tags from the controlled vocabulary. Invalid tags are discarded, and if no valid tag remains the system assigns `uncategorized`.

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

Each blackboard item has structured fields such as a stable ID, macro-round, deliberation cycle, author, item type, content, topic tags, related strategy, criterion, stance, confidence, severity, and optional target item ID for critiques.

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

If a tie still remains after those rules, the consensus result can be marked unresolved and the remaining tied strategies are recorded.

This means the final self-organizing result is not decided by a hidden AI judge. It is calculated from visible scorecards.

## Fault Injection And New Information

The project includes two optional stress tests.

Both are switchable in `config/benchmark.yaml`:

```yaml
fault_injection_enabled: false
new_info_injection_enabled: false
```

The CLI can override those defaults for a single run with `--fault-injection`, `--no-fault-injection`, `--new-info`, and `--no-new-info`.

### Fault Injection

Fault injection gives the system an intentionally overconfident or unsupported claim:

"Full AI-enabled predictive quality management can be deployed across all plants in 3 months with minimal cost and minimal integration risk."

The goal is to see whether the agents challenge that claim instead of blindly accepting it.

This tests resilience.

Implementation detail: in `single` mode the claim is included in the generalist prompt. In `supervisor` and `self_organizing`, it is injected through the Technology Agent path, because the faulty claim is about AI quality deployment feasibility.

### New Information

New information injection tells the system:

"The automotive OEM extends the compliance deadline from 18 months to 24 months."

The goal is to see whether the agents can incorporate the new timing information without restarting the whole process.

This tests adaptability.

Implementation detail: this option is switchable for all three modes. In `single`, the update is included in the generalist prompt. In `supervisor`, the update is provided during final synthesis after isolated expert work. In `self_organizing`, the graph injects it as a `new_information` blackboard item once during the first critique/update cycle, before conflict mapping and belief update. The belief-update prompt tells agents to update from previous blackboard content and the current conflict map without restarting.

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

- private role briefs used for the run
- full state
- final recommendation
- evaluation result
- metrics
- blackboard, where relevant
- scorecards, for self-organizing mode
- salience map, conflict map, and convergence status, for self-organizing mode
- a comparative Markdown report

The supervisor folder also gets a `blackboard.json` file for consistency with the storage layout, but supervisor mode normally does not populate blackboard entries. The meaningful deliberation blackboard is produced by `self_organizing`.

The comparative report summarizes:

- the scenario
- the four strategy options
- the three architectures
- final recommendations by mode
- model calls and estimated tokens
- evaluation scores
- blackboard item counts
- salience summaries
- conflict map summaries
- convergence cycle summaries
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

The evaluator uses decimal scores from `1.0` to `5.0`. Decimal scores reduce evaluator flattening and allow finer distinctions between architectures that choose the same final strategy but differ in reasoning quality, auditability, resilience, or cost.

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
- deliberation cycles used
- maximum deliberation cycle limit
- whether convergence happened before the cycle limit
- conflict counts by type
- unresolved high-severity conflict count
- agreement ratio after each deliberation cycle
- low-confidence agents after each deliberation cycle
- average salience and top salience items

Token counts are approximate whitespace-based estimates from the provider wrapper. They are useful for relative comparison inside this benchmark, not exact billing.

With the deterministic mock provider, a typical `run-all` execution uses:

| Mode | Model calls | Why |
| --- | --- | --- |
| `single` | 2 | one generalist call plus one evaluator call |
| `supervisor` | 9 | one supervisor plan, six domain analyses, one supervisor synthesis, one evaluator call |
| `self_organizing` | Up to 37 by default with the mock provider | six calls in Round 1, six calls for each critique/update cycle, six Round 4 scorecard calls, plus one evaluator call; fewer calls are possible if convergence happens before `max_deliberation_cycles` |

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
| `data/private_role_briefs.yaml` | Synthetic private role-specific knowledge. |
| `src/.../main.py` | Command-line entry point. |
| `src/.../settings.py` | Configuration, environment, scenario, strategy, and private-brief loading. |
| `src/.../prompts.py` | Prompt builders and strict JSON instructions. |
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
9. The evaluator scores the result inside the graph.
10. The system calculates deterministic metrics.
11. The run store writes JSON files.
12. The report writer creates a comparative Markdown report.

The CLI supports these commands:

```bash
python -m agentic_decision_benchmark.main run-single
python -m agentic_decision_benchmark.main run-supervisor
python -m agentic_decision_benchmark.main run-self-organizing
python -m agentic_decision_benchmark.main run-all
```

Common options include:

```bash
--provider mock
--provider ollama
--model llama3.1:8b
--temperature 0.2
--max-tokens 1200
--output-dir runs
--fault-injection
--no-fault-injection
--new-info
--no-new-info
```

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
- The blackboard uses controlled stance values and controlled topic tags.
- Invalid topic tags are discarded, with `uncategorized` as fallback.
- Conflict maps are derived by deterministic Python rules, not by an LLM analyst.
- The agent registry requires exactly six domain agents.
- Prompt visibility tests check that private briefs are not leaked across roles.

These guardrails make the benchmark easier to test and compare.

## What The Tests Cover

The tests check that:

- blackboard items validate correctly
- blackboard topic tags normalize correctly
- initial state is created correctly
- valid modes are accepted and invalid modes are rejected
- blackboard entries append instead of replacing old entries
- salience maps are deterministic and respond to severity and critique counts
- deterministic conflict maps create direct critique, opposing stance, and risk cluster conflicts
- convergence checks respond to agreement, severe conflicts, and max cycle routing
- the provider factory can create mock and Ollama providers
- the mock provider tracks calls
- the evaluator returns valid scores
- each graph can run with the mock provider
- the self-organizing graph creates blackboard items, salience maps, conflict maps, convergence status, scorecards, critiques, and consensus
- MCDA selects the correct top strategy
- MCDA tie-breaking behaves as designed
- minority concerns are preserved
- private role briefs load with all expected roles and fields
- single mode receives all private brief sections
- supervisor and self-organizing role prompts receive only the intended role brief plus visible blackboard content

## Current Limitations

Important limitations of the current repository:

- The case and private role briefs are synthetic.
- The mock provider is deterministic and useful for tests, not for measuring real model reasoning quality.
- Ollama output quality depends on the local model and whether it follows the strict JSON prompts.
- The evaluator is model-assisted, so its scores are useful but not external ground truth.
- Token counts are approximate whitespace estimates.
- OpenAI provider support is intentionally not implemented yet.
- The self-organizing process is structured by the graph; it is not autonomous open-ended negotiation.
- Strategy outcomes are not validated against real financial, operational, legal, or market data.

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
