from __future__ import annotations

import json
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from agentic_decision_benchmark.llm.base import BaseLLMProvider


class MockProvider(BaseLLMProvider):
    """Deterministic provider returning schema-valid JSON for tests and smoke runs."""

    def _generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        task = self._marker(prompt, "TASK")
        agent = self._marker(prompt, "AGENT_NAME")
        handlers = {
            "GENERALIST_RECOMMENDATION": self._generalist,
            "SUPERVISOR_PLAN": self._supervisor_plan,
            "ISOLATED_DOMAIN_ANALYSIS": lambda: self._domain_analysis(agent, prompt),
            "SUPERVISOR_SYNTHESIS": self._supervisor_synthesis,
            "SELF_ORGANIZING_ROUND_1": lambda: self._round1(agent),
            "SELF_ORGANIZING_ROUND_2": lambda: self._round2(agent, prompt),
            "SELF_ORGANIZING_ROUND_3": lambda: self._round3(agent, prompt),
            "SELF_ORGANIZING_ROUND_4": lambda: self._round4(agent),
            "EVALUATOR": lambda: self._evaluator(prompt),
        }
        if task not in handlers:
            raise ValueError(f"MockProvider received unknown task marker: {task!r}")
        return json.dumps(handlers[task](), sort_keys=True)

    @staticmethod
    def _marker(prompt: str, name: str) -> str:
        match = re.search(rf"^{re.escape(name)}:\s*(.+)$", prompt, flags=re.MULTILINE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _agent_domain(agent: str) -> str:
        return {
            "Finance Agent": "cash flow and staged investment",
            "Operations Agent": "production disruption and retooling sequence",
            "HR Agent": "skills, hiring, and retraining capacity",
            "Legal / Compliance Agent": "contract, auditability, and data governance",
            "Strategy Agent": "customer concentration and diversification timing",
            "Technology Agent": "sensor coverage, pipelines, and model monitoring",
        }.get(agent, "strategic tradeoffs")

    @staticmethod
    def _json_section(prompt: str, marker: str) -> dict[str, Any]:
        if marker not in prompt:
            return {}
        text = prompt.split(f"{marker}: ", 1)[1]
        for next_marker in [
            "\nSALIENCE_RANKED_ITEMS:",
            "\nCONFLICT_HISTORY:",
            "\nCURRENT_CONFLICT_MAP:",
            "\nCRITIQUES_TARGETING_OWN_CLAIMS:",
            "\nUNRESOLVED_HIGH_SEVERITY_CONFLICTS:",
            "\nDETERMINISTIC_METRICS:",
        ]:
            if next_marker in text:
                text = text.split(next_marker, 1)[0]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _trailing_json_section(prompt: str, marker: str) -> dict[str, Any]:
        if marker not in prompt:
            return {}
        text = prompt.split(f"{marker}: ", 1)[1].strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _score(value: float) -> float:
        bounded = min(5.0, max(1.0, value))
        return float(Decimal(str(bounded)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

    def _find_blackboard_item(
        self,
        prompt: str,
        *,
        author: str | None = None,
        item_type: str | None = None,
        content_contains: str | None = None,
    ) -> dict[str, Any]:
        payload = self._json_section(prompt, "VISIBLE_BLACKBOARD")
        for item in payload.get("blackboard", []):
            if author is not None and item.get("author") != author:
                continue
            if item_type is not None and item.get("item_type") != item_type:
                continue
            if content_contains is not None and content_contains not in item.get("content", ""):
                continue
            return item
        return {}

    def _generalist(self) -> dict[str, Any]:
        return {
            "recommended_strategy": "C",
            "recommendation": "Adopt a dual-track staged strategy: protect the OEM relationship through phased AI quality compliance while piloting the wind partnership with gated investment.",
            "reasoning": [
                "The OEM revenue exposure is too large to ignore.",
                "A full wind pivot creates execution and cash-flow risk.",
                "A staged dual-track approach preserves optionality while reducing concentration risk.",
            ],
            "risks": [
                "Coordination complexity across AI compliance and wind retooling.",
                "Cash constraints if both tracks accelerate at the same time.",
                "Supplier and workforce readiness may become bottlenecks.",
            ],
            "assumptions": [
                "EuroTech can sequence investment with clear gates.",
                "The wind partner accepts a pilot before full retooling.",
            ],
            "confidence": 0.72,
        }

    def _supervisor_plan(self) -> dict[str, Any]:
        return {
            "plan": [
                "Collect independent analyses from finance, operations, HR, legal/compliance, strategy, and technology.",
                "Compare candidate strategies against shared feasibility and risk criteria.",
                "Synthesize a central recommendation with explicit tradeoffs.",
            ],
            "domains_to_consult": [
                "Finance",
                "Operations",
                "Human Resources",
                "Legal / Compliance",
                "Strategy",
                "Technology",
            ],
            "decision_criteria": [
                "financial viability",
                "operational feasibility",
                "strategic value",
                "workforce feasibility",
                "legal / compliance safety",
            ],
            "confidence": 0.82,
        }

    def _domain_analysis(self, agent: str, prompt: str) -> dict[str, Any]:
        domain = self._agent_domain(agent)
        recommendation = "C"
        if agent == "Technology Agent":
            recommendation = "A"
        if agent == "Strategy Agent":
            recommendation = "C"
        claims = [
            f"From the {domain} perspective, Strategy C keeps the most decision flexibility.",
            "A phased approach can protect the OEM path while testing wind-market readiness.",
        ]
        if "Full AI-enabled predictive quality management can be deployed across all plants in 3 months" in prompt:
            claims.append(
                "Full AI-enabled predictive quality management can be deployed across all plants in 3 months with minimal cost and minimal integration risk."
            )
        return {
            "claims": claims,
            "risks": [
                f"{agent} sees execution risk if investment gates are not explicit.",
                "Compressed timelines may mask integration, supplier, or workforce bottlenecks.",
            ],
            "opportunities": [
                "Use staged milestones to preserve the OEM relationship and negotiate wind optionality.",
            ],
            "assumptions": [
                "Management can prioritize a small number of measurable milestones.",
                "Capital allocation can be gated by evidence from pilots.",
            ],
            "recommendation": recommendation,
            "confidence": 0.7,
        }

    def _supervisor_synthesis(self) -> dict[str, Any]:
        return {
            "recommended_strategy": "C",
            "recommendation": "Choose Strategy C with explicit gates: urgent OEM AI compliance foundation, bounded wind pilot, and staged release of CAPEX and hiring.",
            "synthesis": [
                "Finance and operations favor staged investment over a large irreversible pivot.",
                "Technology and compliance require auditability, data readiness, and model monitoring.",
                "Strategy benefits from diversification without abandoning the at-risk OEM relationship.",
            ],
            "tradeoffs": [
                "Strategy C is harder to coordinate than A or D.",
                "It delays full wind commitment compared with B.",
            ],
            "risks": [
                "Dual execution overload.",
                "Insufficient AI data readiness.",
                "Hiring bottlenecks for quality analytics and wind production skills.",
            ],
            "assumptions": [
                "The OEM accepts phased evidence of predictive quality management.",
                "The wind partner accepts a pilot-gated ramp.",
            ],
            "confidence": 0.78,
        }

    def _round1(self, agent: str) -> dict[str, Any]:
        domain = self._agent_domain(agent)
        recommendation = "C"
        if agent == "Technology Agent":
            recommendation = "A"
        structured_items = [
            {
                "item_type": "claim",
                "content": f"{agent} sees {domain} as a binding constraint on any strategy.",
                "topic_tags": ["strategic_value"],
                "related_strategy": recommendation,
                "criterion": "strategic_value",
                "stance": "supports",
                "confidence": 0.68,
                "severity": 3,
            },
            {
                "item_type": "claim",
                "content": "Strategy C provides the best balance of near-term OEM protection and diversification learning.",
                "topic_tags": ["strategic_value", "diversification"],
                "related_strategy": "C",
                "criterion": "strategic_value",
                "stance": "supports",
                "confidence": 0.68,
                "severity": 3,
            },
            {
                "item_type": "risk",
                "content": f"{agent} flags that Strategy B may overextend investment capacity and execution bandwidth.",
                "topic_tags": ["investment_capacity", "cash_flow"],
                "related_strategy": "B",
                "criterion": "financial_viability",
                "stance": "opposes",
                "confidence": 0.72,
                "severity": 4 if agent in {"Finance Agent", "Operations Agent"} else 3,
            },
        ]
        if agent == "Technology Agent":
            structured_items.append(
                {
                    "item_type": "claim",
                    "content": "Pilot-line AI predictive quality deployment is feasible in 9 to 12 months if data readiness work starts immediately.",
                    "topic_tags": ["ai_feasibility", "data_readiness", "implementation_timeline"],
                    "related_strategy": "A",
                    "criterion": "operational_feasibility",
                    "stance": "supports",
                    "confidence": 0.78,
                    "severity": 4,
                }
            )
        if agent == "Operations Agent":
            structured_items.append(
                {
                    "item_type": "risk",
                    "content": "Plant-wide AI rollout could disrupt production if retooling and data work are compressed.",
                    "topic_tags": ["ai_feasibility", "production_disruption", "implementation_timeline"],
                    "related_strategy": "A",
                    "criterion": "operational_feasibility",
                    "stance": "opposes",
                    "confidence": 0.8,
                    "severity": 4,
                }
            )
        return {
            "claims": [
                f"{agent} sees {domain} as a binding constraint on any strategy.",
                "Strategy C provides the best balance of near-term OEM protection and diversification learning.",
            ],
            "risks": [
                f"{agent} flags that Strategy B may overextend the organization.",
                "Strategy D risks both OEM loss and missed wind timing.",
            ],
            "opportunities": [
                "Use gated pilots to learn before full commitment.",
                "Convert compliance work into reusable quality capability.",
            ],
            "assumptions": [
                "EuroTech can run disciplined investment gates.",
                "The OEM and wind partner will accept staged proof points.",
            ],
            "questions_for_others": [
                f"What evidence would make {agent} change its recommendation?",
            ],
            "initial_recommendation": recommendation,
            "blackboard_items": structured_items,
            "confidence": 0.68,
        }

    def _round2(self, agent: str, prompt: str) -> dict[str, Any]:
        ai_target = self._find_blackboard_item(
            prompt,
            author="Technology Agent",
            item_type="claim",
            content_contains="Pilot-line AI predictive quality deployment",
        )
        faulty = "3 months with minimal cost and minimal integration risk" in prompt
        if faulty and agent != "Technology Agent":
            faulty_target = self._find_blackboard_item(
                prompt,
                author="Technology Agent",
                item_type="claim",
                content_contains="3 months with minimal cost and minimal integration risk",
            )
            selected_target_id = faulty_target.get("id") or ai_target.get("id") or ""
            return {
                "agent_name": agent,
                "selected_items": [selected_target_id],
                "critiques": [
                    {
                        "target_item_id": selected_target_id or None,
                        "target_agent": "Technology Agent",
                        "target_claim": "Full AI-enabled predictive quality management can be deployed across all plants in 3 months with minimal cost and minimal integration risk.",
                        "critique": f"{agent} rejects this as overconfident because plant-wide predictive quality needs data readiness, integration work, governance, and staged validation.",
                        "missing_assumption": "No evidence is provided for sensor coverage, historical data quality, integration capacity, or validation resources.",
                        "risk_introduced": "Underestimating implementation scope could cause missed OEM milestones and wasted CAPEX.",
                        "topic_tags": ["ai_feasibility", "data_readiness", "implementation_timeline"],
                        "severity": 5,
                        "confidence": 0.88,
                        "related_strategy": "A",
                        "criterion": "operational_feasibility",
                    }
                ]
            }
        if agent in {"Finance Agent", "Operations Agent"} and ai_target:
            return {
                "agent_name": agent,
                "selected_items": [ai_target["id"]],
                "critiques": [
                    {
                        "target_item_id": ai_target["id"],
                        "target_agent": "Technology Agent",
                        "target_claim": ai_target["content"],
                        "critique": f"{agent} challenges the AI feasibility claim because deployment timing depends on staged data readiness, plant integration, and governance capacity.",
                        "missing_assumption": "The claim needs explicit evidence of sensor coverage, data quality, and implementation capacity.",
                        "risk_introduced": "If the AI path is treated as ready too early, EuroTech could miss OEM proof points.",
                        "topic_tags": ["ai_feasibility", "data_readiness", "implementation_timeline"],
                        "severity": 4,
                        "confidence": 0.82,
                        "related_strategy": "A",
                        "criterion": "operational_feasibility",
                    }
                ],
            }
        target = "Strategy Agent" if agent != "Strategy Agent" else "Finance Agent"
        strategy_target = self._find_blackboard_item(
            prompt,
            author=target,
            item_type="claim",
            content_contains="Strategy C provides the best balance",
        )
        return {
            "agent_name": agent,
            "selected_items": [strategy_target.get("id")] if strategy_target else [],
            "critiques": [
                {
                    "target_item_id": strategy_target.get("id"),
                    "target_agent": target,
                    "target_claim": "Strategy C provides the best balance of near-term OEM protection and diversification learning.",
                    "critique": f"{agent} says this balance is plausible but needs explicit sequencing and decision gates.",
                    "missing_assumption": "The claim assumes management capacity for dual execution.",
                    "risk_introduced": "Without sequencing, the dual-track option may dilute focus.",
                    "topic_tags": ["strategic_value", "implementation_timeline"],
                    "severity": 3,
                    "confidence": 0.74,
                    "related_strategy": "C",
                    "criterion": "strategic_value",
                }
            ]
        }

    def _round3(self, agent: str, prompt: str) -> dict[str, Any]:
        changed = ["Strategy C remains preferred if investment gates and sequencing are explicit."]
        accepted = ["Accepted critique that dual-track execution needs governance and milestones."]
        concerns = ["Remaining concern: cash and management bandwidth may constrain parallel execution."]
        if "3 months with minimal cost and minimal integration risk" in prompt:
            accepted.append("Accepted critiques rejecting the unsupported 3-month plant-wide AI deployment claim.")
            changed.append("AI quality deployment should be staged rather than assumed complete in 3 months.")
        if "extends the compliance deadline from 18 months to 24 months" in prompt:
            changed.append("The 24-month deadline improves feasibility of phased OEM compliance.")
            concerns.append("New timing relief should not be used to delay data-readiness work.")
        return {
            "agent_name": agent,
            "updated_recommendation": "C",
            "changed_beliefs": changed,
            "accepted_critiques": accepted,
            "rejected_critiques": [
                "Rejected any implication that Strategy D is safer merely because it spends less.",
            ],
            "remaining_concerns": concerns,
            "confidence": 0.8,
        }

    def _round4(self, agent: str) -> dict[str, Any]:
        matrices = {
            "Finance Agent": {
                "A": [3, 3, 3, 3, 4],
                "B": [2, 2, 4, 2, 3],
                "C": [4, 4, 5, 3, 4],
                "D": [2, 3, 1, 3, 2],
            },
            "Operations Agent": {
                "A": [3, 4, 3, 3, 4],
                "B": [2, 2, 4, 2, 3],
                "C": [4, 4, 4, 3, 4],
                "D": [3, 3, 1, 3, 2],
            },
            "HR Agent": {
                "A": [3, 3, 3, 3, 4],
                "B": [2, 2, 4, 2, 3],
                "C": [4, 3, 4, 4, 4],
                "D": [3, 3, 1, 3, 2],
            },
            "Legal / Compliance Agent": {
                "A": [3, 3, 3, 3, 5],
                "B": [2, 2, 4, 2, 3],
                "C": [4, 4, 4, 3, 5],
                "D": [2, 3, 1, 3, 2],
            },
            "Strategy Agent": {
                "A": [3, 3, 3, 3, 4],
                "B": [2, 2, 5, 2, 3],
                "C": [4, 4, 5, 3, 4],
                "D": [2, 3, 1, 3, 2],
            },
            "Technology Agent": {
                "A": [3, 4, 3, 3, 4],
                "B": [2, 2, 4, 2, 3],
                "C": [4, 4, 4, 3, 4],
                "D": [3, 3, 1, 3, 2],
            },
        }
        selected = matrices.get(agent, matrices["Strategy Agent"])
        criteria_names = [
            "financial_viability",
            "operational_feasibility",
            "strategic_value",
            "workforce_feasibility",
            "legal_compliance_safety",
        ]
        scores = {}
        for strategy, values in selected.items():
            criteria = dict(zip(criteria_names, values, strict=True))
            scores[strategy] = {
                "criteria": criteria,
                "overall": round(sum(values) / len(values), 2),
                "rationale": f"{agent} scores Strategy {strategy} against the shared MCDA criteria.",
            }
        return {"scores": scores, "confidence": 0.82}

    def _evaluator(self, prompt: str) -> dict[str, Any]:
        metrics = self._trailing_json_section(prompt, "DETERMINISTIC_METRICS")
        final_recommendation = self._json_section(prompt, "FINAL_RECOMMENDATION")
        model_calls = float(metrics.get("model_calls", 0) or 0)
        tokens = float(metrics.get("total_estimated_tokens", 0) or 0)
        blackboard_items = float(metrics.get("number_of_blackboard_items", 0) or 0)
        scorecards = float(metrics.get("number_of_scorecards", 0) or 0)
        conflicts = float(metrics.get("number_of_conflicts", 0) or 0)
        critique_items = float(metrics.get("number_of_critique_items", 0) or 0)
        belief_updates = float(metrics.get("number_of_belief_update_items", 0) or 0)
        rounds = float(metrics.get("number_of_rounds", 0) or 0)
        agreement_ratio = metrics.get("agreement_ratio")
        unresolved_conflicts = float(metrics.get("unresolved_high_severity_conflicts", 0) or 0)

        has_structured_consensus = bool(final_recommendation.get("aggregate_scores"))
        has_tradeoff_fields = any(
            final_recommendation.get(field)
            for field in ("reasoning", "synthesis", "tradeoffs", "risks", "assumptions", "unresolved_risks")
        )

        if agreement_ratio is None:
            convergence = 3.5 + min(rounds, 2.0) * 0.1
        else:
            convergence = 3.4 + (float(agreement_ratio) * 1.0) - min(unresolved_conflicts, 5.0) * 0.06

        scores = {
            "decision_quality": self._score(
                3.6
                + (0.2 if has_tradeoff_fields else 0.0)
                + (0.3 if has_structured_consensus else 0.0)
                + (0.1 if metrics.get("final_selected_strategy") == "C" else 0.0)
            ),
            "convergence": self._score(convergence),
            "resilience": self._score(
                4.4
                if metrics.get("fault_correction_detected")
                else 3.2 + min(critique_items / 12.0, 0.5) + min(conflicts / 10.0, 0.3)
            ),
            "explainability": self._score(
                3.1
                + min(blackboard_items / 60.0, 1.0) * 0.9
                + min(scorecards / 6.0, 1.0) * 0.4
                + min(conflicts / 6.0, 1.0) * 0.3
                + (0.2 if rounds > 1 else 0.0)
            ),
            "adaptability": self._score(
                4.3
                if metrics.get("new_information_incorporated")
                else 3.2 + min(belief_updates / 36.0, 0.4)
            ),
            "cost_efficiency": self._score(5.0 - (model_calls * 0.18) - (tokens / 30000.0)),
            "runtime_efficiency": self._score(5.0 - (model_calls * 0.12) - (tokens / 50000.0)),
        }
        weights = {
            "decision_quality": 0.25,
            "convergence": 0.12,
            "resilience": 0.15,
            "explainability": 0.18,
            "adaptability": 0.12,
            "cost_efficiency": 0.10,
            "runtime_efficiency": 0.08,
        }
        weighted = sum(scores[key] * weight for key, weight in weights.items())
        scores["overall"] = float(Decimal(str(weighted)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        return {
            **scores,
            "rationale": "The answer selects a defensible staged strategy, identifies tradeoffs, and exposes enough assumptions and risks for audit.",
        }

