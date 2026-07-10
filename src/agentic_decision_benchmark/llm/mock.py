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
            "GENERALIST_RECOMMENDATION": lambda: self._generalist(prompt),
            "SUPERVISOR_PLAN": self._supervisor_plan,
            "ISOLATED_DOMAIN_ANALYSIS": lambda: self._domain_analysis(agent, prompt),
            "SUPERVISOR_SYNTHESIS": lambda: self._supervisor_synthesis(prompt),
            "SELF_ORGANIZING_ROUND_1": lambda: self._round1(agent, prompt),
            "SELF_ORGANIZING_ROUND_2": lambda: self._round2(agent, prompt),
            "SELF_ORGANIZING_ROUND_3": lambda: self._round3(agent, prompt),
            "SELF_ORGANIZING_ROUND_4": lambda: self._round4(agent, prompt),
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
        marker_text = f"{marker}: "
        if marker_text not in prompt:
            return {}
        text = prompt.split(marker_text, 1)[1].lstrip()
        try:
            payload, _ = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

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
    def _candidate_strategies(prompt: str) -> dict[str, Any]:
        context = MockProvider._json_section(prompt, "CONTEXT")
        strategies = context.get("candidate_strategies", {})
        return strategies if isinstance(strategies, dict) else {}

    def _strategy_ids(self, prompt: str) -> tuple[str, ...]:
        strategies = self._candidate_strategies(prompt)
        strategy_ids = tuple(str(key) for key in strategies)
        return strategy_ids or ("A", "B", "C", "D")

    def _strategy_text(self, prompt: str, strategy_id: str) -> str:
        details = self._strategy_details(prompt, strategy_id)
        values = [
            strategy_id,
            details.get("name"),
            details.get("description"),
            details.get("core_tradeoff"),
        ]
        return " ".join(str(value) for value in values if value).lower()

    def _strategy_id_by_keywords(
        self,
        prompt: str,
        keywords: tuple[str, ...],
        *,
        fallback: str | None = None,
    ) -> str | None:
        scored: list[tuple[int, str]] = []
        for strategy_id in self._strategy_ids(prompt):
            text = self._strategy_text(prompt, strategy_id)
            score = sum(1 for keyword in keywords if keyword.lower() in text)
            if score:
                scored.append((score, strategy_id))
        if scored:
            return sorted(scored, key=lambda item: (-item[0], item[1]))[0][1]
        return fallback

    def _strategy_ranking_by_keyword_groups(
        self,
        prompt: str,
        keyword_groups: list[tuple[str, ...]],
    ) -> list[str]:
        ranking: list[str] = []
        for keywords in keyword_groups:
            strategy_id = self._strategy_id_by_keywords(prompt, keywords)
            if strategy_id and strategy_id not in ranking:
                ranking.append(strategy_id)
        for strategy_id in self._strategy_ids(prompt):
            if strategy_id not in ranking:
                ranking.append(strategy_id)
        return ranking

    def _preferred_strategy_id(self, prompt: str) -> str:
        strategy_ids = self._strategy_ids(prompt)
        return self._strategy_id_by_keywords(
            prompt,
            ("dual", "option", "conditional", "pilot", "negotiation", "flexibility"),
            fallback=strategy_ids[0],
        ) or strategy_ids[0]

    def _technology_strategy_id(self, prompt: str) -> str:
        return self._strategy_id_by_keywords(
            prompt,
            ("pilot", "predictive quality", "data", "ai", "roadmap"),
            fallback=self._preferred_strategy_id(prompt),
        ) or self._preferred_strategy_id(prompt)

    def _domain_strategy_id(self, agent: str, prompt: str) -> str:
        keywords_by_agent = {
            "Finance Agent": ("liquidity", "cash", "pause", "freeze"),
            "Operations Agent": ("oem", "assurance", "quality", "sprint"),
            "HR Agent": ("liquidity", "workforce stability", "pause", "freeze"),
            "Legal / Compliance Agent": ("oem", "contract", "compliance", "assurance"),
            "Strategy Agent": ("renewables", "wind", "diversification", "capacity commitment"),
            "Technology Agent": ("pilot", "predictive quality", "data", "roadmap"),
        }
        return self._strategy_id_by_keywords(
            prompt,
            keywords_by_agent.get(agent, ("option", "strategic")),
            fallback=self._preferred_strategy_id(prompt),
        ) or self._preferred_strategy_id(prompt)

    def _domain_ranking(self, agent: str, prompt: str) -> list[str]:
        keyword_groups_by_agent = {
            "Finance Agent": [
                ("liquidity", "cash", "pause", "freeze"),
                ("dual", "option", "conditional", "pilot"),
                ("oem", "assurance", "compliance"),
                ("renewables", "wind", "retooling"),
            ],
            "Operations Agent": [
                ("oem", "assurance", "quality", "sprint"),
                ("dual", "option", "conditional", "pilot"),
                ("liquidity", "pause", "freeze"),
                ("renewables", "wind", "retooling"),
            ],
            "HR Agent": [
                ("liquidity", "workforce stability", "pause", "freeze"),
                ("dual", "option", "conditional", "pilot"),
                ("oem", "assurance", "compliance"),
                ("renewables", "wind", "retooling"),
            ],
            "Legal / Compliance Agent": [
                ("oem", "contract", "compliance", "assurance"),
                ("dual", "option", "conditional", "written"),
                ("liquidity", "pause", "freeze"),
                ("renewables", "wind", "penalties"),
            ],
            "Strategy Agent": [
                ("renewables", "wind", "diversification", "capacity commitment"),
                ("dual", "option", "conditional", "negotiation"),
                ("oem", "assurance", "customer concentration"),
                ("liquidity", "pause", "freeze"),
            ],
            "Technology Agent": [
                ("pilot", "predictive quality", "data", "roadmap"),
                ("oem", "assurance", "ai"),
                ("liquidity", "pause", "freeze"),
                ("renewables", "wind", "process"),
            ],
        }
        return self._strategy_ranking_by_keyword_groups(
            prompt,
            keyword_groups_by_agent.get(agent, [("option", "strategic")]),
        )

    def _organization_ranking(self, prompt: str) -> list[str]:
        return self._strategy_ranking_by_keyword_groups(
            prompt,
            [
                ("dual", "option", "conditional", "pilot", "negotiation"),
                ("oem", "assurance", "compliance"),
                ("renewables", "wind", "diversification"),
                ("liquidity", "pause", "freeze"),
            ],
        )

    def _strategy_details(self, prompt: str, strategy_id: str) -> dict[str, Any]:
        details = self._candidate_strategies(prompt).get(strategy_id, {})
        return details if isinstance(details, dict) else {}

    def _strategy_name(self, prompt: str, strategy_id: str) -> str:
        details = self._strategy_details(prompt, strategy_id)
        return str(details.get("name") or f"Strategy {strategy_id}")

    def _strategy_label(self, prompt: str, strategy_id: str) -> str:
        name = self._strategy_name(prompt, strategy_id)
        fallback = f"Strategy {strategy_id}"
        return f"{fallback} ({name})" if name != fallback else fallback

    def _strategy_description(self, prompt: str, strategy_id: str) -> str:
        details = self._strategy_details(prompt, strategy_id)
        return str(details.get("description") or details.get("name") or f"Strategy {strategy_id}")

    def _strategy_tradeoff(self, prompt: str, strategy_id: str) -> str:
        details = self._strategy_details(prompt, strategy_id)
        return str(details.get("core_tradeoff") or "Execution depends on disciplined milestones and evidence gates.")

    def _strategy_claim(self, prompt: str) -> str:
        strategy_id = self._preferred_strategy_id(prompt)
        return f"{self._strategy_label(prompt, strategy_id)} best matches the case: {self._strategy_description(prompt, strategy_id)}"

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

    def _generalist(self, prompt: str) -> dict[str, Any]:
        has_new_information = "extends the compliance deadline from 18 months to 24 months" in prompt
        preferred = self._preferred_strategy_id(prompt)
        c_label = self._strategy_label(prompt, preferred)
        c_description = self._strategy_description(prompt, preferred)
        c_tradeoff = self._strategy_tradeoff(prompt, preferred)
        reasoning = [
            "The OEM revenue exposure is too large to ignore.",
            "A full wind pivot creates execution and cash-flow risk.",
            f"{c_label} is the best fit among the provided options: {c_description}",
        ]
        risks = [
            f"Core tradeoff for {c_label}: {c_tradeoff}",
            "Cash constraints remain material if milestones expand faster than the capital envelope.",
            "Supplier and workforce readiness may become bottlenecks.",
        ]
        assumptions = [
            f"EuroTech can execute {c_label} with clear decision gates.",
            "External counterparties accept the staged evidence and negotiation path described in the chosen strategy.",
        ]
        if has_new_information:
            reasoning.append(
                "The 24-month compliance deadline improves feasibility of phased OEM AI compliance, but data-readiness work should still start immediately."
            )
            risks.append("The extra deadline relief could create complacency if EuroTech delays sensor, data, and auditability work.")
            assumptions.append("The OEM's 24 months deadline extension is firm enough to use for planning gates.")
        return {
            "recommended_strategy": preferred,
            "recommendation": f"Adopt {c_label}: {c_description}",
            "reasoning": reasoning,
            "risks": risks,
            "assumptions": assumptions,
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
        preferred = self._preferred_strategy_id(prompt)
        c_label = self._strategy_label(prompt, preferred)
        c_description = self._strategy_description(prompt, preferred)
        recommendation = preferred
        if agent == "Technology Agent":
            recommendation = self._technology_strategy_id(prompt)
        if agent == "Strategy Agent":
            recommendation = preferred
        claims = [
            f"From the {domain} perspective, {c_label} keeps the most decision flexibility.",
            f"Its stated action plan is: {c_description}",
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

    def _supervisor_synthesis(self, prompt: str) -> dict[str, Any]:
        has_new_information = "extends the compliance deadline from 18 months to 24 months" in prompt
        preferred = self._preferred_strategy_id(prompt)
        c_label = self._strategy_label(prompt, preferred)
        c_description = self._strategy_description(prompt, preferred)
        c_tradeoff = self._strategy_tradeoff(prompt, preferred)
        synthesis = [
            "Finance and operations favor staged investment over a large irreversible pivot.",
            "Technology and compliance require auditability, data readiness, and model monitoring.",
            f"{c_label} fits the provided option set: {c_description}",
        ]
        tradeoffs = [
            f"Core tradeoff for {c_label}: {c_tradeoff}",
            "It delays full wind commitment compared with B.",
        ]
        risks = [
            "Sequencing and governance overload.",
            "Insufficient AI data readiness.",
            "Hiring bottlenecks for quality analytics and wind production skills.",
        ]
        assumptions = [
            "The OEM accepts phased evidence of predictive quality management or a negotiated conditional path.",
            "The wind partner accepts the timing implied by the selected strategy.",
        ]
        if has_new_information:
            synthesis.append("The 24-month compliance deadline gives the supervisor more room to phase OEM evidence without abandoning urgency.")
            tradeoffs.append(f"The deadline extension improves feasibility of {c_label} but reduces the case for an all-out Strategy A sprint.")
            risks.append("A longer deadline can still be missed if foundational data and auditability milestones slip.")
            assumptions.append("The OEM's 24 months deadline extension applies to the same predictive quality requirement.")
        return {
            "recommended_strategy": preferred,
            "recommendation": f"Choose {c_label} with explicit gates: {c_description}",
            "synthesis": synthesis,
            "tradeoffs": tradeoffs,
            "risks": risks,
            "assumptions": assumptions,
            "confidence": 0.78,
        }

    def _round1(self, agent: str, prompt: str) -> dict[str, Any]:
        domain = self._agent_domain(agent)
        organization_preference = self._preferred_strategy_id(prompt)
        domain_preference = self._domain_strategy_id(agent, prompt)
        domain_ranking = self._domain_ranking(agent, prompt)
        organization_ranking = self._organization_ranking(prompt)
        challenged = [strategy_id for strategy_id in domain_ranking[-2:] if strategy_id != domain_preference]
        supported = [strategy_id for strategy_id in [domain_preference, organization_preference] if strategy_id not in challenged]
        wind_strategy = self._strategy_id_by_keywords(
            prompt,
            ("renewables", "wind", "retooling"),
            fallback=domain_ranking[-1],
        ) or domain_ranking[-1]
        liquidity_strategy = self._strategy_id_by_keywords(
            prompt,
            ("liquidity", "cash", "pause", "freeze"),
            fallback=domain_ranking[-1],
        ) or domain_ranking[-1]
        domain_label = self._strategy_label(prompt, domain_preference)
        organization_label = self._strategy_label(prompt, organization_preference)
        structured_items = [
            {
                "item_type": "claim",
                "content": f"From the {domain} perspective, {domain_label} best fits the role-specific constraints.",
                "topic_tags": ["strategic_value"],
                "related_strategy": domain_preference,
                "strategies_supported": [domain_preference],
                "criterion": "strategic_value",
                "stance": "supports",
                "confidence": 0.68,
                "severity": 3,
            },
            {
                "item_type": "claim",
                "content": f"As an organization-wide compromise, {organization_label} preserves the most negotiable option value.",
                "topic_tags": ["strategic_value", "diversification"],
                "related_strategy": organization_preference,
                "strategies_supported": [organization_preference],
                "criterion": "strategic_value",
                "stance": "supports",
                "confidence": 0.68,
                "severity": 3,
            },
            {
                "item_type": "risk",
                "content": f"{agent} flags that {self._strategy_label(prompt, wind_strategy)} may overextend investment capacity and execution bandwidth.",
                "topic_tags": ["investment_capacity", "cash_flow"],
                "related_strategy": wind_strategy,
                "strategies_challenged": [wind_strategy],
                "criterion": "financial_viability",
                "stance": "opposes",
                "confidence": 0.72,
                "severity": 4 if agent in {"Finance Agent", "Operations Agent"} else 3,
            },
        ]
        if agent == "Technology Agent":
            technology_strategy = self._technology_strategy_id(prompt)
            structured_items.append(
                {
                    "item_type": "claim",
                    "content": "Pilot-line AI predictive quality deployment is feasible in 9 to 12 months if data readiness work starts immediately.",
                    "topic_tags": ["ai_feasibility", "data_readiness", "implementation_timeline"],
                    "related_strategy": technology_strategy,
                    "strategies_supported": [technology_strategy],
                    "criterion": "operational_feasibility",
                    "stance": "supports",
                    "confidence": 0.78,
                    "severity": 4,
                }
            )
        if agent == "Operations Agent":
            technology_strategy = self._technology_strategy_id(prompt)
            structured_items.append(
                {
                    "item_type": "risk",
                    "content": "Plant-wide AI rollout could disrupt production if retooling and data work are compressed.",
                    "topic_tags": ["ai_feasibility", "production_disruption", "implementation_timeline"],
                    "related_strategy": technology_strategy,
                    "strategies_challenged": [technology_strategy],
                    "criterion": "operational_feasibility",
                    "stance": "opposes",
                    "confidence": 0.8,
                    "severity": 4,
                }
            )
        risks = [
            f"{agent} flags that {self._strategy_label(prompt, wind_strategy)} may overextend the organization.",
        ]
        risks.append(
            f"{self._strategy_label(prompt, liquidity_strategy)} risks both OEM loss and missed wind timing if used as the default posture."
        )
        conditions = {
            strategy_id: [
                f"{agent} would accept {self._strategy_label(prompt, strategy_id)} only if it satisfies the {domain} non-negotiables."
            ]
            for strategy_id in domain_ranking
            if strategy_id != domain_preference
        }

        return {
            "claims": [
                f"{agent} sees {domain} as a binding constraint on any strategy.",
                f"{domain_label} is the domain-preferred option for this role.",
                f"{organization_label} is the separate organization-wide compromise.",
            ],
            "risks": risks,
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
            "domain_preferred_strategy": domain_preference,
            "organization_preferred_strategy": organization_preference,
            "domain_ranking": domain_ranking,
            "organization_ranking": organization_ranking,
            "strategies_supported": supported,
            "strategies_challenged": challenged,
            "non_negotiable_constraints": [
                f"{agent} cannot endorse a plan that ignores {domain}.",
                "Any accepted plan must have explicit gates, accountable owners, and evidence milestones.",
            ],
            "conditions_for_accepting_other_strategy": conditions,
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
            technology_strategy = self._technology_strategy_id(prompt)
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
                        "related_strategy": technology_strategy,
                        "criterion": "operational_feasibility",
                    }
                ]
            }
        if agent in {"Finance Agent", "Operations Agent"} and ai_target:
            technology_strategy = self._technology_strategy_id(prompt)
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
                        "related_strategy": technology_strategy,
                        "criterion": "operational_feasibility",
                    }
                ],
            }
        target = "Strategy Agent" if agent != "Strategy Agent" else "Finance Agent"
        strategy_claim = self._strategy_claim(prompt)
        strategy_target = self._find_blackboard_item(
            prompt,
            author=target,
            item_type="claim",
            content_contains=strategy_claim,
        )
        return {
            "agent_name": agent,
            "selected_items": [strategy_target.get("id")] if strategy_target else [],
            "critiques": [
                {
                    "target_item_id": strategy_target.get("id"),
                    "target_agent": target,
                    "target_claim": strategy_target.get("content") or strategy_claim,
                    "critique": f"{agent} says this option is plausible but needs explicit sequencing and decision gates.",
                    "missing_assumption": "The claim assumes management capacity for disciplined staged execution.",
                    "risk_introduced": "Without sequencing, the selected option may dilute focus.",
                    "topic_tags": ["strategic_value", "implementation_timeline"],
                    "severity": 3,
                    "confidence": 0.74,
                    "related_strategy": self._preferred_strategy_id(prompt),
                    "criterion": "strategic_value",
                }
            ]
        }

    def _round3(self, agent: str, prompt: str) -> dict[str, Any]:
        preferred = self._preferred_strategy_id(prompt)
        c_label = self._strategy_label(prompt, preferred)
        changed = [f"{c_label} remains preferred if investment gates and sequencing are explicit."]
        accepted = ["Accepted critique that execution needs governance and milestones."]
        concerns = ["Remaining concern: cash and management bandwidth may constrain sequencing."]
        if "3 months with minimal cost and minimal integration risk" in prompt:
            accepted.append("Accepted critiques rejecting the unsupported 3-month plant-wide AI deployment claim.")
            changed.append("AI quality deployment should be staged rather than assumed complete in 3 months.")
        if "extends the compliance deadline from 18 months to 24 months" in prompt:
            changed.append("The 24-month deadline improves feasibility of phased OEM compliance.")
            concerns.append("New timing relief should not be used to delay data-readiness work.")
        rejected = ["Rejected any implication that a lower-investment option is safer merely because it spends less."]
        if "D" in self._strategy_ids(prompt):
            rejected = ["Rejected any implication that Strategy D is safer merely because it spends less."]

        return {
            "agent_name": agent,
            "updated_recommendation": preferred,
            "changed_beliefs": changed,
            "accepted_critiques": accepted,
            "rejected_critiques": rejected,
            "remaining_concerns": concerns,
            "confidence": 0.8,
        }

    def _round4(self, agent: str, prompt: str) -> dict[str, Any]:
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
        strategy_ids = self._strategy_ids(prompt)
        criteria_names = [
            "financial_viability",
            "operational_feasibility",
            "strategic_value",
            "workforce_feasibility",
            "legal_compliance_safety",
        ]
        scores = {}
        for strategy in strategy_ids:
            values = selected.get(strategy, [3, 3, 3, 3, 3])
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

