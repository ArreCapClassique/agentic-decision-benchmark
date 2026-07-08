from __future__ import annotations

from collections import Counter
from typing import Any

from agentic_decision_benchmark.schemas import BlackboardItem, SalienceRecord
from agentic_decision_benchmark.utils.json_utils import to_jsonable


DOMAIN_TOPIC_RELEVANCE: dict[str, set[str]] = {
    "finance": {"financial_viability", "investment_capacity", "cash_flow", "revenue_exposure"},
    "operations": {"operational_feasibility", "retooling_risk", "production_disruption", "implementation_timeline"},
    "human resources": {"workforce_feasibility", "hiring_timeline", "skill_gap"},
    "legal / compliance": {"compliance", "auditability", "contract_risk"},
    "strategy": {"strategic_value", "diversification", "market_uncertainty", "customer_concentration"},
    "technology": {"data_readiness", "ai_feasibility", "implementation_timeline", "auditability"},
}


def _item_id(item: BlackboardItem | dict[str, Any]) -> str:
    return item.id if isinstance(item, BlackboardItem) else str(item.get("id", ""))


def _item_tags(item: BlackboardItem | dict[str, Any]) -> list[str]:
    tags = item.topic_tags if isinstance(item, BlackboardItem) else item.get("topic_tags", [])
    return [str(tag) for tag in tags] or ["uncategorized"]


def _item_confidence(item: BlackboardItem | dict[str, Any]) -> float:
    value = item.confidence if isinstance(item, BlackboardItem) else item.get("confidence", 0.5)
    return max(0.0, min(1.0, float(value)))


def _item_severity(item: BlackboardItem | dict[str, Any]) -> int:
    value = item.severity if isinstance(item, BlackboardItem) else item.get("severity")
    return int(value) if value is not None else 3


def _item_type(item: BlackboardItem | dict[str, Any]) -> str:
    return item.item_type if isinstance(item, BlackboardItem) else str(item.get("item_type", ""))


def _item_target_id(item: BlackboardItem | dict[str, Any]) -> str | None:
    return item.target_item_id if isinstance(item, BlackboardItem) else item.get("target_item_id")


def domain_relevance_for_tags(tags: list[str], agent_domain: str) -> float:
    relevant_tags = DOMAIN_TOPIC_RELEVANCE.get(agent_domain.strip().lower(), set())
    if not relevant_tags:
        return 0.5
    if "uncategorized" in tags:
        return 0.5
    matches = len(set(tags) & relevant_tags)
    return round(matches / len(set(tags)), 4) if tags else 0.5


def compute_salience_map(
    blackboard: list[BlackboardItem | dict[str, Any]],
    agent_domains: list[str] | None = None,
) -> dict[str, SalienceRecord]:
    critique_counts = Counter(
        target_id
        for item in blackboard
        if _item_type(item) == "critique"
        for target_id in [_item_target_id(item)]
        if target_id
    )
    support_counts = Counter(
        tag
        for item in blackboard
        if (item.stance if isinstance(item, BlackboardItem) else item.get("stance")) == "supports"
        for tag in _item_tags(item)
    )
    tag_seen: Counter[str] = Counter()
    records: dict[str, SalienceRecord] = {}
    domains = agent_domains or []

    for item in blackboard:
        item_id = _item_id(item)
        if not item_id:
            continue
        tags = _item_tags(item)
        prior_tag_count = sum(tag_seen[tag] for tag in tags)
        novelty = 1.0 if prior_tag_count == 0 else round(1.0 / (1.0 + prior_tag_count), 4)
        for tag in tags:
            tag_seen[tag] += 1

        severity = round(max(1, min(5, _item_severity(item))) / 5.0, 4)
        confidence = _item_confidence(item)
        critique_count = int(critique_counts.get(item_id, 0))
        normalized_critique_count = min(1.0, critique_count / 3.0)
        domain_relevance = {
            domain: domain_relevance_for_tags(tags, domain)
            for domain in domains
        }
        domain_relevance_value = max(domain_relevance.values(), default=0.5)
        salience = round(
            (0.30 * severity)
            + (0.25 * novelty)
            + (0.20 * confidence)
            + (0.15 * normalized_critique_count)
            + (0.10 * domain_relevance_value),
            4,
        )
        records[item_id] = SalienceRecord(
            item_id=item_id,
            severity=severity,
            confidence=confidence,
            novelty=novelty,
            critique_count=critique_count,
            support_count=sum(support_counts.get(tag, 0) for tag in tags),
            salience=salience,
            domain_relevance=domain_relevance,
        )
    return records


def rank_salient_items_for_agent(
    blackboard: list[BlackboardItem],
    salience_map: dict[str, SalienceRecord | dict[str, Any]],
    agent_domain: str,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for item in blackboard:
        record = salience_map.get(item.id)
        if record is None:
            continue
        record_data = to_jsonable(record)
        relevance = float(record_data.get("domain_relevance", {}).get(agent_domain, 0.5))
        rank_score = round((0.70 * float(record_data["salience"])) + (0.30 * relevance), 4)
        item_data = to_jsonable(item)
        item_data["salience"] = record_data
        item_data["agent_domain_relevance"] = relevance
        ranked.append((rank_score, item.id, item_data))
    ranked.sort(key=lambda entry: (-entry[0], entry[1]))
    return [entry[2] for entry in ranked[:limit]]

