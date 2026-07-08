from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from agentic_decision_benchmark.schemas import BlackboardItem, ConflictItem, StrategyId


def _stable_conflict_id(cycle_id: int, conflict_type: str, parts: list[Any], item_ids: list[str]) -> str:
    payload = "|".join([str(cycle_id), conflict_type, *[str(part) for part in parts], *sorted(item_ids)])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"conflict-{digest}"


def _item_by_id(blackboard: list[BlackboardItem]) -> dict[str, BlackboardItem]:
    return {item.id: item for item in blackboard}


def _topic_label(tags: list[str]) -> str:
    return ",".join(sorted(tags)) if tags else "uncategorized"


def _unique_sorted(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def _direct_critique_conflicts(blackboard: list[BlackboardItem], cycle_id: int) -> list[ConflictItem]:
    by_target: dict[str, list[BlackboardItem]] = defaultdict(list)
    items_by_id = _item_by_id(blackboard)
    for item in blackboard:
        if item.item_type == "critique" and item.target_item_id:
            by_target[item.target_item_id].append(item)

    conflicts: list[ConflictItem] = []
    for target_id, critiques in sorted(by_target.items()):
        target = items_by_id.get(target_id)
        tags = target.topic_tags if target else sorted({tag for critique in critiques for tag in critique.topic_tags})
        item_ids = [target_id, *[critique.id for critique in critiques]]
        agents = [critique.author for critique in critiques]
        if target:
            agents.append(target.author)
        severity = max([critique.severity or 3 for critique in critiques] or [3])
        conflicts.append(
            ConflictItem(
                id=_stable_conflict_id(cycle_id, "direct_critique", [target_id], item_ids),
                cycle_id=cycle_id,
                conflict_type="direct_critique",
                topic=_topic_label(list(tags)),
                related_strategy=target.related_strategy if target else critiques[0].related_strategy,
                criterion=target.criterion if target else critiques[0].criterion,
                central_item_id=target_id,
                supporting_items=[target_id],
                challenging_items=sorted(critique.id for critique in critiques),
                agents_involved=_unique_sorted(agents),
                severity=severity,
                summary=f"{len(critiques)} critique(s) target blackboard item {target_id}.",
            )
        )
    return conflicts


def _opposing_stance_conflicts(blackboard: list[BlackboardItem], cycle_id: int) -> list[ConflictItem]:
    grouped: dict[tuple[str, str, str], list[BlackboardItem]] = defaultdict(list)
    for item in blackboard:
        if item.related_strategy is None or item.stance not in {"supports", "opposes"}:
            continue
        criterion = item.criterion or "general"
        for tag in item.topic_tags:
            grouped[(item.related_strategy, criterion, tag)].append(item)

    conflicts: list[ConflictItem] = []
    for (strategy, criterion, tag), items in sorted(grouped.items()):
        supporting = [item for item in items if item.stance == "supports"]
        opposing = [item for item in items if item.stance == "opposes"]
        if not supporting or not opposing:
            continue
        item_ids = [item.id for item in items]
        conflicts.append(
            ConflictItem(
                id=_stable_conflict_id(cycle_id, "opposing_stances", [strategy, criterion, tag], item_ids),
                cycle_id=cycle_id,
                conflict_type="opposing_stances",
                topic=tag,
                related_strategy=strategy,  # type: ignore[arg-type]
                criterion=None if criterion == "general" else criterion,
                supporting_items=sorted(item.id for item in supporting),
                challenging_items=sorted(item.id for item in opposing),
                agents_involved=_unique_sorted([item.author for item in items]),
                severity=max(item.severity or 3 for item in items),
                summary=f"Agents have opposing stances on Strategy {strategy}, {criterion}, topic {tag}.",
            )
        )
    return conflicts


def _risk_cluster_conflicts(blackboard: list[BlackboardItem], cycle_id: int) -> list[ConflictItem]:
    grouped: dict[str, list[BlackboardItem]] = defaultdict(list)
    for item in blackboard:
        if item.item_type != "risk" or (item.severity or 0) < 4:
            continue
        for tag in item.topic_tags:
            grouped[tag].append(item)

    conflicts: list[ConflictItem] = []
    for tag, items in sorted(grouped.items()):
        agents = _unique_sorted([item.author for item in items])
        if len(agents) < 2:
            continue
        strategies = {item.related_strategy for item in items if item.related_strategy is not None}
        related_strategy: StrategyId | None = next(iter(strategies)) if len(strategies) == 1 else None
        item_ids = [item.id for item in items]
        conflicts.append(
            ConflictItem(
                id=_stable_conflict_id(cycle_id, "risk_cluster", [tag], item_ids),
                cycle_id=cycle_id,
                conflict_type="risk_cluster",
                topic=tag,
                related_strategy=related_strategy,
                supporting_items=[],
                challenging_items=sorted(item_ids),
                agents_involved=agents,
                severity=max(item.severity or 4 for item in items),
                summary=f"{len(agents)} agents posted high-severity risks on topic {tag}.",
            )
        )
    return conflicts


def build_conflict_map(blackboard: list[BlackboardItem], cycle_id: int) -> list[ConflictItem]:
    conflicts = [
        *_direct_critique_conflicts(blackboard, cycle_id),
        *_opposing_stance_conflicts(blackboard, cycle_id),
        *_risk_cluster_conflicts(blackboard, cycle_id),
    ]
    return sorted(
        conflicts,
        key=lambda item: (
            item.conflict_type,
            item.topic,
            item.related_strategy or "",
            item.criterion or "",
            item.id,
        ),
    )

