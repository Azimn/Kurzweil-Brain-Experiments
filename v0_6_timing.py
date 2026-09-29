from __future__ import annotations

import copy
from typing import Any, Dict

from v0_5_experiment import build_world_pair, TARGET_EVENT_ID, REPLACEMENT_EVENT_ID

LATE_BOUNDARY_POLICY = "penultimate developmental event"
EARLY_INDEX = 5


def _ordered_events(world: Dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(world["events"], key=lambda e: (float(e["age"]), e["id"]))


def _late_index(world: Dict[str, Any]) -> int:
    events = _ordered_events(world)
    if len(events) < 3:
        raise ValueError("world has no eligible late pre-mature boundary")
    return len(events) - 2


def _move_by_age_swap(world: Dict[str, Any], target_id: str, target_index: int) -> Dict[str, Any]:
    """Move one atomic event by swapping only its developmental age with the displaced event.

    DevelopmentalWorld orders events by (age, id), so list-order swaps are scientifically
    inert. Age is the representation's scheduling coordinate. All event content, exposure,
    affordances, rewards, scalars, and action opportunities remain attached to the event.
    """
    result = copy.deepcopy(world)
    ordered = _ordered_events(result)
    source_index = next(i for i, e in enumerate(ordered) if e["id"] == target_id)
    displaced = ordered[target_index]
    target = ordered[source_index]
    target_age = target["age"]
    displaced_age = displaced["age"]
    target["age"] = displaced_age
    displaced["age"] = target_age
    reordered = _ordered_events(result)
    if reordered[target_index]["id"] != target_id:
        raise ValueError("age swap did not place target at requested boundary")
    return result


def build_timing_arm(timing: str) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    canonical, counterfactual = build_world_pair()
    canonical_order = _ordered_events(canonical)
    counterfactual_order = _ordered_events(counterfactual)
    ci = next(i for i, e in enumerate(canonical_order) if e["id"] == TARGET_EVENT_ID)
    xi = next(i for i, e in enumerate(counterfactual_order) if e["id"] == REPLACEMENT_EVENT_ID)
    if ci != xi or ci != EARLY_INDEX:
        raise ValueError("frozen v0.5 intervention position drift")

    late = _late_index(canonical)
    if timing == "EARLY":
        return canonical, counterfactual, {"timing": timing, "target_index": ci, "late_index": late, "policy": LATE_BOUNDARY_POLICY}
    if timing != "LATE":
        raise ValueError("timing must be EARLY or LATE")

    late_c = _move_by_age_swap(canonical, TARGET_EVENT_ID, late)
    late_x = _move_by_age_swap(counterfactual, REPLACEMENT_EVENT_ID, late)
    return late_c, late_x, {
        "timing": timing,
        "target_index": late,
        "displaced_index": ci,
        "late_index": late,
        "policy": LATE_BOUNDARY_POLICY,
    }


def protected_event_multiset(world: Dict[str, Any]) -> list[dict[str, Any]]:
    """Event content excluding age, the single preregistered timing coordinate."""
    rows = []
    for event in world["events"]:
        row = copy.deepcopy(event)
        row.pop("age", None)
        rows.append(row)
    return sorted(rows, key=lambda e: e["id"])


def validate_timing_arms() -> Dict[str, Any]:
    early_c, early_x, early_meta = build_timing_arm("EARLY")
    late_c, late_x, late_meta = build_timing_arm("LATE")
    if protected_event_multiset(early_c) != protected_event_multiset(late_c):
        raise ValueError("canonical protected event content changed")
    if protected_event_multiset(early_x) != protected_event_multiset(late_x):
        raise ValueError("counterfactual protected event content changed")
    if len(early_c["events"]) != len(late_c["events"]):
        raise ValueError("opportunity count changed")
    if _ordered_events(late_c)[late_meta["target_index"]]["id"] != TARGET_EVENT_ID:
        raise ValueError("canonical target timing move ineffective")
    if _ordered_events(late_x)[late_meta["target_index"]]["id"] != REPLACEMENT_EVENT_ID:
        raise ValueError("counterfactual target timing move ineffective")
    return {"early": early_meta, "late": late_meta}
