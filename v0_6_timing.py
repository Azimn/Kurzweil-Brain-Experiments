from __future__ import annotations

import copy
from typing import Any, Dict

from v0_5_experiment import build_world_pair, TARGET_EVENT_ID, REPLACEMENT_EVENT_ID

LATE_BOUNDARY_POLICY = "penultimate developmental event"
EARLY_INDEX = 5


def _late_index(world: Dict[str, Any]) -> int:
    events = world["events"]
    if len(events) < 3:
        raise ValueError("world has no eligible late pre-mature boundary")
    return len(events) - 2


def _swap(events: list[dict[str, Any]], left: int, right: int) -> list[dict[str, Any]]:
    result = copy.deepcopy(events)
    result[left], result[right] = result[right], result[left]
    return result


def build_timing_arm(timing: str) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    canonical, counterfactual = build_world_pair()
    ci = next(i for i, e in enumerate(canonical["events"]) if e["id"] == TARGET_EVENT_ID)
    xi = next(i for i, e in enumerate(counterfactual["events"]) if e["id"] == REPLACEMENT_EVENT_ID)
    if ci != xi or ci != EARLY_INDEX:
        raise ValueError("frozen v0.5 intervention position drift")

    if timing == "EARLY":
        late = _late_index(canonical)
        return canonical, counterfactual, {"timing": timing, "target_index": ci, "late_index": late, "policy": LATE_BOUNDARY_POLICY}
    if timing != "LATE":
        raise ValueError("timing must be EARLY or LATE")

    late = _late_index(canonical)
    late_c = copy.deepcopy(canonical)
    late_x = copy.deepcopy(counterfactual)
    late_c["events"] = _swap(canonical["events"], ci, late)
    late_x["events"] = _swap(counterfactual["events"], xi, late)
    return late_c, late_x, {"timing": timing, "target_index": late, "displaced_index": ci, "late_index": late, "policy": LATE_BOUNDARY_POLICY}


def event_multiset(world: Dict[str, Any]) -> list[dict[str, Any]]:
    return sorted((copy.deepcopy(e) for e in world["events"]), key=lambda e: e["id"])


def validate_timing_arms() -> Dict[str, Any]:
    early_c, early_x, early_meta = build_timing_arm("EARLY")
    late_c, late_x, late_meta = build_timing_arm("LATE")
    if event_multiset(early_c) != event_multiset(late_c):
        raise ValueError("canonical event multiset changed")
    if event_multiset(early_x) != event_multiset(late_x):
        raise ValueError("counterfactual event multiset changed")
    if len(early_c["events"]) != len(late_c["events"]):
        raise ValueError("opportunity count changed")
    return {"early": early_meta, "late": late_meta}
