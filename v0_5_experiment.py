from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent
CANONICAL_WORLD = ROOT / "data" / "development" / "kurzweil_canonical.json"
INTERVENTION_DESCRIPTOR = ROOT / "data" / "v0_5" / "mentor_matched_counterfactual.json"
TARGET_EVENT_ID = "senior_ai_mentor_access"
REPLACEMENT_EVENT_ID = "senior_general_problem_solving_mentor_access"
PROTECTED_EVENT_FIELDS = ("age", "exposure_weight", "scalars", "available_actions")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_event(world: Dict[str, Any]) -> Dict[str, Any]:
    matches = [event for event in world["events"] if event["id"] == TARGET_EVENT_ID]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {TARGET_EVENT_ID} event")
    return matches[0]


def reward_signature(event: Dict[str, Any]) -> Dict[str, float]:
    return {
        action: float(event["consequences"][action].get("reward", 0.0))
        for action in event["available_actions"]
    }


def assert_matched_intervention(canonical: Dict[str, Any], replacement: Dict[str, Any]) -> None:
    if replacement.get("id") != REPLACEMENT_EVENT_ID:
        raise ValueError("unexpected replacement event identity")
    for field in PROTECTED_EVENT_FIELDS:
        if canonical.get(field) != replacement.get(field):
            raise ValueError(f"protected intervention field differs: {field}")
    if reward_signature(canonical) != reward_signature(replacement):
        raise ValueError("action-specific reward schedule differs")
    if set(canonical["consequences"]) != set(replacement["consequences"]):
        raise ValueError("consequence action vocabulary differs")


def build_world_pair() -> tuple[Dict[str, Any], Dict[str, Any]]:
    canonical = load_json(CANONICAL_WORLD)
    descriptor = load_json(INTERVENTION_DESCRIPTOR)
    replacement = descriptor["replacement_event"]
    original = canonical_event(canonical)
    assert_matched_intervention(original, replacement)

    counterfactual = copy.deepcopy(canonical)
    replaced = 0
    for index, event in enumerate(counterfactual["events"]):
        if event["id"] == TARGET_EVENT_ID:
            counterfactual["events"][index] = copy.deepcopy(replacement)
            replaced += 1
    if replaced != 1:
        raise ValueError("counterfactual must replace exactly one event")
    counterfactual["condition"] = "v0.5_matched_general_problem_solving_mentor"
    counterfactual["method_note"] = canonical.get("method_note", "")
    return canonical, counterfactual


def non_target_digest(world: Dict[str, Any]) -> str:
    payload = [event for event in world["events"] if event["id"] not in {TARGET_EVENT_ID, REPLACEMENT_EVENT_ID}]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_world_pair() -> None:
    canonical, counterfactual = build_world_pair()
    if len(canonical["events"]) != len(counterfactual["events"]):
        raise ValueError("opportunity count differs")
    if non_target_digest(canonical) != non_target_digest(counterfactual):
        raise ValueError("non-target developmental events differ")
    original = canonical_event(canonical)
    replacement = next(event for event in counterfactual["events"] if event["id"] == REPLACEMENT_EVENT_ID)
    assert_matched_intervention(original, replacement)


if __name__ == "__main__":
    validate_world_pair()
    print("v0.5 matched-intervention descriptor: OK")
