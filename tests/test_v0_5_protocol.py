from __future__ import annotations

import copy

import pytest

from v0_5_experiment import (
    REPLACEMENT_EVENT_ID,
    assert_matched_intervention,
    build_world_pair,
    canonical_event,
    non_target_digest,
    reward_signature,
    validate_world_pair,
)


def test_v0_5_world_pair_matches_all_preregistered_protected_fields():
    canonical, counterfactual = build_world_pair()
    original = canonical_event(canonical)
    replacement = next(e for e in counterfactual["events"] if e["id"] == REPLACEMENT_EVENT_ID)
    assert_matched_intervention(original, replacement)
    assert len(canonical["events"]) == len(counterfactual["events"])
    assert non_target_digest(canonical) == non_target_digest(counterfactual)
    assert reward_signature(original) == reward_signature(replacement)


def test_v0_5_changes_exactly_one_event():
    canonical, counterfactual = build_world_pair()
    pairs = list(zip(canonical["events"], counterfactual["events"]))
    changed = [(left, right) for left, right in pairs if left != right]
    assert len(changed) == 1
    assert changed[0][0]["id"] == "senior_ai_mentor_access"
    assert changed[0][1]["id"] == REPLACEMENT_EVENT_ID


def test_v0_5_matching_gate_fails_closed_on_reward_drift():
    canonical, counterfactual = build_world_pair()
    original = canonical_event(canonical)
    replacement = copy.deepcopy(next(e for e in counterfactual["events"] if e["id"] == REPLACEMENT_EVENT_ID))
    replacement["consequences"]["communicate"]["reward"] += 0.01
    with pytest.raises(ValueError, match="reward schedule"):
        assert_matched_intervention(original, replacement)


def test_v0_5_matching_gate_fails_closed_on_scalar_drift():
    canonical, counterfactual = build_world_pair()
    original = canonical_event(canonical)
    replacement = copy.deepcopy(next(e for e in counterfactual["events"] if e["id"] == REPLACEMENT_EVENT_ID))
    replacement["scalars"]["technical_context"] -= 0.01
    with pytest.raises(ValueError, match="protected intervention field"):
        assert_matched_intervention(original, replacement)


def test_v0_5_descriptor_validation_passes():
    validate_world_pair()
