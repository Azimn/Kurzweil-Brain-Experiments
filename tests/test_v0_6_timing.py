from v0_5_experiment import REPLACEMENT_EVENT_ID, TARGET_EVENT_ID
from v0_6_timing import (
    _ordered_events,
    build_timing_arm,
    protected_event_multiset,
    validate_timing_arms,
)


def _by_id(world):
    return {event["id"]: event for event in world["events"]}


def test_timing_arms_preserve_protected_event_content():
    meta = validate_timing_arms()
    early_c, early_x, _ = build_timing_arm("EARLY")
    late_c, late_x, _ = build_timing_arm("LATE")

    assert protected_event_multiset(early_c) == protected_event_multiset(late_c)
    assert protected_event_multiset(early_x) == protected_event_multiset(late_x)
    assert len(early_c["events"]) == len(late_c["events"])
    assert len(early_x["events"]) == len(late_x["events"])

    late_index = meta["late"]["target_index"]
    assert late_index == len(_ordered_events(late_c)) - 2
    assert _ordered_events(late_c)[late_index]["id"] == TARGET_EVENT_ID
    assert _ordered_events(late_x)[late_index]["id"] == REPLACEMENT_EVENT_ID


def test_late_arm_changes_only_target_and_displaced_event_ages():
    early_c, early_x, early = build_timing_arm("EARLY")
    late_c, late_x, late = build_timing_arm("LATE")

    early_c_order = _ordered_events(early_c)
    early_x_order = _ordered_events(early_x)
    late_index = late["target_index"]

    displaced_c_id = early_c_order[late_index]["id"]
    displaced_x_id = early_x_order[late_index]["id"]

    early_c_by_id, late_c_by_id = _by_id(early_c), _by_id(late_c)
    early_x_by_id, late_x_by_id = _by_id(early_x), _by_id(late_x)

    assert late_c_by_id[TARGET_EVENT_ID]["age"] == early_c_by_id[displaced_c_id]["age"]
    assert late_c_by_id[displaced_c_id]["age"] == early_c_by_id[TARGET_EVENT_ID]["age"]
    assert late_x_by_id[REPLACEMENT_EVENT_ID]["age"] == early_x_by_id[displaced_x_id]["age"]
    assert late_x_by_id[displaced_x_id]["age"] == early_x_by_id[REPLACEMENT_EVENT_ID]["age"]

    for event_id, event in early_c_by_id.items():
        if event_id not in (TARGET_EVENT_ID, displaced_c_id):
            assert late_c_by_id[event_id] == event
    for event_id, event in early_x_by_id.items():
        if event_id not in (REPLACEMENT_EVENT_ID, displaced_x_id):
            assert late_x_by_id[event_id] == event

    assert _ordered_events(late_c)[late_index]["id"] == TARGET_EVENT_ID
    assert _ordered_events(late_x)[late_index]["id"] == REPLACEMENT_EVENT_ID
    assert early["target_index"] != late["target_index"]
