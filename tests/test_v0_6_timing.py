from v0_6_timing import build_timing_arm, event_multiset, validate_timing_arms


def test_timing_arms_preserve_event_multisets():
    meta = validate_timing_arms()
    early_c, early_x, _ = build_timing_arm("EARLY")
    late_c, late_x, _ = build_timing_arm("LATE")
    assert event_multiset(early_c) == event_multiset(late_c)
    assert event_multiset(early_x) == event_multiset(late_x)
    assert meta["late"]["target_index"] == len(late_c["events"]) - 2


def test_late_arm_is_atomic_position_swap():
    early_c, early_x, early = build_timing_arm("EARLY")
    late_c, late_x, late = build_timing_arm("LATE")
    i, j = early["target_index"], late["target_index"]
    assert late_c["events"][j] == early_c["events"][i]
    assert late_c["events"][i] == early_c["events"][j]
    assert late_x["events"][j] == early_x["events"][i]
    assert late_x["events"][i] == early_x["events"][j]
    for k in range(len(early_c["events"])):
        if k not in (i, j):
            assert late_c["events"][k] == early_c["events"][k]
            assert late_x["events"][k] == early_x["events"][k]
