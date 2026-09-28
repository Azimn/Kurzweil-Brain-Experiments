from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from attractor_net import ExperienceEncoder, PlasticRecurrentAttractorNet, DevelopmentalWorld
from v0_5_trajectory import (
    jensen_shannon_divergence,
    mean_absolute_value_distance,
    network_state_digest,
    restore_checkpoint,
    serialize_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    cfg = json.loads((ROOT / "config" / "default.json").read_text(encoding="utf-8"))["network"]
    cfg = copy.deepcopy(cfg)
    cfg["neurons"] = 128
    cfg["avg_recurrent_degree"] = 8
    cfg["action_population_size"] = 8
    return cfg


def _world():
    payload = json.loads((ROOT / "data" / "development" / "kurzweil_canonical.json").read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    payload["events"] = payload["events"][:3]
    return payload


def _net(encoder):
    net = PlasticRecurrentAttractorNet(_cfg(), encoder, seed=1842)
    net.set_experience_seed(31001)
    return net


def test_event_boundary_slices_equal_uninterrupted_execution():
    encoder = ExperienceEncoder(64)
    world = DevelopmentalWorld(_world(), encoder)
    whole = _net(encoder)
    whole_trace = world.run(whole, total_ticks=360, input_noise=0.01, decision_interval=24)

    sliced = _net(encoder)
    first = world.run(sliced, total_ticks=360, input_noise=0.01, decision_interval=24, event_start=0, event_stop=1)
    second = world.run(sliced, total_ticks=360, input_noise=0.01, decision_interval=24, event_start=1)

    assert network_state_digest(whole) == network_state_digest(sliced)
    assert whole_trace.decisions == first.decisions + second.decisions
    assert whole_trace.event_counts == {**first.event_counts, **second.event_counts}


def test_serialized_checkpoint_replays_exact_continuation():
    encoder = ExperienceEncoder(64)
    world = DevelopmentalWorld(_world(), encoder)
    original = _net(encoder)
    world.run(original, total_ticks=360, input_noise=0.01, decision_interval=24, event_start=0, event_stop=1)
    payload = serialize_checkpoint(original)
    restored = restore_checkpoint(payload)
    assert network_state_digest(original) == network_state_digest(restored)

    uninterrupted = world.run(original, total_ticks=360, input_noise=0.01, decision_interval=24, event_start=1)
    replayed = world.run(restored, total_ticks=360, input_noise=0.01, decision_interval=24, event_start=1)
    assert uninterrupted.decisions == replayed.decisions
    assert network_state_digest(original) == network_state_digest(restored)


def test_preregistered_trajectory_metrics_are_deterministic_and_symmetric():
    left = [
        {"event_id": "a", "action": "analyze", "contextual_action_values": {"analyze": 0.2, "communicate": 0.1}},
        {"event_id": "a", "action": "analyze", "contextual_action_values": {"analyze": 0.3, "communicate": 0.1}},
    ]
    right = [
        {"event_id": "a", "action": "communicate", "contextual_action_values": {"analyze": 0.1, "communicate": 0.4}},
        {"event_id": "a", "action": "communicate", "contextual_action_values": {"analyze": 0.1, "communicate": 0.5}},
    ]
    js_lr = jensen_shannon_divergence(left, right)
    js_rl = jensen_shannon_divergence(right, left)
    assert js_lr == js_rl
    assert js_lr > 0.0

    ltab = {"a": {"analyze": 0.3, "communicate": 0.1}}
    rtab = {"a": {"analyze": 0.1, "communicate": 0.5}, "b": {"analyze": 0.2}}
    assert mean_absolute_value_distance(ltab, rtab) == mean_absolute_value_distance(rtab, ltab)
    assert np.isclose(mean_absolute_value_distance(ltab, rtab), (0.2 + 0.4 + 0.2) / 3.0)
