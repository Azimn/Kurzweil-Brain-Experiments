from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from attractor_net import ExperienceEncoder, PlasticRecurrentAttractorNet, load_probes, run_probe_battery
from run_v0_2 import EXPERIENCE_SEEDS, FOUNDER_SEED, TICKS, CONTEXT_CFG, condition_payloads, load_cfg, run_one

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_v02_constants():
    cfg = load_cfg()
    assert FOUNDER_SEED == 1842
    assert EXPERIENCE_SEEDS == list(range(31001, 31013))
    assert TICKS == 15000
    assert cfg["network"]["neurons"] == 1024
    assert cfg["experiment"]["experience_seeds"] == EXPERIENCE_SEEDS
    assert CONTEXT_CFG == {"policy_update_rule": "value_prediction", "policy_context_gain": 1.0, "policy_semantic_gain": 0.0, "policy_affordance_gain": 0.0}


def test_reward_matched_world_changes_only_preregistered_event():
    worlds = condition_payloads()
    canonical = worlds["canonical"]
    matched = worlds["reward_matched"]
    assert len(canonical["events"]) == len(matched["events"])
    for ce, me in zip(canonical["events"], matched["events"]):
        if ce["id"] != "early_computer_access":
            assert ce == me
        else:
            assert ce["age"] == me["age"]
            assert ce["exposure_weight"] == me["exposure_weight"]
            assert ce["scalars"] == me["scalars"]
            assert ce["available_actions"] == me["available_actions"]
            assert {a: ce["consequences"][a]["reward"] for a in ce["available_actions"]} == {a: me["consequences"][a]["reward"] for a in me["available_actions"]}
            assert ce["situation"] != me["situation"]


def test_smoke_identical_history_is_deterministic_and_isolated(tmp_path):
    world = condition_payloads()["canonical"]
    before = deepcopy(world)
    a = run_one("canonical", world, 31001, smoke=True)
    b = run_one("canonical", world, 31001, smoke=True)
    assert world == before
    assert a["state_digest"] == b["state_digest"]
    assert a["trajectory"] == b["trajectory"]
    assert a["phenotype_vector"] == b["phenotype_vector"]
    assert a["development_ticks"] == 480


def test_trajectory_contains_every_preregistered_audit_field():
    run = run_one("canonical", condition_payloads()["canonical"], 31001, smoke=True)
    assert run["trajectory"]
    required = {
        "tick",
        "condition",
        "experience_seed",
        "event_id",
        "allowed_actions",
        "action",
        "reward",
        "action_probabilities",
        "contextual_action_values",
    }
    for decision in run["trajectory"]:
        assert required <= set(decision)
        assert decision["condition"] == "canonical"
        assert decision["experience_seed"] == 31001
        assert set(decision["action_probabilities"]) == set(decision["allowed_actions"])
        assert set(decision["contextual_action_values"]) == set(decision["allowed_actions"])


def test_evaluation_restores_all_mutable_network_state():
    cfg = load_cfg()
    cfg["network"]["neurons"] = 192
    cfg["network"]["avg_recurrent_degree"] = 8
    cfg["network"]["action_population_size"] = 8
    enc = ExperienceEncoder(cfg["network"]["sensory_dim"])
    net = PlasticRecurrentAttractorNet(cfg["network"], enc, seed=FOUNDER_SEED)
    net.set_experience_seed(31001)
    net.step(np.zeros(enc.input_dim, dtype=np.float32), reward=0.0, learn=False)
    before = {
        "W": net.W.data.copy(),
        "v": net.v.copy(),
        "rate": net.rate.copy(),
        "policy_w": net.policy_w.copy(),
        "policy_context_w": net.policy_context_w.copy(),
        "policy_semantic_w": net.policy_semantic_w.copy(),
        "policy_affordance_w": net.policy_affordance_w.copy(),
        "policy_b": net.policy_b.copy(),
        "last_context": net.last_context.copy(),
        "last_semantic": net.last_semantic.copy(),
        "last_affordance": net.last_affordance.copy(),
        "tick": net.tick,
        "rng": deepcopy(net.rng.bit_generator.state),
    }
    probes = load_probes(ROOT / "evaluation/adult_phenotype_probes.json")
    run_probe_battery(net, enc, probes, settle_ticks=2, probe_ticks=3)
    assert np.array_equal(net.W.data, before["W"])
    assert np.array_equal(net.v, before["v"])
    assert np.array_equal(net.rate, before["rate"])
    assert np.array_equal(net.policy_w, before["policy_w"])
    assert np.array_equal(net.policy_context_w, before["policy_context_w"])
    assert np.array_equal(net.policy_semantic_w, before["policy_semantic_w"])
    assert np.array_equal(net.policy_affordance_w, before["policy_affordance_w"])
    assert np.array_equal(net.policy_b, before["policy_b"])
    assert np.array_equal(net.last_context, before["last_context"])
    assert np.array_equal(net.last_semantic, before["last_semantic"])
    assert np.array_equal(net.last_affordance, before["last_affordance"])
    assert net.tick == before["tick"]
    assert net.rng.bit_generator.state == before["rng"]


def test_compatibility_smoke_is_deterministic_and_disables_new_channels():
    world = condition_payloads()["compatibility"]
    a = run_one("compatibility", world, 31001, smoke=True)
    b = run_one("compatibility", world, 31001, smoke=True)
    assert a["state_digest"] == b["state_digest"]
    assert a["trajectory"] == b["trajectory"]
    assert a["phenotype_vector"] == b["phenotype_vector"]
    cfg = load_cfg()
    assert cfg["network"].get("policy_update_rule", "policy_gradient") == "policy_gradient"
    assert float(cfg["network"].get("policy_context_gain", 0.0)) == 0.0
    assert float(cfg["network"].get("policy_semantic_gain", 0.0)) == 0.0
    assert float(cfg["network"].get("policy_affordance_gain", 0.0)) == 0.0


def test_preregistration_contains_anti_tuning_and_control_contract():
    text = (ROOT / "V0_2_PREREGISTRATION.md").read_text(encoding="utf-8")
    assert "Anti-tuning rule" in text
    assert "reward-matched" in text
    assert "identical-history determinism" in text
    assert "15,000 developmental ticks" in text
