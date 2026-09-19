from copy import deepcopy
import json
from pathlib import Path

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


def test_preregistration_contains_anti_tuning_and_control_contract():
    text = (ROOT / "V0_2_PREREGISTRATION.md").read_text(encoding="utf-8")
    assert "Anti-tuning rule" in text
    assert "reward-matched" in text
    assert "identical-history determinism" in text
    assert "15,000 developmental ticks" in text
