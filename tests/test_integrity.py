from copy import deepcopy
from pathlib import Path
import json
import numpy as np

from attractor_net import ExperienceEncoder, PlasticRecurrentAttractorNet, DevelopmentalWorld
from attractor_net.evaluation import load_probes, run_probe_battery

ROOT = Path(__file__).resolve().parents[1]


def tiny_cfg():
    cfg = json.loads((ROOT / "config" / "default.json").read_text())
    cfg["network"]["neurons"] = 192
    cfg["network"]["avg_recurrent_degree"] = 8
    cfg["network"]["action_population_size"] = 8
    return cfg


def test_same_seed_means_same_newborn():
    cfg = tiny_cfg()
    enc = ExperienceEncoder(cfg["network"]["sensory_dim"])
    a = PlasticRecurrentAttractorNet(cfg["network"], enc, seed=1842)
    b = PlasticRecurrentAttractorNet(cfg["network"], enc, seed=1842)
    assert np.array_equal(a.W.data, b.W.data)
    assert np.array_equal(a.policy_w, b.policy_w)
    assert np.array_equal(a.policy_context_w, b.policy_context_w)
    assert np.array_equal(a.policy_semantic_w, b.policy_semantic_w)
    assert np.array_equal(a.policy_affordance_w, b.policy_affordance_w)
    assert np.count_nonzero(a.policy_context_w) == 0
    assert np.count_nonzero(a.policy_semantic_w) == 0
    assert np.count_nonzero(a.policy_affordance_w) == 0
    assert np.array_equal(a.action_populations["explore"], b.action_populations["explore"])


def test_evaluation_is_noninvasive():
    cfg = tiny_cfg()
    enc = ExperienceEncoder(cfg["network"]["sensory_dim"])
    net = PlasticRecurrentAttractorNet(cfg["network"], enc, seed=1842)
    probes = load_probes(ROOT / "evaluation" / "adult_phenotype_probes.json")[:2]
    before = (net.v.copy(), net.rate.copy(), net.W.data.copy(), net.policy_w.copy(), net.policy_context_w.copy(), net.policy_semantic_w.copy(), net.policy_affordance_w.copy(), net.last_context.copy(), net.last_semantic.copy(), net.last_affordance.copy(), net.tick)
    run_probe_battery(net, enc, probes, settle_ticks=2, probe_ticks=3)
    after = (net.v, net.rate, net.W.data, net.policy_w, net.policy_context_w, net.policy_semantic_w, net.policy_affordance_w, net.last_context, net.last_semantic, net.last_affordance, net.tick)
    assert np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])
    assert np.array_equal(before[2], after[2])
    assert np.array_equal(before[3], after[3])
    assert np.array_equal(before[4], after[4])
    assert np.array_equal(before[5], after[5])
    assert np.array_equal(before[6], after[6])
    assert np.array_equal(before[7], after[7])
    assert np.array_equal(before[8], after[8])
    assert np.array_equal(before[9], after[9])
    assert before[10] == after[10]


def test_no_historical_action_labels_in_development():
    for f in (ROOT / "data" / "development").glob("*.json"):
        data = json.loads(f.read_text())
        for e in data["events"]:
            assert "action" not in e
            assert "target_action_weights" not in e


def test_world_can_run_without_answer_key():
    cfg = tiny_cfg()
    enc = ExperienceEncoder(cfg["network"]["sensory_dim"])
    net = PlasticRecurrentAttractorNet(cfg["network"], enc, seed=1842)
    world_data = json.loads((ROOT / "data" / "development" / "kurzweil_canonical.json").read_text())
    world_data = deepcopy(world_data)
    world_data["events"] = world_data["events"][:1]
    world = DevelopmentalWorld(world_data, enc)
    trace = world.run(net, total_ticks=96, input_noise=0.0, decision_interval=24)
    assert trace.ticks == 96
    assert len(trace.decisions) == 4


def test_reward_matched_ablation_preserves_age12_structure():
    canonical = json.loads((ROOT / "data" / "development" / "kurzweil_canonical.json").read_text())
    replacement = json.loads((ROOT / "data" / "v0_4" / "reward_matched_early_event.json").read_text())
    ce = next(e for e in canonical["events"] if e["id"] == "early_computer_access")
    ae = replacement
    assert ce["exposure_weight"] == ae["exposure_weight"]
    assert ce["scalars"] == ae["scalars"]
    assert ce["available_actions"] == ae["available_actions"]
    assert {a: ce["consequences"][a]["reward"] for a in ce["available_actions"]} == {a: ae["consequences"][a]["reward"] for a in ae["available_actions"]}
    assert ce["situation"] != ae["situation"]


def test_value_prediction_has_diminishing_error():
    cfg = tiny_cfg()
    cfg["network"]["policy_update_rule"] = "value_prediction"
    cfg["network"]["policy_context_gain"] = 0.0
    cfg["network"]["policy_semantic_gain"] = 0.0
    cfg["network"]["policy_affordance_gain"] = 0.0
    enc = ExperienceEncoder(cfg["network"]["sensory_dim"])
    net = PlasticRecurrentAttractorNet(cfg["network"], enc, seed=1842)
    features = net._policy_features()
    context = net._policy_context_features()
    semantic = net._policy_semantic_features()
    affordance = net._policy_affordance_features()
    probs = {"create": 1.0}
    ai = list(net.action_populations).index("create")
    before = float(net.policy_w[ai].dot(features) + net.policy_b[ai])
    for _ in range(100):
        net.reinforce_action("create", 0.5, probs, features, context, semantic, affordance)
    after = float(net.policy_w[ai].dot(features) + net.policy_b[ai])
    assert abs(0.5 - after) < abs(0.5 - before)
    assert after < 0.75
