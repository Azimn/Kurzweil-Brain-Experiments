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
    assert np.array_equal(a.action_populations["explore"], b.action_populations["explore"])


def test_evaluation_is_noninvasive():
    cfg = tiny_cfg()
    enc = ExperienceEncoder(cfg["network"]["sensory_dim"])
    net = PlasticRecurrentAttractorNet(cfg["network"], enc, seed=1842)
    probes = load_probes(ROOT / "evaluation" / "adult_phenotype_probes.json")[:2]
    before = (net.v.copy(), net.rate.copy(), net.W.data.copy(), net.policy_w.copy(), net.tick)
    run_probe_battery(net, enc, probes, settle_ticks=2, probe_ticks=3)
    after = (net.v, net.rate, net.W.data, net.policy_w, net.tick)
    assert np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])
    assert np.array_equal(before[2], after[2])
    assert np.array_equal(before[3], after[3])
    assert before[4] == after[4]


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
