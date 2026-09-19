from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from attractor_net import ExperienceEncoder, PlasticRecurrentAttractorNet, DevelopmentalWorld, load_world, load_probes, run_probe_battery, phenotype_similarity
from attractor_net.evaluation import phenotype_vector

ROOT = Path(__file__).resolve().parent
FOUNDER_SEED = 1842
EXPERIENCE_SEEDS = list(range(31001, 31013))
TICKS = 15_000
CONTEXT_CFG = {"policy_update_rule": "value_prediction", "policy_context_gain": 1.0, "policy_semantic_gain": 0.0, "policy_affordance_gain": 0.0}


def load_cfg() -> Dict:
    cfg = json.loads((ROOT / "config/default.json").read_text(encoding="utf-8"))
    cfg["network"]["neurons"] = 1024
    cfg["development"]["ticks"] = TICKS
    cfg["experiment"]["founder_seed"] = FOUNDER_SEED
    cfg["experiment"]["experience_seeds"] = EXPERIENCE_SEEDS
    return cfg


def reward_matched_world() -> Dict:
    world = deepcopy(load_world(ROOT / "data/development/kurzweil_canonical.json"))
    replacement = json.loads((ROOT / "data/v0_4/reward_matched_early_event.json").read_text(encoding="utf-8"))
    idx = next(i for i, e in enumerate(world["events"]) if e["id"] == "early_computer_access")
    world["events"][idx] = replacement
    world["condition"] = "reward_matched"
    return world


def condition_payloads() -> Dict[str, Dict]:
    return {
        "canonical": load_world(ROOT / "data/development/kurzweil_canonical.json"),
        "matched": load_world(ROOT / "data/development/matched_creative_humanities.json"),
        "reward_matched": reward_matched_world(),
        "compatibility": load_world(ROOT / "data/development/kurzweil_canonical.json"),
    }


def state_digest(net: PlasticRecurrentAttractorNet) -> str:
    h = hashlib.sha256()
    for a in (net.W.data, net.v, net.rate, net.policy_w, net.policy_context_w, net.policy_semantic_w, net.policy_affordance_w, net.policy_b):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(str(net.tick).encode())
    return h.hexdigest()


def run_one(condition: str, world_payload: Dict, seed: int, smoke: bool = False) -> Dict:
    cfg = load_cfg()
    if smoke:
        cfg["network"]["neurons"] = 192
        cfg["network"]["avg_recurrent_degree"] = 8
        cfg["network"]["action_population_size"] = 8
        cfg["development"]["ticks"] = 480
        cfg["evaluation"]["settle_ticks"] = 2
        cfg["evaluation"]["probe_ticks"] = 3
    if condition != "compatibility":
        cfg["network"].update(CONTEXT_CFG)
    else:
        cfg["network"]["policy_update_rule"] = "policy_gradient"
        cfg["network"]["policy_context_gain"] = 0.0
        cfg["network"]["policy_semantic_gain"] = 0.0
        cfg["network"]["policy_affordance_gain"] = 0.0

    enc = ExperienceEncoder(cfg["network"]["sensory_dim"])
    net = PlasticRecurrentAttractorNet(cfg["network"], enc, seed=FOUNDER_SEED)
    net.set_experience_seed(seed)
    world = DevelopmentalWorld(world_payload, enc)
    trace = world.run(net, total_ticks=cfg["development"]["ticks"], input_noise=cfg["development"]["input_noise"], decision_interval=cfg["development"]["decision_interval"])
    probes = load_probes(ROOT / "evaluation/adult_phenotype_probes.json")
    results = run_probe_battery(net, enc, probes, settle_ticks=cfg["evaluation"]["settle_ticks"], probe_ticks=cfg["evaluation"]["probe_ticks"])
    decisions = [{"index": i, "event_id": d["event_id"], "age": d["age"], "action": d["action"], "reward": d["reward"], "action_probabilities": d["action_probabilities"]} for i, d in enumerate(trace.decisions)]
    return {"condition": condition, "founder_seed": FOUNDER_SEED, "experience_seed": seed, "development_ticks": trace.ticks, "state_digest": state_digest(net), "phenotype_similarity": phenotype_similarity(results), "phenotype_vector": phenotype_vector(results).tolist(), "final_probes": results, "trajectory": decisions}


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen v0.2 contextual-learning experiment")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "results/v0_2")
    args = p.parse_args()
    seeds = EXPERIENCE_SEEDS[: (args.seeds if args.seeds is not None else (1 if args.smoke else len(EXPERIENCE_SEEDS)))]
    worlds = condition_payloads()
    args.out.mkdir(parents=True, exist_ok=True)
    runs: List[Dict] = []
    for seed in seeds:
        for condition, payload in worlds.items():
            run = run_one(condition, payload, seed, smoke=args.smoke)
            runs.append(run)
            (args.out / f"{condition}_{seed}.json").write_text(json.dumps(run, indent=2, sort_keys=True), encoding="utf-8")
    if not args.smoke:
        repeat = run_one("canonical", worlds["canonical"], seeds[0], smoke=False)
        first = next(r for r in runs if r["condition"] == "canonical" and r["experience_seed"] == seeds[0])
        if repeat["state_digest"] != first["state_digest"] or repeat["trajectory"] != first["trajectory"]:
            raise RuntimeError("identical-history determinism gate failed")
    summary = {"protocol": "V0_2_PREREGISTRATION.md", "founder_seed": FOUNDER_SEED, "experience_seeds": seeds, "ticks": 480 if args.smoke else TICKS, "runs": runs}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
