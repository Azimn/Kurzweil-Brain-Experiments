from __future__ import annotations

import copy
import json
from pathlib import Path

from attractor_net import DevelopmentalWorld
from run_v0_5_causal import _config, _new_runtime
from v0_5_experiment import validate_world_pair
from v0_5_trajectory import (
    encountered_context_value_table,
    jensen_shannon_divergence,
    mean_absolute_value_distance,
)
from v0_6_timing import build_timing_arm, validate_timing_arms

SEEDS = tuple(range(31001, 31013))


def _run(world, experience_seed, cfg, founder_seed):
    encoder, net = _new_runtime(cfg, founder_seed, experience_seed)
    trace = DevelopmentalWorld(world, encoder).run(
        net,
        total_ticks=int(cfg["development"]["ticks"]),
        input_noise=cfg["development"]["input_noise"],
        decision_interval=cfg["development"]["decision_interval"],
    )
    return trace.decisions, net


def run():
    validate_world_pair()
    timing_meta = validate_timing_arms()
    cfg, configured_seeds, founder_seed = _config()
    if tuple(configured_seeds) != SEEDS:
        raise RuntimeError("v0.6 experience-seed drift")
    result = {"status":"exploratory_unreviewed","timing":timing_meta,"founder_seed":founder_seed,"seeds":[]}
    for seed in SEEDS:
        row={"seed":seed,"arms":{}}
        for timing in ("EARLY","LATE"):
            canonical, counterfactual, meta = build_timing_arm(timing)
            left, _ = _run(canonical, seed, cfg, founder_seed)
            right, _ = _run(counterfactual, seed, cfg, founder_seed)
            intervention_index = meta["target_index"]
            checkpoints=[]
            for i in range(intervention_index, len(left)):
                lv=encountered_context_value_table(left[:i+1])
                rv=encountered_context_value_table(right[:i+1])
                checkpoints.append({
                    "event_index":i,
                    "event_id":canonical["events"][i]["id"],
                    "value_distance":mean_absolute_value_distance(lv,rv),
                    "action_js":jensen_shannon_divergence(left[:i+1],right[:i+1]),
                })
            row["arms"][timing]={"meta":meta,"trajectory":checkpoints,"mature":checkpoints[-1]}
        result["seeds"].append(row)
    out=Path("artifacts/v0_6_timing")
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")


if __name__=="__main__":
    run()
