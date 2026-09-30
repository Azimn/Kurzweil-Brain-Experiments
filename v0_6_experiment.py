from __future__ import annotations

import hashlib
import json
from pathlib import Path

from attractor_net import DevelopmentalWorld
from run_v0_5_causal import _config, _new_runtime
from v0_5_experiment import REPLACEMENT_EVENT_ID, TARGET_EVENT_ID, validate_world_pair
from v0_5_trajectory import (
    encountered_context_value_table,
    jensen_shannon_divergence,
    mean_absolute_value_distance,
    network_state_digest,
    serialize_checkpoint,
)
from v0_6_timing import _ordered_events, build_timing_arm, validate_timing_arms

SEEDS = tuple(range(31001, 31013))


def _run_timing_pair(canonical, counterfactual, experience_seed, cfg, founder_seed):
    ordered_c = _ordered_events(canonical)
    ordered_x = _ordered_events(counterfactual)
    ci = next(i for i, event in enumerate(ordered_c) if event["id"] == TARGET_EVENT_ID)
    xi = next(i for i, event in enumerate(ordered_x) if event["id"] == REPLACEMENT_EVENT_ID)
    if ci != xi:
        raise RuntimeError("timing arm intervention positions differ")

    enc_c, net_c = _new_runtime(cfg, founder_seed, experience_seed)
    enc_x, net_x = _new_runtime(cfg, founder_seed, experience_seed)
    world_c = DevelopmentalWorld(canonical, enc_c)
    world_x = DevelopmentalWorld(counterfactual, enc_x)
    kwargs = {
        "total_ticks": int(cfg["development"]["ticks"]),
        "input_noise": cfg["development"]["input_noise"],
        "decision_interval": cfg["development"]["decision_interval"],
    }

    pre_c = world_c.run(net_c, event_start=0, event_stop=ci, **kwargs)
    pre_x = world_x.run(net_x, event_start=0, event_stop=xi, **kwargs)
    pre_bytes_c = serialize_checkpoint(net_c)
    pre_bytes_x = serialize_checkpoint(net_x)
    if pre_bytes_c != pre_bytes_x or pre_c.decisions != pre_x.decisions:
        raise RuntimeError(f"pre-intervention identity failure for seed {experience_seed}")

    checkpoints = []
    for idx in range(ci, len(ordered_c)):
        tc = world_c.run(net_c, event_start=idx, event_stop=idx + 1, **kwargs)
        tx = world_x.run(net_x, event_start=idx, event_stop=idx + 1, **kwargs)
        checkpoints.append(
            {
                "event_index": idx,
                "canonical_event_id": ordered_c[idx]["id"],
                "counterfactual_event_id": ordered_x[idx]["id"],
                "canonical_state_sha256": network_state_digest(net_c),
                "counterfactual_state_sha256": network_state_digest(net_x),
                "value_distance": mean_absolute_value_distance(
                    encountered_context_value_table(tc.decisions),
                    encountered_context_value_table(tx.decisions),
                ),
                "action_js": jensen_shannon_divergence(tc.decisions, tx.decisions),
            }
        )

    return {
        "pre_intervention_state_sha256": network_state_digest(net_c) if not checkpoints else hashlib.sha256(pre_bytes_c).hexdigest(),
        "pre_intervention_serialization_sha256": hashlib.sha256(pre_bytes_c).hexdigest(),
        "intervention_index": ci,
        "trajectory": checkpoints,
        "mature": checkpoints[-1],
    }


def run():
    validate_world_pair()
    timing_meta = validate_timing_arms()
    cfg, configured_seeds, founder_seed = _config()
    if tuple(configured_seeds) != SEEDS:
        raise RuntimeError("v0.6 experience-seed drift")

    result = {
        "status": "exploratory_unreviewed",
        "timing": timing_meta,
        "founder_seed": founder_seed,
        "seeds": [],
    }
    for seed in SEEDS:
        row = {"seed": seed, "arms": {}}
        for timing in ("EARLY", "LATE"):
            canonical, counterfactual, meta = build_timing_arm(timing)
            pair = _run_timing_pair(canonical, counterfactual, seed, cfg, founder_seed)
            pair["meta"] = meta
            row["arms"][timing] = pair
        result["seeds"].append(row)

    out = Path("artifacts/v0_6_timing")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
