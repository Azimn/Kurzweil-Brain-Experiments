from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from attractor_net import ExperienceEncoder, PlasticRecurrentAttractorNet, DevelopmentalWorld, load_probes, run_probe_battery
from attractor_net.evaluation import phenotype_vector
from v0_5_experiment import build_world_pair, validate_world_pair, TARGET_EVENT_ID, REPLACEMENT_EVENT_ID
from v0_5_trajectory import (
    action_counts,
    encountered_context_value_table,
    jensen_shannon_divergence,
    make_checkpoint,
    mean_absolute_value_distance,
    network_state_digest,
    restore_checkpoint,
    serialize_checkpoint,
)

ROOT = Path(__file__).resolve().parent
EXTENDED = ROOT / "config" / "extended_2026-09-17.json"
DEFAULT = ROOT / "config" / "default.json"
PROBES = ROOT / "evaluation" / "adult_phenotype_probes.json"
OUT = ROOT / "results" / "v0_5"


def _config() -> Tuple[Dict[str, Any], List[int], int]:
    cfg = json.loads(DEFAULT.read_text(encoding="utf-8"))
    ext = json.loads(EXTENDED.read_text(encoding="utf-8"))
    cfg["network"].update(ext["network_overrides"])
    cfg["development"].update(ext["development_overrides"])
    return cfg, [int(x) for x in ext["experiment"]["experience_seeds"]], int(ext["experiment"]["founder_seed"])


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _new_runtime(cfg: Dict[str, Any], founder_seed: int, experience_seed: int) -> Tuple[ExperienceEncoder, PlasticRecurrentAttractorNet]:
    encoder = ExperienceEncoder(cfg["network"]["sensory_dim"])
    net = PlasticRecurrentAttractorNet(copy.deepcopy(cfg["network"]), encoder, seed=founder_seed)
    net.set_experience_seed(experience_seed)
    return encoder, net


def _reward(decisions: List[Dict[str, Any]]) -> float:
    return float(sum(float(row["reward"]) for row in decisions))


def _evaluation(net: PlasticRecurrentAttractorNet, encoder: ExperienceEncoder, cfg: Dict[str, Any]) -> Dict[str, Any]:
    before = network_state_digest(net)
    probes = load_probes(PROBES)
    result = run_probe_battery(net, encoder, probes, settle_ticks=cfg["evaluation"]["settle_ticks"], probe_ticks=cfg["evaluation"]["probe_ticks"])
    after = network_state_digest(net)
    if before != after:
        raise RuntimeError("adult evaluation mutated learned/runtime state")
    return {"state_sha256": before, "phenotype_vector": phenotype_vector(result).tolist(), "probes": result}


def _run_pair(seed: int, cfg: Dict[str, Any], founder_seed: int, canonical: Dict[str, Any], counterfactual: Dict[str, Any]) -> Dict[str, Any]:
    ci = next(i for i, e in enumerate(canonical["events"]) if e["id"] == TARGET_EVENT_ID)
    xi = next(i for i, e in enumerate(counterfactual["events"]) if e["id"] == REPLACEMENT_EVENT_ID)
    if ci != xi:
        raise RuntimeError("intervention event positions differ")

    enc_c, net_c = _new_runtime(cfg, founder_seed, seed)
    enc_x, net_x = _new_runtime(cfg, founder_seed, seed)
    world_c = DevelopmentalWorld(canonical, enc_c)
    world_x = DevelopmentalWorld(counterfactual, enc_x)
    ticks = int(cfg["development"]["ticks"])
    kwargs = {"total_ticks": ticks, "input_noise": cfg["development"]["input_noise"], "decision_interval": cfg["development"]["decision_interval"]}

    pre_c = world_c.run(net_c, event_start=0, event_stop=ci, **kwargs)
    pre_x = world_x.run(net_x, event_start=0, event_stop=xi, **kwargs)
    pre_bytes_c = serialize_checkpoint(net_c)
    pre_bytes_x = serialize_checkpoint(net_x)
    if pre_bytes_c != pre_bytes_x or pre_c.decisions != pre_x.decisions:
        raise RuntimeError(f"pre-intervention identity failure for seed {seed}")

    replay = restore_checkpoint(pre_bytes_c)
    replay_trace = world_c.run(replay, event_start=ci, event_stop=ci + 1, **kwargs)
    original_probe = restore_checkpoint(pre_bytes_c)
    original_trace = world_c.run(original_probe, event_start=ci, event_stop=ci + 1, **kwargs)
    if network_state_digest(replay) != network_state_digest(original_probe) or replay_trace.decisions != original_trace.decisions:
        raise RuntimeError(f"serialization/replay failure for seed {seed}")

    cumulative_c = _reward(pre_c.decisions)
    cumulative_x = _reward(pre_x.decisions)
    checkpoints: List[Dict[str, Any]] = []
    for idx in range(ci, len(canonical["events"])):
        tc = world_c.run(net_c, event_start=idx, event_stop=idx + 1, **kwargs)
        tx = world_x.run(net_x, event_start=idx, event_stop=idx + 1, **kwargs)
        cumulative_c += _reward(tc.decisions)
        cumulative_x += _reward(tx.decisions)
        cc = make_checkpoint(canonical["events"][idx]["id"], tc.decisions, net_c, cumulative_c)
        cx = make_checkpoint(counterfactual["events"][idx]["id"], tx.decisions, net_x, cumulative_x)
        checkpoints.append({
            "event_index": idx,
            "canonical": cc,
            "counterfactual": cx,
            "value_distance": mean_absolute_value_distance(encountered_context_value_table(tc.decisions), encountered_context_value_table(tx.decisions)),
            "action_js_divergence": jensen_shannon_divergence(tc.decisions, tx.decisions),
        })

    eval_c = _evaluation(net_c, enc_c, cfg)
    eval_x = _evaluation(net_x, enc_x, cfg)
    return {
        "experience_seed": seed,
        "founder_seed": founder_seed,
        "pre_intervention_state_sha256": network_state_digest(restore_checkpoint(pre_bytes_c)),
        "pre_intervention_serialization_sha256": hashlib.sha256(pre_bytes_c).hexdigest(),
        "checkpoints": checkpoints,
        "canonical_evaluation": eval_c,
        "counterfactual_evaluation": eval_x,
    }


def _identical_history_gate(seed: int, cfg: Dict[str, Any], founder_seed: int, canonical: Dict[str, Any]) -> None:
    enc_a, a = _new_runtime(cfg, founder_seed, seed)
    enc_b, b = _new_runtime(cfg, founder_seed, seed)
    wa, wb = DevelopmentalWorld(canonical, enc_a), DevelopmentalWorld(canonical, enc_b)
    kwargs = {"total_ticks": int(cfg["development"]["ticks"]), "input_noise": cfg["development"]["input_noise"], "decision_interval": cfg["development"]["decision_interval"]}
    ta, tb = wa.run(a, **kwargs), wb.run(b, **kwargs)
    if ta.decisions != tb.decisions or network_state_digest(a) != network_state_digest(b):
        raise RuntimeError("identical-history determinism gate failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen v0.5 paired causal trajectory experiment")
    parser.add_argument("--preflight", action="store_true", help="Run structural gates only, without scientific 15,000-tick paired outcomes")
    args = parser.parse_args()
    validate_world_pair()
    cfg, seeds, founder_seed = _config()
    canonical, counterfactual = build_world_pair()
    if len(seeds) != 12 or int(cfg["network"]["neurons"]) != 1024 or int(cfg["development"]["ticks"]) != 15000:
        raise RuntimeError("frozen v0.5 configuration drift")
    if args.preflight:
        print("v0.5 preflight: descriptor/configuration gates OK; scientific outcomes not executed")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    _identical_history_gate(seeds[0], cfg, founder_seed, canonical)
    manifest = {"commit_sha": _git_sha(), "founder_seed": founder_seed, "experience_seeds": seeds, "neurons": 1024, "ticks_per_life": 15000, "protocol": "V0_5_PREREGISTRATION.md", "pairs": []}
    for seed in seeds:
        result = _run_pair(seed, cfg, founder_seed, canonical, counterfactual)
        path = OUT / f"seed_{seed}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        manifest["pairs"].append({"seed": seed, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
