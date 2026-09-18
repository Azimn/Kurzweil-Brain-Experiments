from __future__ import annotations

import argparse
from pathlib import Path
import json
import time
from typing import Dict, List

import numpy as np

from attractor_net import (
    ExperienceEncoder,
    PlasticRecurrentAttractorNet,
    DevelopmentalWorld,
    load_world,
    load_probes,
    run_probe_battery,
    phenotype_similarity,
)
from attractor_net.evaluation import phenotype_vector

ROOT = Path(__file__).resolve().parent

CONDITIONS = {
    "canonical": ROOT / "data" / "development" / "kurzweil_canonical.json",
    "matched": ROOT / "data" / "development" / "matched_creative_humanities.json",
    "computer_ablation": ROOT / "data" / "development" / "computer_access_ablation.json",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired developmental-attractor experiment using a Kurzweil-derived environmental history."
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Very small validation run.")
    mode.add_argument("--full", action="store_true", help="20 seeds, 4096 neurons, 100k ticks per life.")
    p.add_argument("--neurons", type=int, default=None)
    p.add_argument("--ticks", type=int, default=None)
    p.add_argument("--seeds", type=int, default=None, help="Use the first N deterministic seeds.")
    p.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted(CONDITIONS),
        default=list(CONDITIONS),
        help="Conditions to run. Canonical should normally remain included.",
    )
    p.add_argument("--no-trajectory", action="store_true", help="Skip phenotype probes after each developmental event.")
    return p.parse_args()


def dispersion(vectors: List[np.ndarray]) -> float:
    if len(vectors) <= 1:
        return 0.0
    mat = np.vstack(vectors)
    centroid = mat.mean(axis=0)
    return float(np.mean(np.linalg.norm(mat - centroid, axis=1) / np.sqrt(mat.shape[1])))


def pairwise_distance(vectors: List[np.ndarray]) -> float:
    if len(vectors) <= 1:
        return 0.0
    vals = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            vals.append(float(np.linalg.norm(vectors[i] - vectors[j]) / np.sqrt(vectors[i].size)))
    return float(np.mean(vals)) if vals else 0.0


def summarize_decisions(decisions: List[Dict]) -> Dict:
    actions: Dict[str, int] = {}
    rewards: List[float] = []
    by_event: Dict[str, Dict[str, int]] = {}
    for d in decisions:
        a = d["action"]
        actions[a] = actions.get(a, 0) + 1
        rewards.append(float(d["reward"]))
        ev = by_event.setdefault(d["event_id"], {})
        ev[a] = ev.get(a, 0) + 1
    return {
        "action_counts": actions,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "by_event": by_event,
    }


def run_one(condition_name: str, world_payload: Dict, cfg: Dict, founder_seed: int, experience_seed: int, probes: List[Dict], trajectory: bool) -> Dict:
    encoder = ExperienceEncoder(cfg["network"]["sensory_dim"])
    net = PlasticRecurrentAttractorNet(cfg["network"], encoder, seed=founder_seed)
    net.set_experience_seed(experience_seed)
    world = DevelopmentalWorld(world_payload, encoder)

    initial_probes = run_probe_battery(
        net, encoder, probes,
        settle_ticks=cfg["evaluation"]["settle_ticks"],
        probe_ticks=cfg["evaluation"]["probe_ticks"],
    )
    initial_vector = phenotype_vector(initial_probes)
    checkpoints = []

    def checkpoint(event: Dict, current_net: PlasticRecurrentAttractorNet) -> None:
        if not trajectory:
            return
        res = run_probe_battery(
            current_net, encoder, probes,
            settle_ticks=cfg["evaluation"]["settle_ticks"],
            probe_ticks=cfg["evaluation"]["probe_ticks"],
        )
        checkpoints.append({
            "event_id": event["id"],
            "age": event["age"],
            "phenotype_similarity": phenotype_similarity(res),
            "phenotype_vector": phenotype_vector(res).tolist(),
        })

    start = time.time()
    trace = world.run(
        net,
        total_ticks=cfg["development"]["ticks"],
        input_noise=cfg["development"]["input_noise"],
        decision_interval=cfg["development"]["decision_interval"],
        on_event_complete=checkpoint,
    )
    elapsed = time.time() - start

    final_probes = run_probe_battery(
        net, encoder, probes,
        settle_ticks=cfg["evaluation"]["settle_ticks"],
        probe_ticks=cfg["evaluation"]["probe_ticks"],
    )
    final_vector = phenotype_vector(final_probes)

    return {
        "condition": condition_name,
        "world_condition": world_payload.get("condition", condition_name),
        "founder_seed": founder_seed,
        "experience_seed": experience_seed,
        "development_ticks": trace.ticks,
        "elapsed_seconds": elapsed,
        "initial_similarity": phenotype_similarity(initial_probes),
        "final_similarity": phenotype_similarity(final_probes),
        "similarity_gain": phenotype_similarity(final_probes) - phenotype_similarity(initial_probes),
        "initial_vector": initial_vector.tolist(),
        "final_vector": final_vector.tolist(),
        "initial_probes": initial_probes,
        "final_probes": final_probes,
        "trajectory": checkpoints,
        "development": summarize_decisions(trace.decisions),
        "network": net.summary(),
    }


def main() -> None:
    args = parse_args()
    cfg = json.loads((ROOT / "config" / "default.json").read_text(encoding="utf-8"))

    if args.smoke:
        cfg["network"]["neurons"] = 384
        cfg["network"]["avg_recurrent_degree"] = 12
        cfg["network"]["action_population_size"] = 12
        cfg["development"]["ticks"] = 1800
        cfg["evaluation"]["settle_ticks"] = 8
        cfg["evaluation"]["probe_ticks"] = 12
        experience_seeds = [31001, 31002]
    elif args.full:
        cfg["network"]["neurons"] = 4096
        cfg["network"]["avg_recurrent_degree"] = 32
        cfg["network"]["action_population_size"] = 72
        cfg["development"]["ticks"] = 100000
        experience_seeds = list(range(31001, 31021))
    else:
        experience_seeds = list(cfg["experiment"]["experience_seeds"])

    if args.neurons is not None:
        cfg["network"]["neurons"] = args.neurons
    if args.ticks is not None:
        cfg["development"]["ticks"] = args.ticks
    if args.seeds is not None:
        if args.seeds < 1:
            raise SystemExit("--seeds must be >= 1")
        experience_seeds = list(range(31001, 31001 + args.seeds))

    founder_seed = int(cfg["experiment"]["founder_seed"])
    worlds = {name: load_world(CONDITIONS[name]) for name in args.conditions}
    probes = load_probes(ROOT / "evaluation" / "adult_phenotype_probes.json")

    out_dir = ROOT / "results"
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    all_runs: List[Dict] = []
    print("Developmental attractor experiment")
    print(f"conditions: {', '.join(args.conditions)}")
    print(f"founder seed: {founder_seed} | experience seeds: {len(experience_seeds)} | neurons: {cfg['network']['neurons']:,} | ticks/life: {cfg['development']['ticks']:,}")
    print("Development receives no adult phenotype labels or historical subject actions.\n")

    for experience_seed in experience_seeds:
        for condition in args.conditions:
            print(f"experience seed {experience_seed} | {condition} ...", flush=True)
            run = run_one(condition, worlds[condition], cfg, founder_seed, experience_seed, probes, trajectory=not args.no_trajectory)
            all_runs.append(run)
            cdir = runs_dir / condition
            cdir.mkdir(exist_ok=True)
            (cdir / f"experience_seed_{experience_seed}.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
            print(f"  similarity {run['initial_similarity']:.3f} -> {run['final_similarity']:.3f} ({run['similarity_gain']:+.3f})")

    summary: Dict = {
        "configuration": cfg,
        "conditions": args.conditions,
        "founder_seed": founder_seed,
        "experience_seeds": experience_seeds,
        "method": {
            "fixed_founder": "Every organism begins with the exact same founder network. Experience seeds affect only stochastic developmental noise and action sampling, not initial wiring or weights.",
            "paired_lives": "For a given experience seed, each condition uses the same stochastic seed so differences are paired by developmental noise as well as founder state.",
            "developmental_firewall": "Adult phenotype probes and target action weights are evaluation-only and are not passed to the world, network learning rule, or reward function.",
            "historical_action_firewall": "The canonical world contains environmental circumstances and opportunities, not Ray Kurzweil's historical action choices.",
            "interpretation": "Evidence for an attractor requires a pattern across replication, separation, convergence, and ablation. A high score in one run is not sufficient."
        },
        "condition_summary": {},
        "paired_margins": {},
    }

    for condition in args.conditions:
        rows = [r for r in all_runs if r["condition"] == condition]
        initial_vecs = [np.asarray(r["initial_vector"], dtype=float) for r in rows]
        final_vecs = [np.asarray(r["final_vector"], dtype=float) for r in rows]
        summary["condition_summary"][condition] = {
            "n": len(rows),
            "mean_initial_similarity": float(np.mean([r["initial_similarity"] for r in rows])),
            "mean_final_similarity": float(np.mean([r["final_similarity"] for r in rows])),
            "mean_similarity_gain": float(np.mean([r["similarity_gain"] for r in rows])),
            "std_final_similarity": float(np.std([r["final_similarity"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "initial_dispersion": dispersion(initial_vecs),
            "final_dispersion": dispersion(final_vecs),
            "initial_pairwise_distance": pairwise_distance(initial_vecs),
            "final_pairwise_distance": pairwise_distance(final_vecs),
        }

    if "canonical" in args.conditions:
        canon = {r["experience_seed"]: r for r in all_runs if r["condition"] == "canonical"}
        for other in args.conditions:
            if other == "canonical":
                continue
            ctrl = {r["experience_seed"]: r for r in all_runs if r["condition"] == other}
            common = sorted(set(canon) & set(ctrl))
            margins = [canon[s]["final_similarity"] - ctrl[s]["final_similarity"] for s in common]
            summary["paired_margins"][other] = {
                "n": len(margins),
                "mean_canonical_minus_control": float(np.mean(margins)) if margins else 0.0,
                "median_canonical_minus_control": float(np.median(margins)) if margins else 0.0,
                "fraction_canonical_higher": float(np.mean([m > 0 for m in margins])) if margins else 0.0,
                "per_seed": {str(s): canon[s]["final_similarity"] - ctrl[s]["final_similarity"] for s in common},
            }

    (out_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSummary")
    for condition, s in summary["condition_summary"].items():
        print(
            f"  {condition:18s} final={s['mean_final_similarity']:.3f} "
            f"gain={s['mean_similarity_gain']:+.3f} dispersion {s['initial_dispersion']:.4f}->{s['final_dispersion']:.4f}"
        )
    for other, s in summary["paired_margins"].items():
        print(f"  canonical - {other}: {s['mean_canonical_minus_control']:+.3f}")
    print(f"\nSaved {out_dir / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
