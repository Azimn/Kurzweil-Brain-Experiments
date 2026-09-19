from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json
import numpy as np

from .encoding import ACTIONS, ExperienceEncoder, SCALAR_KEYS
from .network import PlasticRecurrentAttractorNet


def load_probes(path: str | Path) -> List[Dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["probes"]


def _normalize_target(weights: Dict[str, float]) -> np.ndarray:
    v = np.asarray([max(float(weights.get(a, 0.0)), 0.0) for a in ACTIONS], dtype=np.float64)
    if v.sum() <= 0:
        v[:] = 1.0
    return v / v.sum()


def _js_similarity(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0); p /= p.sum()
    q = np.clip(q, eps, 1.0); q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    js = 0.5 * (kl_pm + kl_qm)
    return float(np.clip(1.0 - js, 0.0, 1.0))


def run_probe_battery(net: PlasticRecurrentAttractorNet, encoder: ExperienceEncoder,
                      probes: List[Dict], settle_ticks: int = 30, probe_ticks: int = 50) -> List[Dict]:
    results = []
    zero = np.zeros(encoder.input_dim, dtype=np.float32)
    saved_v, saved_rate = net.v.copy(), net.rate.copy()
    saved_tick = net.tick
    saved_rng = net.rng.bit_generator.state
    saved_context = net.last_context.copy()
    saved_semantic = net.last_semantic.copy()
    saved_affordance = net.last_affordance.copy()
    try:
        for probe in probes:
            net.reset_fast_state(noise=0.0)
            net.set_affordances(probe.get("affordances", {}))
            for _ in range(settle_ticks):
                net.step(zero, reward=0.0, learn=False)
            scalars = {k: float(probe.get("scalars", {}).get(k, 0.0)) for k in SCALAR_KEYS}
            x = encoder.encode(probe["scenario"], scalars=scalars).vector
            accum = {a: 0.0 for a in ACTIONS}
            for _ in range(probe_ticks):
                net.step(x, reward=0.0, learn=False)
                probs = net.action_probabilities(ACTIONS, add_noise=False)
                for a in ACTIONS:
                    accum[a] += probs[a]
            avg = {a: accum[a] / probe_ticks for a in ACTIONS}
            target = _normalize_target(probe["target_action_weights"])
            obs = np.asarray([avg[a] for a in ACTIONS], dtype=np.float64)
            results.append({
                "id": probe["id"], "scenario": probe["scenario"],
                "action_probabilities": avg,
                "target_action_weights": probe["target_action_weights"],
                "similarity": _js_similarity(obs, target),
            })
    finally:
        net.v[:] = saved_v; net.rate[:] = saved_rate
        net.tick = saved_tick
        net.rng.bit_generator.state = saved_rng
        net.last_context[:] = saved_context
        net.last_semantic[:] = saved_semantic
        net.last_affordance[:] = saved_affordance
    return results


def phenotype_similarity(results: List[Dict]) -> float:
    return float(np.mean([r["similarity"] for r in results])) if results else 0.0


def phenotype_vector(results: List[Dict]) -> np.ndarray:
    vals = []
    for r in results:
        vals.extend(r["action_probabilities"][a] for a in ACTIONS)
    return np.asarray(vals, dtype=np.float64)
