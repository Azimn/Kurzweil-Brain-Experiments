from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from typing import Any, Dict, Iterable, List

import numpy as np

from attractor_net.encoding import ACTIONS
from attractor_net.network import PlasticRecurrentAttractorNet


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def network_state_digest(net: PlasticRecurrentAttractorNet) -> str:
    """Digest all mutable state that can affect deterministic continuation."""
    h = hashlib.sha256()
    arrays = (
        net.v, net.rate, net.bias, net.W.data, net.W.indices, net.W.indptr,
        net.eligibility, net.policy_w, net.policy_context_w,
        net.policy_semantic_w, net.policy_affordance_w, net.policy_b,
        net.last_context, net.last_semantic, net.last_affordance,
    )
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str(a.dtype).encode("ascii"))
        h.update(_canonical_json(list(a.shape)))
        h.update(a.tobytes(order="C"))
    h.update(_canonical_json({
        "tick": int(net.tick),
        "reward_baseline": float(net.reward_baseline),
        "rng_state": net.rng.bit_generator.state,
    }))
    return h.hexdigest()


def serialize_checkpoint(net: PlasticRecurrentAttractorNet) -> bytes:
    """Serialize an isolated runtime checkpoint, including RNG continuation state."""
    return pickle.dumps(net, protocol=5)


def restore_checkpoint(payload: bytes) -> PlasticRecurrentAttractorNet:
    net = pickle.loads(payload)
    if not isinstance(net, PlasticRecurrentAttractorNet):
        raise TypeError("checkpoint did not contain a PlasticRecurrentAttractorNet")
    return net


def checkpoint_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def action_counts(decisions: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter(str(row["action"]) for row in decisions)
    return {action: int(counts.get(action, 0)) for action in ACTIONS}


def normalized_action_distribution(decisions: Iterable[Dict[str, Any]]) -> np.ndarray:
    counts = action_counts(decisions)
    values = np.asarray([counts[action] for action in ACTIONS], dtype=np.float64)
    total = float(values.sum())
    return values / total if total else np.zeros_like(values)


def jensen_shannon_divergence(left: Iterable[Dict[str, Any]], right: Iterable[Dict[str, Any]]) -> float:
    p = normalized_action_distribution(left)
    q = normalized_action_distribution(right)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask]))) if np.any(mask) else 0.0

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def encountered_context_value_table(decisions: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Preserve preregistered action-value observations without inventing missing entries."""
    table: Dict[str, Dict[str, float]] = {}
    for row in decisions:
        event_id = str(row["event_id"])
        values = row.get("contextual_action_values", {})
        table[event_id] = {str(action): float(value) for action, value in sorted(values.items())}
    return table


def mean_absolute_value_distance(left: Dict[str, Dict[str, float]], right: Dict[str, Dict[str, float]]) -> float:
    """Compare union of encountered context/action entries, representing absence explicitly as zero."""
    keys = sorted({(c, a) for c, row in left.items() for a in row} | {(c, a) for c, row in right.items() for a in row})
    if not keys:
        return 0.0
    return float(np.mean([abs(left.get(c, {}).get(a, 0.0) - right.get(c, {}).get(a, 0.0)) for c, a in keys]))


def make_checkpoint(event_id: str, decisions: List[Dict[str, Any]], net: PlasticRecurrentAttractorNet,
                    cumulative_reward: float) -> Dict[str, Any]:
    payload = serialize_checkpoint(net)
    return {
        "event_id": str(event_id),
        "tick": int(net.tick),
        "action_counts": action_counts(decisions),
        "contextual_action_values": encountered_context_value_table(decisions),
        "network_state_sha256": network_state_digest(net),
        "serialization_sha256": checkpoint_digest(payload),
        "cumulative_reward": float(cumulative_reward),
    }
