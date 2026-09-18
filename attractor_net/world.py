from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json
import numpy as np

from .encoding import ExperienceEncoder, SCALAR_KEYS
from .network import PlasticRecurrentAttractorNet


@dataclass
class DevelopmentTrace:
    ticks: int
    decisions: List[Dict]
    event_counts: Dict[str, int]


def load_world(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class DevelopmentalWorld:
    """Lets the organism live through environmental circumstances.

    Events contain situations and action-contingent consequences, never the
    historical subject's action. The organism chooses from generic actions.
    """

    def __init__(self, world: Dict, encoder: ExperienceEncoder):
        self.world = world
        self.encoder = encoder
        self.events = sorted(world["events"], key=lambda e: (float(e["age"]), e["id"]))

    def allocations(self, total_ticks: int) -> Dict[str, int]:
        weights = np.asarray([max(float(e.get("exposure_weight", 1.0)), 0.01) for e in self.events])
        raw = weights / weights.sum() * int(total_ticks)
        alloc = np.maximum(20, np.floor(raw).astype(int))
        # Normalize while keeping each event long enough to make decisions.
        scale = total_ticks / max(int(alloc.sum()), 1)
        alloc = np.maximum(20, np.floor(alloc * scale).astype(int))
        diff = int(total_ticks - alloc.sum())
        order = np.argsort(-(raw - np.floor(raw)))
        i = 0
        while diff != 0 and len(order):
            idx = int(order[i % len(order)])
            if diff > 0:
                alloc[idx] += 1
                diff -= 1
            elif alloc[idx] > 20:
                alloc[idx] -= 1
                diff += 1
            i += 1
            if i > total_ticks * 2:
                break
        return {e["id"]: int(a) for e, a in zip(self.events, alloc)}

    def run(self, net: PlasticRecurrentAttractorNet, total_ticks: int,
            input_noise: float = 0.025, decision_interval: int = 24,
            on_event_complete=None) -> DevelopmentTrace:
        alloc = self.allocations(total_ticks)
        zero_action_from = self.encoder.action_offset
        decisions: List[Dict] = []
        event_counts: Dict[str, int] = {}

        for event in self.events:
            ticks = alloc[event["id"]]
            event_counts[event["id"]] = ticks
            scalars = {k: float(event.get("scalars", {}).get(k, 0.0)) for k in SCALAR_KEYS}
            base = self.encoder.encode(event["situation"], scalars=scalars).vector
            allowed = event.get("available_actions") or list(event.get("consequences", {}).keys())
            if not allowed:
                raise ValueError(f"Event {event['id']} has no available actions")

            spent = 0
            while spent < ticks:
                block = min(decision_interval, ticks - spent)
                context_ticks = max(1, block // 3)
                for _ in range(context_ticks):
                    x = base.copy()
                    x[zero_action_from:] = 0.0
                    if input_noise:
                        x[:zero_action_from] += net.rng.normal(0.0, input_noise, zero_action_from).astype(np.float32)
                    net.step(x, reward=0.0, learn=True)

                action, probs, policy_features = net.choose_action(allowed)
                consequence = event["consequences"].get(action, event.get("default_consequence", {"reward": 0.0, "feedback": "The situation continues."}))
                reward = float(consequence.get("reward", 0.0))
                net.reinforce_action(action, reward, probs, policy_features)
                feedback = str(consequence.get("feedback", "The situation responds to the choice."))
                outcome_scalars = dict(scalars)
                for k, v in consequence.get("scalar_delta", {}).items():
                    if k in outcome_scalars:
                        outcome_scalars[k] = float(np.clip(outcome_scalars[k] + float(v), -1.0, 1.0))
                outcome = self.encoder.encode(feedback, scalars=outcome_scalars, action=action, reward=reward).vector
                for _ in range(block - context_ticks):
                    x = outcome.copy()
                    if input_noise:
                        x[:zero_action_from] += net.rng.normal(0.0, input_noise * 0.6, zero_action_from).astype(np.float32)
                    net.step(x, reward=reward, learn=True)

                decisions.append({
                    "age": event["age"], "event_id": event["id"], "action": action,
                    "reward": reward, "action_probabilities": probs,
                })
                spent += block

            if on_event_complete is not None:
                on_event_complete(event, net)

        return DevelopmentTrace(ticks=int(sum(event_counts.values())), decisions=decisions, event_counts=event_counts)
