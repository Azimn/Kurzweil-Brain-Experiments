from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable
import numpy as np
from scipy import sparse

from .encoding import ACTIONS, SCALAR_KEYS, AFFORDANCE_KEYS, ExperienceEncoder


class PlasticRecurrentAttractorNet:
    """Sparse recurrent developmental network with local plasticity.

    There are no subject-specific trait variables and no supervised historical
    action labels. Behavior is read directly from generic action populations.
    """

    def __init__(self, cfg: Dict, encoder: ExperienceEncoder, seed: int):
        self.cfg = dict(cfg)
        self.encoder = encoder
        self.n = int(cfg["neurons"])
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.target_rate = float(cfg["target_rate"])
        self.tau = float(cfg["tau"])
        self.dt = float(cfg["dt"])
        self.global_inhibition = float(cfg["global_inhibition"])
        self.plasticity_interval = int(cfg["plasticity_interval"])
        self.hebb_lr = float(cfg["hebb_lr"])
        self.reward_lr = float(cfg["reward_lr"])
        self.weight_decay = float(cfg["weight_decay"])
        self.homeostatic_lr = float(cfg["homeostatic_lr"])
        self.eligibility_decay = float(cfg["eligibility_decay"])
        self.max_abs_weight = float(cfg["max_abs_weight"])
        self.action_temperature = float(cfg.get("action_temperature", 0.12))
        self.action_noise = float(cfg.get("action_noise", 0.025))
        self.policy_lr = float(cfg.get("policy_lr", 0.05))
        self.policy_decay = float(cfg.get("policy_decay", 0.00002))
        self.baseline_lr = float(cfg.get("baseline_lr", 0.04))
        self.policy_update_rule = str(cfg.get("policy_update_rule", "policy_gradient"))
        self.policy_context_lr = float(cfg.get("policy_context_lr", self.policy_lr))
        self.policy_semantic_lr = float(cfg.get("policy_semantic_lr", self.policy_lr))
        self.policy_affordance_lr = float(cfg.get("policy_affordance_lr", self.policy_lr))
        self.policy_context_decay = float(cfg.get("policy_context_decay", self.policy_decay))
        self.policy_semantic_decay = float(cfg.get("policy_semantic_decay", self.policy_decay))
        self.policy_affordance_decay = float(cfg.get("policy_affordance_decay", self.policy_decay))
        self.policy_context_gain = float(cfg.get("policy_context_gain", 1.0))
        self.policy_semantic_gain = float(cfg.get("policy_semantic_gain", 0.0))
        self.policy_affordance_gain = float(cfg.get("policy_affordance_gain", 0.0))
        self.policy_bias_lr_scale = float(cfg.get("policy_bias_lr_scale", 0.08))

        self.excitatory = self.rng.random(self.n) < float(cfg["excitatory_fraction"])
        self.W = self._make_recurrent()
        self.post_idx = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.W.indptr))
        self.pre_idx = self.W.indices.astype(np.int32, copy=False)
        self.eligibility = np.zeros_like(self.W.data, dtype=np.float32)

        self.action_populations = self._make_action_populations()
        self.Win = self._make_input_matrix()
        # Generic actor head. It is never supervised with historical actions.
        # It learns only from consequences of the organism's own choices.
        self.policy_w = self.rng.normal(0.0, 0.01, size=(len(ACTIONS), self.n)).astype(np.float32)
        # v0.2 contextual actor channel. Zero initialization preserves the exact
        # v0.1 founder policy before developmental learning begins.
        self.policy_context_w = np.zeros((len(ACTIONS), len(SCALAR_KEYS)), dtype=np.float32)
        self.policy_semantic_w = np.zeros((len(ACTIONS), self.encoder.sensory_dim), dtype=np.float32)
        self.policy_affordance_w = np.zeros((len(ACTIONS), len(AFFORDANCE_KEYS)), dtype=np.float32)
        self.policy_b = np.zeros(len(ACTIONS), dtype=np.float32)
        self.reward_baseline = 0.0
        self.last_context = np.zeros(len(SCALAR_KEYS), dtype=np.float32)
        self.last_semantic = np.zeros(self.encoder.sensory_dim, dtype=np.float32)
        self.last_affordance = np.zeros(len(AFFORDANCE_KEYS), dtype=np.float32)

        self.v = np.zeros(self.n, dtype=np.float32)
        self.rate = np.full(self.n, self.target_rate, dtype=np.float32)
        self.bias = np.full(self.n, -2.45, dtype=np.float32)
        self.tick = 0


    def set_affordances(self, affordances: Dict[str, float] | None) -> None:
        values = affordances or {}
        for i, key in enumerate(AFFORDANCE_KEYS):
            self.last_affordance[i] = float(np.clip(values.get(key, 0.0), -1.0, 1.0))

    def set_experience_seed(self, seed: int) -> None:
        """Change only stochastic experience sampling, never founder weights or wiring."""
        self.rng = np.random.default_rng(int(seed))

    def _make_recurrent(self) -> sparse.csr_matrix:
        k = int(self.cfg["avg_recurrent_degree"])
        m = self.n * k
        post = np.repeat(np.arange(self.n, dtype=np.int32), k)
        pre = self.rng.integers(0, self.n, size=m, dtype=np.int32)
        mask = pre == post
        while np.any(mask):
            pre[mask] = self.rng.integers(0, self.n, size=int(mask.sum()), dtype=np.int32)
            mask = pre == post
        scale = float(self.cfg["recurrent_scale"]) / np.sqrt(max(k, 1))
        mag = self.rng.lognormal(mean=-1.0, sigma=0.45, size=m).astype(np.float32) * scale
        sign = np.where(self.excitatory[pre], 1.0, -1.0).astype(np.float32)
        W = sparse.csr_matrix((mag * sign, (post, pre)), shape=(self.n, self.n), dtype=np.float32)
        W.sum_duplicates()
        return W

    def _make_action_populations(self) -> Dict[str, np.ndarray]:
        size = min(int(self.cfg["action_population_size"]), max(8, self.n // (len(ACTIONS) * 2)))
        available = np.arange(self.n, dtype=np.int32)
        self.rng.shuffle(available)
        pops = {}
        cursor = 0
        for action in ACTIONS:
            if cursor + size > len(available):
                self.rng.shuffle(available)
                cursor = 0
            pops[action] = np.sort(available[cursor:cursor + size])
            cursor += size
        return pops

    def _make_input_matrix(self) -> sparse.csr_matrix:
        input_dim = self.encoder.input_dim
        k = int(self.cfg["input_degree"])
        m = self.n * k
        rows = np.repeat(np.arange(self.n, dtype=np.int32), k)
        cols = self.rng.integers(0, input_dim, size=m, dtype=np.int32)
        data = self.rng.normal(0.0, float(self.cfg["input_scale"]) / np.sqrt(max(k, 1)), size=m).astype(np.float32)

        # Efference-copy channels connect strongly to their generic action populations.
        extra_rows, extra_cols, extra_data = [], [], []
        scale = float(self.cfg["action_efference_scale"])
        for ai, action in enumerate(ACTIONS):
            col = self.encoder.action_offset + ai
            pop = self.action_populations[action]
            extra_rows.extend(pop.tolist())
            extra_cols.extend([col] * len(pop))
            extra_data.extend([scale] * len(pop))
        rows = np.concatenate([rows, np.asarray(extra_rows, dtype=np.int32)])
        cols = np.concatenate([cols, np.asarray(extra_cols, dtype=np.int32)])
        data = np.concatenate([data, np.asarray(extra_data, dtype=np.float32)])
        Win = sparse.csr_matrix((data, (rows, cols)), shape=(self.n, input_dim), dtype=np.float32)
        Win.sum_duplicates()
        return Win

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        z = np.clip(x, -10.0, 10.0)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32, copy=False)

    def reset_fast_state(self, noise: float = 0.0) -> None:
        self.v.fill(0.0)
        self.rate.fill(self.target_rate)
        if noise:
            self.v += self.rng.normal(0.0, noise, self.n).astype(np.float32)

    def step(self, x: np.ndarray, reward: float = 0.0, learn: bool = True) -> None:
        # Preserve generic situation scalars as an explicit actor context. The
        # channel contains no subject identity, phenotype labels, or historical
        # action targets. Outcome steps can update it through scalar deltas.
        self.last_semantic[:] = x[:self.encoder.sensory_dim]
        self.last_context[:] = x[self.encoder.scalar_offset:self.encoder.action_offset]
        syn = self.W.dot(self.rate)
        ext = self.Win.dot(x)
        excess = max(float(self.rate.mean()) - self.target_rate, 0.0)
        self.v += (self.dt / self.tau) * (-self.v + syn + ext + self.bias - self.global_inhibition * excess)
        self.rate = self._sigmoid(self.v)
        self.tick += 1

        if learn and self.tick % self.plasticity_interval == 0:
            centered_pre = self.rate[self.pre_idx] - self.target_rate
            centered_post = self.rate[self.post_idx] - self.target_rate
            corr = centered_pre * centered_post
            self.eligibility *= self.eligibility_decay
            self.eligibility += corr.astype(np.float32)
            delta = self.hebb_lr * corr + self.reward_lr * float(reward) * self.eligibility - self.weight_decay * self.W.data
            self.W.data += delta.astype(np.float32)
            pre_exc = self.excitatory[self.pre_idx]
            self.W.data[pre_exc] = np.clip(self.W.data[pre_exc], 0.0, self.max_abs_weight)
            self.W.data[~pre_exc] = np.clip(self.W.data[~pre_exc], -self.max_abs_weight, 0.0)
            self.bias += self.homeostatic_lr * (self.target_rate - self.rate)

    def _policy_features(self) -> np.ndarray:
        f = (self.rate - self.target_rate).astype(np.float32, copy=True)
        norm = float(np.linalg.norm(f))
        if norm > 1e-8:
            f /= norm
        return f

    def _policy_context_features(self) -> np.ndarray:
        c = self.last_context.astype(np.float32, copy=True)
        norm = float(np.linalg.norm(c))
        if norm > 1e-8:
            c /= norm
        return c

    def _policy_semantic_features(self) -> np.ndarray:
        s = self.last_semantic.astype(np.float32, copy=True)
        norm = float(np.linalg.norm(s))
        if norm > 1e-8:
            s /= norm
        return s

    def _policy_affordance_features(self) -> np.ndarray:
        a = self.last_affordance.astype(np.float32, copy=True)
        norm = float(np.linalg.norm(a))
        if norm > 1e-8:
            a /= norm
        return a

    def action_probabilities(self, allowed: Iterable[str] | None = None, add_noise: bool = False) -> Dict[str, float]:
        allowed_set = set(allowed) if allowed is not None else set(ACTIONS)
        f = self._policy_features()
        c = self._policy_context_features()
        s = self._policy_semantic_features()
        a = self._policy_affordance_features()
        logits_full = (self.policy_w.dot(f) + self.policy_context_gain * self.policy_context_w.dot(c) + self.policy_semantic_gain * self.policy_semantic_w.dot(s) + self.policy_affordance_gain * self.policy_affordance_w.dot(a) + self.policy_b).astype(np.float64)
        names = [a for a in ACTIONS if a in allowed_set]
        idxs = [ACTIONS.index(a) for a in names]
        logits = logits_full[idxs]
        if add_noise and self.action_noise:
            logits = logits + self.rng.normal(0.0, self.action_noise, size=len(logits))
        logits -= logits.max() if logits.size else 0.0
        probs = np.exp(logits / max(self.action_temperature, 1e-4))
        probs /= probs.sum()
        return {a: float(p) for a, p in zip(names, probs)}

    def choose_action(self, allowed: Iterable[str]) -> tuple[str, Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        allowed = list(allowed)
        f = self._policy_features()
        c = self._policy_context_features()
        s = self._policy_semantic_features()
        a = self._policy_affordance_features()
        probs = self.action_probabilities(allowed, add_noise=True)
        names = list(probs)
        p = np.asarray([probs[n] for n in names], dtype=np.float64)
        idx = int(self.rng.choice(len(names), p=p / p.sum()))
        return names[idx], probs, f, c, s, a

    def reinforce_action(self, action: str, reward: float, probs: Dict[str, float], features: np.ndarray, context_features: np.ndarray, semantic_features: np.ndarray, affordance_features: np.ndarray) -> None:
        if action not in ACTIONS or action not in probs:
            return
        ai = ACTIONS.index(action)

        if self.policy_update_rule == "value_prediction":
            # Contextual action-value learning. The chosen action predicts its
            # observed outcome and updates toward that bounded target. Unlike
            # direct policy-gradient accumulation, repeated success naturally
            # produces diminishing prediction error rather than unbounded
            # preference growth.
            estimate = float(
                self.policy_w[ai].dot(features)
                + self.policy_context_gain * self.policy_context_w[ai].dot(context_features)
                + self.policy_semantic_gain * self.policy_semantic_w[ai].dot(semantic_features)
                + self.policy_affordance_gain * self.policy_affordance_w[ai].dot(affordance_features)
                + self.policy_b[ai]
            )
            delta = float(np.clip(float(reward) - estimate, -1.0, 1.0))
            self.policy_w[ai] *= (1.0 - self.policy_decay)
            self.policy_context_w[ai] *= (1.0 - self.policy_context_decay)
            self.policy_semantic_w[ai] *= (1.0 - self.policy_semantic_decay)
            self.policy_affordance_w[ai] *= (1.0 - self.policy_affordance_decay)
            self.policy_w[ai] += self.policy_lr * delta * features
            self.policy_context_w[ai] += self.policy_context_lr * delta * self.policy_context_gain * context_features
            self.policy_semantic_w[ai] += self.policy_semantic_lr * delta * self.policy_semantic_gain * semantic_features
            self.policy_affordance_w[ai] += self.policy_affordance_lr * delta * self.policy_affordance_gain * affordance_features
            self.policy_b[ai] += (self.policy_lr * self.policy_bias_lr_scale) * delta
            return

        if self.policy_update_rule != "policy_gradient":
            raise ValueError(f"Unknown policy_update_rule {self.policy_update_rule!r}")

        advantage = float(reward) - float(self.reward_baseline)
        self.reward_baseline += self.baseline_lr * (float(reward) - self.reward_baseline)
        error = np.zeros(len(ACTIONS), dtype=np.float32)
        for a, p in probs.items():
            error[ACTIONS.index(a)] = -float(p)
        error[ai] += 1.0
        self.policy_w *= (1.0 - self.policy_decay)
        self.policy_context_w *= (1.0 - self.policy_context_decay)
        self.policy_semantic_w *= (1.0 - self.policy_semantic_decay)
        self.policy_affordance_w *= (1.0 - self.policy_affordance_decay)
        self.policy_w += self.policy_lr * advantage * np.outer(error, features).astype(np.float32)
        self.policy_context_w += self.policy_context_lr * advantage * np.outer(error, context_features).astype(np.float32)
        self.policy_semantic_w += self.policy_semantic_lr * advantage * np.outer(error, semantic_features).astype(np.float32)
        self.policy_affordance_w += self.policy_affordance_lr * advantage * np.outer(error, affordance_features).astype(np.float32)
        self.policy_b += (self.policy_lr * self.policy_bias_lr_scale) * advantage * error

    def summary(self) -> Dict[str, float]:
        return {
            "neurons": int(self.n),
            "recurrent_synapses": int(self.W.data.size),
            "mean_abs_weight": float(np.mean(np.abs(self.W.data))),
            "mean_rate": float(self.rate.mean()),
            "tick": int(self.tick),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, v=self.v, rate=self.rate, bias=self.bias,
                            w_data=self.W.data, w_indices=self.W.indices,
                            w_indptr=self.W.indptr, excitatory=self.excitatory,
                            eligibility=self.eligibility, policy_w=self.policy_w, policy_context_w=self.policy_context_w, policy_semantic_w=self.policy_semantic_w, policy_affordance_w=self.policy_affordance_w, policy_b=self.policy_b,
                            reward_baseline=np.asarray([self.reward_baseline], dtype=np.float32),
                            tick=np.asarray([self.tick], dtype=np.int64))
