from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

# Generic behavioral primitives. None are Kurzweil-specific traits.
ACTIONS = [
    "explore",
    "analyze",
    "create",
    "persist",
    "collaborate",
    "challenge",
    "comply",
    "withdraw",
    "seek_feedback",
    "apply",
    "communicate",
    "switch_goal",
]

# Generic descriptions of the experienced situation, not personality dimensions.
SCALAR_KEYS = [
    "valence",
    "arousal",
    "social",
    "authority",
    "autonomy",
    "novelty",
    "achievement_opportunity",
    "uncertainty",
    "resource_access",
    "creative_context",
    "technical_context",
    "human_need",
    "criticism",
    "time_horizon",
]


# Generic environmental affordances used by later experimental variants.
# These describe what the current environment permits, not subject traits.
AFFORDANCE_KEYS = [
    "symbolic_manipulation",
    "programmable_rules",
    "rapid_iteration",
    "pattern_comparison",
    "physical_manipulation",
    "cross_domain_bridge",
    "large_information_space",
    "audience_interaction",
]

@dataclass
class EncodedExperience:
    vector: np.ndarray
    reward: float


class ExperienceEncoder:
    """Deterministic model-free encoder.

    Narrative content is signed feature hashing. Generic scalar channels describe
    the external situation. The action channel is reserved for the organism's
    own chosen action as efference copy. Historical subject actions are never
    supplied during development.
    """

    def __init__(self, sensory_dim: int = 512):
        self.sensory_dim = int(sensory_dim)
        self.scalar_offset = self.sensory_dim
        self.action_offset = self.scalar_offset + len(SCALAR_KEYS)
        self.input_dim = self.action_offset + len(ACTIONS)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        toks = re.findall(r"[a-z0-9']+", text.lower())
        return toks + [f"{a}::{b}" for a, b in zip(toks, toks[1:])]

    def _hash_index_sign(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        return value % self.sensory_dim, (1.0 if ((value >> 8) & 1) else -1.0)

    def encode(
        self,
        text: str,
        scalars: Optional[Dict[str, float]] = None,
        action: Optional[str] = None,
        reward: float = 0.0,
    ) -> EncodedExperience:
        x = np.zeros(self.input_dim, dtype=np.float32)
        tokens = self._tokens(text)
        for tok in tokens:
            idx, sign = self._hash_index_sign(tok)
            x[idx] += sign
        if tokens:
            norm = float(np.linalg.norm(x[: self.sensory_dim]))
            if norm > 1e-8:
                x[: self.sensory_dim] /= norm

        scalars = scalars or {}
        for i, key in enumerate(SCALAR_KEYS):
            x[self.scalar_offset + i] = float(np.clip(scalars.get(key, 0.0), -1.0, 1.0))

        if action is not None:
            if action not in ACTIONS:
                raise ValueError(f"Unknown action {action!r}")
            x[self.action_offset + ACTIONS.index(action)] = 1.0

        return EncodedExperience(x, float(np.clip(reward, -1.0, 1.0)))
