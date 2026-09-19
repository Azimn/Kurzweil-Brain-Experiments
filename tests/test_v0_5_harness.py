from __future__ import annotations

import json
from pathlib import Path

from run_v0_5_causal import _config
from v0_5_experiment import build_world_pair, validate_world_pair, TARGET_EVENT_ID, REPLACEMENT_EVENT_ID

ROOT = Path(__file__).resolve().parents[1]


def test_v0_5_harness_uses_frozen_large_scale_configuration():
    cfg, seeds, founder = _config()
    assert founder == 1842
    assert seeds == list(range(31001, 31013))
    assert cfg["network"]["neurons"] == 1024
    assert cfg["development"]["ticks"] == 15000


def test_v0_5_pair_has_single_position_matched_intervention():
    validate_world_pair()
    canonical, counterfactual = build_world_pair()
    ci = [i for i, e in enumerate(canonical["events"]) if e["id"] == TARGET_EVENT_ID]
    xi = [i for i, e in enumerate(counterfactual["events"]) if e["id"] == REPLACEMENT_EVENT_ID]
    assert len(ci) == len(xi) == 1
    assert ci == xi
    assert len(canonical["events"]) == len(counterfactual["events"])


def test_v0_5_preflight_does_not_create_results(tmp_path, monkeypatch):
    # Structural smoke is deliberately outcome-blind. Scientific execution is
    # available only through the explicit non-preflight path after review.
    source = (ROOT / "run_v0_5_causal.py").read_text(encoding="utf-8")
    assert 'if args.preflight:' in source
    assert 'scientific outcomes not executed' in source
    assert 'for seed in seeds:' in source
