# Kurzweil Brain Experiments

This repository contains an artificial developmental-attractor experiment built around a narrow question: can the same artificial newborn become systematically different adults when raised in different information-rich developmental environments, and can one reconstructed life move that organism toward a held-out behavioral phenotype associated with a known person?

The first subject is Ray Kurzweil. This is not a role-play system and it is not a claim to reconstruct his mind. The developmental organism receives no adult prose, personality adjectives, historical action labels, or phenotype answer key. It experiences environmental circumstances and learns only from the consequences of its own choices.

## Current result

The first replicated v0.1 test used one fixed 1,024-neuron founder, 12 paired experience seeds, three developmental environments, and 15,000 developmental ticks per life. That produced 36 complete simulated lives and 540,000 developmental ticks.

All organisms began at approximately 0.625 held-out phenotype similarity. Mean final similarity was 0.632 for the canonical Kurzweil-derived environment, 0.596 for the matched-rich creative-humanities control, and 0.673 for the early-computer-access ablation.

Canonical development exceeded the matched-rich control in 11 of 12 paired lives, with a mean paired margin of +0.036. However, the computer-access ablation exceeded canonical development in all 12 paired lives, with a mean canonical-minus-ablation margin of -0.041.

The v0.1 result therefore supports developmental differentiation in this simulator, not a Kurzweil-specific developmental attractor. The same founder repeatedly becomes different adult behavioral policies under different experienced histories. The failed ablation also exposes a useful architectural problem: canonical development appears to overgeneralize repeated reinforcement of `create`, producing a broad creation bias instead of a sufficiently context-dependent policy.

See [RESULTS.md](RESULTS.md) for the interpretation and `results/extended_2026-09-17/` for the preserved aggregate outputs.

## Experimental design

Every organism begins from the exact same founder network. Founder wiring, initial recurrent weights, generic action populations, policy weights, and starting neural state are held constant. Experience seeds affect only developmental noise and stochastic action selection after the founder exists. Each experience seed is reused across conditions, producing paired counterfactual lives.

The initial conditions are `canonical`, an environment reconstructed from publicly documented early circumstances associated with Kurzweil; `matched`, an equally rich creative-humanities life preserving broad opportunity, mentoring, autonomy, challenge, and recognition; and `computer_ablation`, the canonical environment with unusual early programmable-computer access removed.

The matched condition is intentionally not a bland baseline. The experiment is designed to distinguish two information-rich lives rather than a distinctive life from an empty one.

## Developmental firewall

Developmental inputs live under `data/development/`. They contain situations, generic available actions, and action-contingent consequences. They do not contain the historical action Kurzweil took in a situation or the target adult phenotype.

The held-out adult battery lives under `evaluation/`. It is evaluation-only. `audit_firewall.py` checks for forbidden answer-key fields in developmental data, and the test suite verifies that phenotype evaluation cannot modify network state or learning parameters.

Run:

```text
python audit_firewall.py
pytest -q
```

## Behavioral vocabulary

The organism has a generic behavioral vocabulary:

`explore`, `analyze`, `create`, `persist`, `collaborate`, `challenge`, `comply`, `withdraw`, `seek_feedback`, `apply`, `communicate`, and `switch_goal`.

These are action primitives, not subject-specific personality traits. The developmental policy begins generic and is reinforced only by consequences of the organism's own choices.

## Quick start

Python 3.11 or 3.12 is recommended.

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python audit_firewall.py
pytest -q
python run_experiment.py --smoke
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

Windows launchers are included for the smoke, pilot, and full presets.

A reproducible medium-scale run comparable to the preserved extended test can be launched with:

```text
python run_experiment.py --neurons 1024 --ticks 15000 --seeds 12 --no-trajectory
```

The bundled `--full` preset is substantially more expensive: one fixed 4,096-neuron founder, 20 paired experience seeds per condition, and 100,000 developmental ticks per life.

## Repository guide

`attractor_net/` contains the recurrent developmental network, experience encoder, simulated world, and evaluation utilities.

`data/development/` contains the three v0.1 developmental environments.

`evaluation/` contains the held-out adult phenotype probes and target action distributions.

`EXPERIMENT_PROTOCOL.md` records the research question, hypotheses, controls, leakage rules, and interpretation matrix established before the extended result was interpreted.

`SOURCE_LEDGER.md` separates biographical material used for developmental reconstruction from adult material used to justify the answer key.

`REPRODUCIBILITY.md` documents seed separation, firewall structure, and the preserved extended-run configuration.

`RESULTS.md` records the current experimental result without rewriting the failed ablation as a success.

`ROADMAP.md` defines the v0.2 milestone: context-conditioned action learning plus a reward-matched ablation while preserving the v0.1 founder and paired seeds.

## What would count as stronger evidence

A developmental-attractor claim requires more than a high score in one run. The stronger pattern would be replicated movement of the canonical condition toward the held-out phenotype, separation from an equally rich matched control, targeted ablations that alter specific parts of that movement, stable developmental trajectories, and replication under a second distinctive subject using the same generic architecture.

v0.1 does not satisfy all of those requirements. That is why it is being preserved as a baseline rather than retroactively tuned.

## Scientific scope

This is an artificial developmental model. It does not establish causal facts about Ray Kurzweil, human development, personality formation, or neuroscience. Public biographies are incomplete and selectively preserve unusual events. Counterfactual consequences in the simulated worlds are experimenter-authored. The neural architecture is synthetic and intentionally minimal.

The useful scientific object here is the simulator itself: whether a fixed artificial organism can acquire differentiated, persistent behavioral structure from experienced history under controlled counterfactual manipulation.
