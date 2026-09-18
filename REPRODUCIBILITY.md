# Reproducibility notes

The experiment separates founder randomness from developmental randomness. `founder_seed` constructs the initial recurrent wiring, input wiring, action populations, policy weights, and starting neural state. Every condition receives the same founder. `experience_seed` controls only stochastic developmental sampling after the founder exists, and each experience seed is reused across conditions for paired counterfactual lives.

The developmental firewall is structural. Files under `data/development/` contain environmental circumstances, generic available actions, and action-contingent consequences. They must not contain the adult phenotype answer key or historical action labels. The held-out phenotype battery lives under `evaluation/`. `audit_firewall.py` and the integrity tests enforce the current separation.

The 36-life extended result preserved in this repository used 1,024 neurons, 15,000 developmental ticks per life, 12 paired experience seeds numbered 31001 through 31012, and all three v0.1 environments. The result files in `results/extended_2026-09-17/` are the preserved summary outputs from that run.

For a quick software check, use `python run_experiment.py --smoke`. For a new larger run, specify the desired resources explicitly, for example `python run_experiment.py --neurons 1024 --ticks 15000 --seeds 12`. The bundled `--full` preset is intentionally more expensive at 4,096 neurons, 100,000 ticks per life, and 20 paired seeds.

Evaluation is read-only. The test suite verifies that running phenotype probes does not modify recurrent weights, policy weights, neural state, or developmental time.
