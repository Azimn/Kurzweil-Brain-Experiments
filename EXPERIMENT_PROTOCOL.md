# Experiment Protocol: Developmental Attractor Test

## Research question

Can a fixed artificial organism, exposed only to a structured developmental environment associated with a known person, develop a held-out behavioral fingerprint resembling that person's documented adult patterns more strongly than it does under equally rich counterfactual environments?

## Primary hypothesis

Repeated exposure to the canonical developmental environment will shift the fixed founder's behavioral policy toward the held-out target phenotype more strongly than the matched-rich creative-humanities environment.

## Secondary hypotheses

Removing early programmable-computer access while leaving the rest of the canonical life intact will reduce at least part of the canonical shift. Across stochastic developmental replications, the direction of the canonical-minus-control difference should be reasonably stable. Developmental checkpoint trajectories should show cumulative change rather than dependence on one final event.

## Fixed founder design

The experiment uses one founder seed. The recurrent weights, input wiring, action populations, policy head, homeostatic parameters, and initial state distribution are identical for every organism in every condition.

Experience seeds are separate. They affect input noise, stochastic action selection, and other moment-to-moment developmental randomness after the founder has been constructed. Each experience seed is reused across all conditions to create paired counterfactual lives.

This deliberately avoids turning the first experiment into a nature-versus-nurture study.

## Independent variable

Developmental environment.

The initial three levels are canonical Kurzweil-derived environment, matched creative-humanities environment, and canonical environment with early computer access ablated.

## Dependent variables

The primary outcome is held-out phenotype similarity across 12 novel probe situations. Similarity is calculated per probe from the Jensen-Shannon similarity between the organism's behavioral action distribution and a predeclared target distribution.

Secondary outcomes are canonical-minus-control paired margins, pre-development to post-development gain, phenotype-vector dispersion across stochastic lives, pairwise distance across lives, action frequencies during development, reward history, and checkpoint trajectory.

## Leakage controls

Development receives no adult phenotype target distributions.

Development receives no historical action labels such as "Kurzweil persisted," "Kurzweil invented," or "Kurzweil challenged criticism."

Biographical facts that describe his own behavior are used only to construct or justify the evaluation target, unless they can be converted into an external opportunity without preserving his historical response.

Evaluation is read-only and noninvasive. Unit tests verify that running the probe battery does not alter recurrent weights, policy weights, neural state, or the developmental clock.

## Canonical input rule

Only externally experienced circumstances can be presented directly. Examples include an intellectually supportive artistic family, rare access to a programmable computer through an engineer relative, receptive teachers, a strong science environment, access to a senior AI researcher who takes ideas seriously, MIT resources, and exposure to a real information problem among students.

The simulator is not told what the historical person did with those circumstances.

## Control design

The principal control must be information-rich. It should preserve broad developmental advantages such as intellectual support, autonomy, mentorship, competitive opportunities, access to high-quality resources, and meaningful projects while changing the recurring causal structure and domain affordances.

A bland or deprived control would make success too easy to interpret.

## Interpretation matrix

Canonical greater than matched, with positive canonical gain and replication across experience seeds, supports the narrow claim that this artificial developmental environment contains information that moves this architecture toward the specified adult behavioral region.

Canonical approximately equal to matched suggests that the target phenotype may be too generic, the two environments may be behaviorally equivalent to the architecture, or the representation may not capture the relevant differences.

Canonical greater than matched but canonical approximately equal to computer ablation suggests that the early computer-access event is not a major causal contributor in this model.

All conditions approximately equal to baseline suggests inadequate developmental learning or an evaluation battery that is insensitive to learned differences.

Large seed-to-seed reversals suggest the developmental dynamics are unstable and do not yet support attractor language.

## Confirmatory run

The bundled `--full` preset uses 4,096 neurons, 100,000 developmental ticks per life, 20 paired experience seeds, and all three initial conditions. The exact configuration is written into every result summary.

The included smoke run is not confirmatory evidence and should be ignored when evaluating the substantive hypothesis.
