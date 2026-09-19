# v0.2 Contextual Learning Preregistration

Status: frozen before any full v0.2 developmental phenotype execution.

## Scientific question

The preserved v0.1 result showed developmental differentiation but did not establish a Kurzweil-specific developmental attractor. In particular, the early-computer-access ablation outperformed canonical development. v0.2 asks whether the smallest contextual action-learning extension can prevent reward learned in one situation from becoming a global action bias, and whether canonical development then separates from an equally rewarding non-computing developmental control.

This experiment does not alter or reinterpret v0.1.

## Frozen inheritance

The first v0.2 comparison reuses founder seed `1842`, exactly 1,024 neurons, exactly 15,000 developmental ticks per life, the twelve experience seeds `31001` through `31012`, the existing canonical and matched-rich developmental worlds, and the existing held-out 12-probe adult phenotype battery. Evaluation remains non-learning and must restore all mutable network and actor state after each probe.

The developmental firewall remains absolute: adult probe prose, phenotype labels, historical answer labels, target action labels, and evaluation-derived information may not enter development.

## Minimal architectural intervention

The v0.2 intervention is the already-implemented contextual actor/value-learning extension. The generic actor retains the recurrent-state policy features from v0.1 and adds zero-initialized action-value weights over generic scalar context. The primary v0.2 condition uses `policy_update_rule = value_prediction`, `policy_context_gain = 1.0`, `policy_semantic_gain = 0.0`, and `policy_affordance_gain = 0.0`. Semantic and affordance channels therefore remain disabled in this first causal test. This avoids introducing multiple new representational mechanisms at once.

The chosen action predicts its observed scalar reward. Its update target is bounded prediction error `clip(reward - estimate, -1, 1)`. Context weights begin at exactly zero, so the newborn founder's initial action policy is unchanged by the added channel. No subject identity, historical action answer, adult phenotype label, or probe-derived feature is available to this learner.

A compatibility condition disables the contextual intervention and must reproduce the preserved v0.1 architecture and behavior within deterministic reproducibility tolerance. This is a regression condition, not a new scientific treatment.

## Reward-matched causal control

The primary causal control replaces the canonical `early_computer_access` developmental event with `data/v0_4/reward_matched_early_event.json`. The replacement must preserve the canonical event's exposure weight, scalar state, available-action set, opportunity timing, and per-action reward magnitudes exactly while changing the domain-specific affordance from programmable-computer access to a non-computing mechanical design workshop.

No reward, opportunity, scalar, action availability, or exposure parameter may be tuned after results are observed. If broader event-level matching is later judged scientifically necessary, that is a separately preregistered experiment rather than a post-result modification of this one.

## Conditions

For each of the twelve paired experience seeds, run: canonical v0.2; matched-rich v0.2; reward-matched early-event v0.2; and v0.1-compatible canonical regression. All conditions use the same founder seed, neuron count, developmental tick budget, evaluation battery, and paired experience seed for a given replicate.

An identical-history determinism replicate must rerun at least one complete condition with the same founder, experience seed, world payload, and configuration and require byte-identical serialized adult model state and decision trace under deterministic settings. Failure is an execution blocker, not a phenotype result.

## Primary outcomes

The primary adult outcome is the existing held-out phenotype score/signature from the 12-probe battery, unchanged from v0.1. Report all twelve paired-seed outcomes before aggregate summaries. The primary causal contrast is canonical v0.2 versus reward-matched v0.2. The matched-rich condition remains an information-rich developmental comparator. The compatibility condition is evaluated only as a v0.1 regression check.

A canonical score increase by itself is not evidence for the target hypothesis. Evidence is stronger only if canonical development separates consistently from the reward-matched control across paired seeds while the reward-matched control has retained the frozen reward/opportunity structure and the compatibility condition preserves v0.1 behavior.

## Trajectory outputs

Unlike the preserved aggregate-only v0.1 run, v0.2 records developmental trajectories for scientific audit. At every developmental decision, preserve tick, condition, experience seed, event identifier, allowed actions, selected action, scalar reward, actor action probabilities, and bounded summaries of contextual action-value state. Trajectory recording is observational only and must not feed information back into the organism.

## Deterministic gates before full execution

The full experiment may run only after tests establish: exact founder identity and zero-initialized contextual weights; exact reuse of all twelve paired experience seeds; 15,000 ticks per life; developmental-firewall integrity; evaluation non-mutation; exact reward-matched event structure; deterministic identical-history replay; condition-state isolation; and successful v0.1 compatibility regression on a smoke-scale preflight.

## Interpretation matrix

If canonical separates from both reward-matched and matched-rich controls with stable paired effects and all controls pass, v0.2 supports the claim that contextual learning plus canonical developmental history produces a more specific adult attractor in this simulator. It does not establish human-like development, personality, consciousness, or a general theory of identity.

If canonical and reward-matched remain similar, the evidence does not isolate domain-specific early computing from generic rewarding creative opportunity. If reward-matched exceeds canonical, the Kurzweil-specific attractor hypothesis remains unsupported under this architecture. If compatibility fails, the architectural comparison is invalid until the regression defect is repaired. If determinism, firewall, reward matching, or evaluation non-mutation fails, no scientific phenotype interpretation is permitted.

## Anti-tuning rule

After a full phenotype result exists, do not change the conditioning representation, gains, update rule, reward-matched event, seeds, founder, tick budget, probe battery, scoring, inclusion criteria, or interpretation thresholds to improve the result. Any such scientifically material change becomes a new preregistered version.