# v0.5 Preregistration: Senior AI Mentor Access Causal Edit

Status: PROTOCOL ONLY, AWAITING INDEPENDENT REVIEW. No implementation or outcome inspection is authorized by this commit.

Parent frozen substrate: `v0.4-bounded-value-learning` at `7c94ddc82dad71cd3c8b2d6473bbdb4f4a24da4b`.

Governing design gate: Issue #3.

## Scientific question

Holding the v0.4 organism, bounded contextual value learner, founder, developmental schedule, action vocabulary, reward schedule, evaluation instruments, and all non-target developmental events fixed, does access to a senior AI mentor at age 16 exert detectable causal leverage on subsequent developmental policy trajectories?

This is a trajectory experiment. The existing adult phenotype batteries are secondary validation instruments only and are not confirmatory success criteria.

## Frozen manipulation

The sole intervention target is canonical event `senior_ai_mentor_access` at age 16.

The canonical condition is unchanged.

The counterfactual condition replaces only the social/authority identity and technical-domain specificity of that age-16 opportunity with a matched `senior_general_problem_solving_mentor_access` event. The matched event preserves the canonical event's age, exposure weight, scalar vector, available action set, per-action reward magnitudes, number of opportunities, and consequence valence. Feedback text is rewritten only to remove artificial-intelligence-specific and subject-specific mentor content while preserving the functional consequence of each action. The replacement mentor is an unnamed senior researcher who engages strong problem-solving ideas regardless of age or status, provides demanding expert criticism, permits continuing intellectual contact, and does not introduce AI-specific concepts, historical answers, adult phenotype information, or subject labels.

This is not a mentor-removal test. It is a domain-specific mentor-access test chosen because outright removal would necessarily change opportunity count and reward exposure. The causal contrast is therefore: AI-specific senior mentor opportunity versus reward- and opportunity-matched senior general problem-solving mentor opportunity.

No other developmental event may be edited. No downstream event is conditionally generated from this intervention in the current world representation, so the preregistered unavoidable downstream world consequence is `none`. If implementation reveals hidden conditional generation, stop and return to preregistration rather than silently accommodating it.

## Timing and paired lives

The intervention occurs at the existing age-16 event position and exposure weight. Total life length remains 15,000 ticks. Use the exact 12 paired extended-run seeds already frozen for v0.4. For every seed, canonical and counterfactual lives must begin from the exact same 1,024-neuron founder and independent fresh runtime state.

Before the age-16 event begins, canonical and counterfactual serialized organism state and developmental trace must be byte-identical under deterministic settings. Divergence before the intervention is an execution failure.

## Checkpointing

Record a deterministic checkpoint immediately before the age-16 intervention, immediately after completion of the intervention event, and after every subsequent developmental event through the end of the 15,000-tick life. Each checkpoint must include tick/event identity, action counts or frequencies since the previous checkpoint, contextual action-value state, relevant network state digest, cumulative reward, and a canonical serialization digest sufficient to test replay identity without adding new learning state.

The primary pre-window is the complete event immediately preceding `senior_ai_mentor_access`. The immediate post-window is the intervention event itself plus the next canonical developmental event. The persistent post-window is all remaining developmental events through the mature endpoint. Event boundaries, rather than retrospectively chosen tick widths, define these windows.

## Matching requirements

The counterfactual must match canonical on founder, seed, total ticks, event position, exposure weight, scalar vector, available actions, action-specific reward magnitudes, opportunity count, capacity, and evaluation semantics. The only intended difference is the event's AI-specific semantic/domain affordance, replaced by a general problem-solving research-mentor affordance.

A deterministic descriptor test must fail closed if any protected field differs. Firewall auditing must confirm that neither event contains adult prose, personality labels, historical action labels, phenotype target answers, or evaluation-derived information.

## Primary trajectory metrics

Primary analysis is paired and per-seed. For each post-intervention checkpoint, report the canonical-versus-counterfactual distance in the full contextual action-value table over contexts actually encountered in either paired life, using mean absolute value difference with absent entries represented explicitly rather than silently dropped. Also report Jensen-Shannon divergence of normalized action-frequency distributions for the corresponding checkpoint interval. These two metrics are fixed before implementation and neither has a preferred direction.

For each seed, report the full checkpoint trajectory before any aggregate. Aggregate summaries are the median paired distance at each checkpoint and the number of seeds with non-zero divergence at each checkpoint. These are descriptive causal-effect summaries, not population-statistical claims.

No adult phenotype score is part of the primary success definition.

## Secondary validation reporting

At the mature endpoint, run both already-existing adult phenotype batteries with evaluation-state preservation exactly as in v0.4. Report paired per-seed scores and aggregates as secondary validation instruments only. Do not use either battery to tune the intervention, trajectory metrics, learner, world, or inclusion rules, and do not describe these batteries as untouched confirmatory tests.

Also report cumulative reward, action counts, policy entropy where already derivable without changing the learner, and any saturation/collapse diagnostic already present in v0.4.

## Null and controls

The paired canonical life is the scientific null/comparison condition. Add an identical-history determinism control in which two fresh canonical runtimes with the same founder and seed consume byte-identical developmental input and must reproduce identical checkpoints, mature state digest, and evaluation outputs.

Add a serialization/replay control at the pre-intervention checkpoint: resuming from the serialized checkpoint must reproduce the uninterrupted canonical continuation exactly under deterministic settings.

These controls are execution gates, not scientific outcomes.

## Exclusion and failure rules

No preregistered seed may be excluded because of its outcome. A life is invalid only for a deterministic execution failure such as founder mismatch, seed contamination, firewall violation, protected-field mismatch, pre-intervention divergence, serialization/replay mismatch, evaluation mutation, corrupt artifact, or unequal tick/opportunity exposure. Repairable implementation defects require repair and complete rerun of the affected paired experiment without changing this protocol. A scientific-design defect requires a separately versioned preregistration.

## Interpretation matrix

If paired trajectories remain identical after the intervention, v0.5 finds no detectable causal leverage from AI-specific mentor-domain content under this frozen model and matched mentor opportunity.

If trajectories diverge immediately but reconverge before maturity, the result supports a transient developmental effect but not persistent leverage.

If trajectories diverge after the intervention and remain separated across later checkpoints in a stable subset of paired seeds while matching and determinism gates pass, the result supports persistent causal leverage of AI-specific versus general problem-solving mentor-domain affordance in this toy developmental model.

If adult validation scores differ without corresponding post-intervention trajectory divergence, treat that as a metric/audit warning rather than evidence for the hypothesis.

If divergence is explained by reward, opportunity, action-vocabulary, capacity, founder, seed, or evaluation differences, the scientific result is invalid.

A null or contrary result is retained unchanged. No metric threshold, event wording, seed, inclusion rule, or interpretation rule may be changed after outcomes are inspected to obtain a preferred result.

## Anti-tuning and project boundary

The v0.4 organism, learner, generic architecture, founder, core world outside the single matched intervention, and both evaluation batteries are frozen. No language generation enters causal cognition. This experiment does not test consciousness, sentience, selfhood, or general lifelikeness.

Issue #4 remains inactive. Multi-subject developmental generalization and any Pretorius causal-core bridge are explicitly outside v0.5 and may not be activated by this protocol.

## Independent-review gate

This protocol commits to the first manipulation before implementation or outcome inspection. The Shared Scientific Reviewer must approve this exact protocol head before producer implementation begins. Any protocol-changing commit invalidates that approval. After implementation, the exact scientific implementation/protocol head requires a second independent approval before expensive execution.