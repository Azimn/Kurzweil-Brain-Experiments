# v0.2 frozen result

Status: completed and frozen on 2026-09-19.

Scientific workflow run: `35440728368`

Execution commit: `9bbfe85210991387a3739d1fd5d8b84d76626a5e`

Preserved artifact: `kurzweil-v0-2-results-9bbfe85210991387a3739d1fd5d8b84d76626a5e`

Artifact digest: `sha256:a40a96a86e0dfbd06db1d94a106df2951b9c736dd35c0aa53137cc5604e07c19`

The workflow completed successfully after rerunning the developmental firewall and deterministic pre-execution gates. The preserved artifact contains the frozen preregistration, all 48 per-life JSON outputs, `summary.json`, and SHA-256 checksums.

## Frozen aggregate result

All four conditions used founder seed `1842`, 1,024 neurons, 15,000 developmental ticks, the twelve paired experience seeds `31001` through `31012`, and the unchanged held-out adult phenotype battery.

Mean adult phenotype similarity was 0.724719 for canonical v0.2, 0.724346 for reward-matched v0.2, 0.622323 for matched-rich v0.2, and 0.632335 for the v0.1-compatible canonical regression.

The primary preregistered paired contrast, canonical minus reward-matched, had a mean difference of only +0.000374 and a median difference of +0.00000369. Canonical was higher in 7 of 12 paired seeds and reward-matched was higher in 5 of 12. The largest positive paired difference was +0.004804 and the largest negative paired difference was -0.005110. This is not stable canonical separation from the reward-matched causal control.

Canonical did separate from matched-rich in all 12 paired seeds, with a mean canonical-minus-matched difference of approximately +0.1024. That comparison does not rescue the target causal claim because the frozen protocol identifies canonical versus reward-matched as the primary causal contrast.

The compatibility condition averaged 0.632335, matching the preserved v0.1 canonical aggregate of approximately 0.632. Its adult action distribution also reproduces the previously documented broad creation bias, with mean held-out `create` probability approximately 0.401. This supports the intended compatibility regression rather than suggesting that v0.1 was silently rewritten.

## Paired-seed audit

| Seed | Canonical | Reward-matched | Matched-rich | Compatibility | Canonical minus reward-matched |
| --- | ---: | ---: | ---: | ---: | ---: |
| 31001 | 0.737806 | 0.737875 | 0.628429 | 0.637398 | -0.000069 |
| 31002 | 0.742148 | 0.742182 | 0.618795 | 0.625374 | -0.000034 |
| 31003 | 0.719686 | 0.719659 | 0.611550 | 0.603627 | +0.000027 |
| 31004 | 0.731880 | 0.728931 | 0.637849 | 0.674209 | +0.002948 |
| 31005 | 0.721656 | 0.721654 | 0.624993 | 0.612359 | +0.000002 |
| 31006 | 0.720253 | 0.725364 | 0.627710 | 0.625143 | -0.005110 |
| 31007 | 0.726569 | 0.724650 | 0.622225 | 0.641920 | +0.001919 |
| 31008 | 0.716117 | 0.716112 | 0.612940 | 0.621120 | +0.000005 |
| 31009 | 0.731527 | 0.726724 | 0.632003 | 0.666968 | +0.004804 |
| 31010 | 0.724223 | 0.724205 | 0.616263 | 0.629718 | +0.000018 |
| 31011 | 0.715845 | 0.715854 | 0.624627 | 0.626641 | -0.000009 |
| 31012 | 0.708922 | 0.708939 | 0.610495 | 0.623540 | -0.000017 |

## Trajectory and policy audit

The contextual learner materially changes the adult policy relative to compatibility mode, so v0.2 is not simply reproducing the old global creation bias. Across the held-out probes, canonical v0.2 has mean `create` probability approximately 0.210 rather than approximately 0.401 in compatibility mode. Canonical instead retains a broader mixture led by `create` 0.210, `explore` 0.138, `analyze` 0.131, `seek_feedback` 0.117, `persist` 0.099, and `collaborate` 0.098.

However, the reward-matched condition produces essentially the same adult policy: `create` 0.210, `explore` 0.138, `analyze` 0.132, `seek_feedback` 0.116, `collaborate` 0.101, and `persist` 0.098. The per-seed phenotype results likewise track canonical extremely closely. This pattern is inconsistent with a domain-specific canonical advantage and is more consistent with the contextual learner responding to the rich reward and opportunity structure shared by the two conditions.

The matched-rich comparator develops a different policy, led by `communicate` 0.204, `seek_feedback` 0.190, `collaborate` 0.166, and `create` 0.157. This confirms that the architecture remains capable of developmental differentiation. It does not isolate the early-computing affordance because the reward-matched control is the relevant causal comparison.

No evidence in the preserved outputs suggests that aggregate scores are hiding a stable canonical effect. The primary paired effect changes sign across seeds, is near zero for most pairs, and has two larger effects in each direction. There is therefore no basis for attributing the high canonical score to the domain-specific early-computing event rather than the matched reinforcement structure.

## Frozen interpretation

Under the preregistered interpretation matrix, v0.2 does not support the Kurzweil-specific developmental-attractor hypothesis. The smallest context-conditioned learning change appears to correct an important architectural failure from v0.1: the adult policy is substantially less dominated by a single globally reinforced `create` action, while developmental histories still produce differentiated adult policies. That is a useful architectural result.

The stronger causal claim fails, however. Canonical and reward-matched development are effectively indistinguishable on the preregistered adult phenotype outcome and closely match at the action-policy level. The evidence therefore does not isolate domain-specific early programmable-computer access from a generic, equally rewarding creative opportunity with matched timing, novelty, autonomy, difficulty, recognition, mentorship, opportunity density, and reinforcement magnitude.

This result must not be retuned. The conditioning representation, gains, update rule, reward-matched event, seeds, founder, tick budget, probe battery, scoring, inclusion criteria, and interpretation thresholds are frozen with this result. Any material attempt to distinguish domain-specific affordances more strongly belongs to a separately preregistered v0.3 or later experiment.

The scientific claim remains narrow: in this simulator, contextual action learning reduces the previously observed global action-bias failure while preserving experience-dependent developmental differentiation. v0.2 does not demonstrate a Kurzweil-specific attractor, human developmental causation, personality reconstruction, identity, consciousness, or general lifelikeness.
