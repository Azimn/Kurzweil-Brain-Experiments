# Results to date

## Extended v0.1 developmental test

The first replicated test used one fixed 1,024-neuron founder, 12 paired experience seeds, three developmental environments, and 15,000 developmental ticks per life. This produced 36 complete simulated lives and 540,000 developmental ticks. Adult phenotype probes were held out from development.

All organisms began at approximately 0.625 phenotype similarity. Mean final similarity was 0.632 for the canonical developmental environment, 0.596 for the matched-rich creative-humanities environment, and 0.673 for the early-computer-access ablation.

Canonical development exceeded the matched-rich control in 11 of 12 paired lives. The mean canonical-minus-matched margin was +0.036. The paired t-test p value was 0.000530 and the Wilcoxon p value was 0.000977. These statistics describe behavior within this simulator. They are not evidence about human developmental causation.

The targeted computer-access ablation exceeded canonical development in all 12 paired lives, with a mean canonical-minus-ablation margin of -0.041. This contradicts the preregistered secondary expectation that removing early programmable-computer access would weaken movement toward the target phenotype.

The main architectural diagnostic is policy overgeneralization. Canonical organisms placed roughly 40 percent of held-out action probability on `create`, suggesting that repeated successful creation experiences generalized into a broad action bias. The ablation retained a more differentiated mixture of creation, analysis, exploration, application, persistence, collaboration, communication, and feedback seeking.

The current result therefore supports a narrower claim: identical artificial founders can become systematically different adult behavioral policies solely through different experienced developmental histories in this architecture. It does not yet establish a Kurzweil-specific developmental attractor.

The preserved aggregate statistics and paired values are under `results/extended_2026-09-17/`.
