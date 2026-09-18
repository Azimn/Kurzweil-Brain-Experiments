# Kurzweil Developmental Attractor, extended test

This run used one fixed 1,024-neuron founder, 12 paired experience seeds, three developmental environments, and 15,000 developmental ticks per life. That is 36 complete simulated lives and 540,000 developmental ticks. Adult phenotype probes were held out from development.

The common pre-development similarity was approximately 0.625. Canonical development finished at 0.632 ± 0.021, a mean gain of +0.007. The matched-rich control finished at 0.596 ± 0.016, a mean gain of -0.029. The computer-access ablation finished at 0.673 ± 0.011, a mean gain of +0.048.

Canonical exceeded the matched-rich control in 11 of 12 paired lives, with a mean paired margin of +0.036. The paired t-test p value was 0.000530 and Wilcoxon p value was 0.000977.

The targeted ablation produced the strongest phenotype match. It exceeded canonical in all 12 paired lives. Equivalently, canonical minus ablation averaged -0.041. This fails the preregistered secondary expectation that removing early programmable-computer access would weaken the canonical shift.

The most plausible architectural diagnostic is policy overgeneralization. Across held-out probes, canonical organisms placed about 40 percent of total action probability on `create`, while the ablation retained a more distributed policy. This suggests that repeated developmental reinforcement can currently create a broad action bias rather than sufficiently context-dependent behavioral tendencies. The result therefore supports developmental differentiation in the simulator, but does not yet establish the intended Kurzweil developmental attractor.
