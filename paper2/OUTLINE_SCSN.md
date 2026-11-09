# Paper 2 v2 — outline (stage-conditional shift normalisation)

Working title: **Stage-conditional shift normalisation removes the patient
signature that marginal adaptation leaves behind in EEG sleep staging**

Target: Neurocomputing (method-driven). Running title: Conditional shift
normalisation for sleep staging.

## Claim

Marginal test-time normalisation (AdaBN, post-hoc mean removal) is the default
label-free remedy for patient shift, but in sleep staging it removes only a
recording-wide offset. The patient-specific part of the representation is
stage-conditional: within Wake, N2, N3 and REM a linear probe still identifies
the patient after marginal correction, and on a large cohort the correction
can *lower* accuracy and per-patient coverage because it subtracts a global
mean that encodes the patient's sleep architecture. Estimating and removing the
shift separately for each sleep stage, using the model's own soft predictions
and no labels, removes the conditional signature as well, preserves accuracy,
and makes per-patient conformal coverage more uniform.

## Contributions

1. A decomposition of the penultimate representation into a recording offset,
   a stage mean and a stage-conditional subject shift, with a proposition
   showing that marginal correction leaves (and can amplify) the conditional
   term and biases predictions by the recording's stage mix.
2. Stage-conditional shift normalisation (SCSN): closed-form, post-hoc,
   label-free, model-agnostic, two passes over a recording; no weights, no
   optimiser, no subject identifiers.
3. A reliability evaluation with split conformal prediction that reports
   per-patient coverage dispersion and the worst-covered patient, not only
   marginal coverage; SCSN improves both over AdaBN on HMC.
4. Controls that isolate the mechanism: subspace orthogonal to the classifier
   row space leaves predictions unchanged; mismatched statistics do not remove
   identity; hard pseudo-labels over-correct and lose accuracy.

## Theory

- Decomposition and the marginal-residual proposition.
- Conditional-residual proposition: with accurate soft assignments the
  within-stage subject mean is removed.
- Task preservation in the classifier null space.
- Relation to BN, AdaBN, GroupNorm, class-conditional TTA.

## Experiments

Cohorts: Sleep-EDF-20 (20 subjects), HMC (151 subjects); cross-cohort transfer
Sleep-EDF -> HMC where available. Backbones: EEGNet, SleepCNN.

Arms: BN, marginal (post-hoc AdaBN), layerwise AdaBN, SCSN, SCSN lambda grid,
low-rank SCSN, hard-label SCSN, orthogonalised SCSN, mismatched-statistics
control, GroupNorm-trained model.

Metrics: accuracy, kappa, macro-F1, ECE; conformal coverage, worst-subject
coverage, 10th-percentile coverage, median coverage, coverage SD, fraction of
subjects below nominal, set size; subject probe overall and within stage.

Planned figures:

1. Method schematic (two-pass recording-level correction).
2. Within-stage subject probe by stage, BN / marginal / SCSN, both cohorts.
3. Per-recording accuracy change vs stage-mix divergence from the cohort
   (marginal negative, SCSN flat), HMC.
4. Per-subject coverage ECDF and worst-tail, HMC.
5. Lambda and rank ablations.

Planned tables:

1. Main results, HMC and Sleep-EDF-20, two backbones.
2. Ablations and controls.
3. Cross-cohort transfer.
4. Runtime and hyperparameters (appendix).

## Positioning

- PSDNorm (ICLR 2026): input-domain PSD/Monge test-time normalisation; ours is
  representation-level and stage-conditional. Cite and contrast.
- S3D-Net (DASFAA 2026): disentangled subject-invariant architecture; requires
  retraining and subject supervision; ours is post-hoc.
- CAFA, Tent, UnMix-TNS, FairBN: class-aware / statistical TTA; contrast
  closed-form, no gradients, stage-conditional, CP reliability.
- CroSS: subject-aware BN during training; ours at inference on a frozen model.
- Paper 1 / current paper 2 analysis is reused as the layerwise motivation.

## Risks and mitigations

- "Why not just AdaBN?" -> HMC failure analysis, per-stage probe, controls.
- "Coverage is nominal by construction." -> report dispersion and worst tail.
- "Probe is a proxy for privacy." -> call it patient-identifying information;
  support with accuracy retention and mismatch controls.
- "Class-conditional TTA exists." -> closed form, no optimisation, sleep-stage
  conditional structure, and the failure analysis are the new parts.
