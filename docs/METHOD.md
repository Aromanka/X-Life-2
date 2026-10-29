# Maintained architecture: v3

Numeric actions contain six dietary factors (carbs, protein, fat, fiber, mass in grams; energy in kcal), three exercise factors (minutes, kcal, source-native intensity), masks, and onset defined as origin minus action start in minutes. Future actions have negative onset. Unknown intensity stays masked; candidate named-scale intensity is retained in audit output and masked from numeric models whose training scale is undefined.

A shared state encoder combines valid-weighted CGM patches, recent patch, numeric clinical values/masks, last CGM and observation gap. Domain-specific conditioning, time-shared basis matrices and residual dynamics produce future states. Causal trajectory Transformers and physiological readouts are domain-specific. Fusion starts with block-diagonal bases and three independently sigmoid-gated increments. A shared bounded ranking head reads trajectories independently from physiological prediction.

Single-domain training learns action/state encoders and domain branches. Fusion freezes action/state encoders and domain dynamics/readouts while training conditioners and fusion components. The shared physiological_loss in train.py serves training and controlled experiments. No true future values enter rollout.

Professional then physiological alignment uses a frozen prior-stage reference, weighted negative logsigmoid of reference-adjusted bounded score margins, and default 25% replay. Only ranking parameters update. Physiological preferences require independently sourced factual observation IDs. No fabricated counterfactual curves are accepted as evidence. User feedback training is not implemented.

Checkpoints record architecture, action backend, normalizers, trained domains/ranked modes, splits, phase, optimizer and random state. Existing numeric v3 defaults retain state-dict compatibility. HTTP latent encoders additionally require artifact identity and dimension; changing backend requires compatible projections/retraining. Remote extraction is frozen inference plus a trainable local projection, not full-parameter SFT. Full-parameter language-model SFT remains unverified.

Controlled variants: patch_size, state_pooling=both|pooled|recent, gap_feature=true|false, readout=direct|autoregressive. The retained autoregressive numerical decoder uses free rollout. Defaults preserve previous numerical behavior. Change one factor on fixed seeds/data/epochs before attributing improvements. No improvement is assumed.

Persistence and least-squares linear extrapolation read observed history only. DLinear independently implements moving-average trend/residual linear prediction; optional numeric action correction is labelled separately. No external code was copied. Controlled experiments currently train single-domain world models; joint report counts are zero. Evaluate trained fusion artifacts through train.py --predict-only --mode joint. RevIN, TCN, self-supervision and multivariate sensor fusion remain optional future experiments.

Five-seed participant80/unseen20 experiments reserve a purged late holdout within seen participants. Statistics use training records only. Report zero-shot unseen separately from optional factual participant fine-tuning, per mode/history missingness, horizon, MAE/mean per-curve Pearson, elapsed time and peak allocated CUDA memory. Local smoke results do not establish generalization.

CBG/night task adapters validate explicit measured points, units, label provenance and task windows on a 5-minute grid. Night requires recorded bedtime/wake anchors. Task-specific action semantics and training remain prerequisites; event-CGM experiments do not establish overnight capability.
