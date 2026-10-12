# X-Life method and architecture

X-Life combines diet and exercise action encoders, an action-conditioned metabolic world model, a prescription ranking module and local knowledge-graph agents.

## Action and physiological encoders

The diet action encoder uses local Qwen2.5-VL-72B-Instruct to encode a meal image and its corresponding dietary description jointly. Attention-mask mean pooling of the last hidden layer is followed by a projection into the diet action latent space.

The exercise action encoder uses local Qwen3-8B to encode exercise type, duration, intensity and metabolic equivalents (MET). Mean-pooled hidden states are projected into the exercise action latent space. Dietary descriptions preserve nutrition, ingredients and preparation information; exercise records retain intensity and MET fields.

CGM values are grouped into fixed-size, non-overlapping patches. Each patch concatenates its values and validity bits before projection and positional encoding. A Transformer models inter-patch dependencies. Incomplete patches are padded and masked; pooling weights count valid measurements. Fully missing histories use a placeholder with zero pooling weight. Metadata and sleep features are encoded with Qwen3 and combined with the pooled CGM features to produce the physiological state.

For timestamped data, measured values occupy configurable interval slots. Missing positions remain masked. Numeric-only sequences are supported when no event timestamp is available. The configuration specifies the sampling interval, horizon and patch size; defaults are 5 minutes, 36 slots and 6 readings per patch.

## Transition, fusion and ranking

Diet and exercise use separate transition modules with the same architecture:

1. A two-token Transformer integrates the current state and action.
2. A GRUCell updates the dynamic latent state autoregressively.
3. A causal trajectory Transformer models the sequence of future latent states.

Cross-attention combines the diet and exercise trajectories. The ranking Transformer receives the current-state token followed by the future latent tokens. An MLP reads the mean transformed future representation to produce a prescription score. The Transformer glucose head G decodes the same representation into the future CGM sequence.

Ranking and CGM prediction share the ranking Transformer. The CGM loss updates this shared representation and G; the preference loss also updates the scalar score head. Teacher forcing and autoregressive generation use the same BOS/previous-value input shift and causal masks.

## Phase 1: Action encoder pretraining

Diet and exercise action encoders are trained with structured nutrition and exercise annotations. Annotations are standardized as sorted JSON and encoded by a frozen local Qwen3 teacher. Attention-mask mean pooling produces cached semantic vectors in `training_targets`.

Training-only linear heads map the action latents to these semantic targets. The objective is the sum of the diet and exercise semantic losses. The default loss is cosine distance; MSE is configurable. Available annotations are selected by target masks. Full action-backbone parameters, action projections and annotation heads are trainable in this phase.

Use `encode_annotations.py` to cache annotation embeddings and `train.py --phase encoder_pretraining` to train the encoders.

## Phase 2: World model training

Pretrained action encoders and action projections are frozen. Trainable components include the physiological projections, CGM encoder, transition and fusion modules, ranking/G modules and auxiliary prediction heads.

The world-model objective combines:

- Measured future CGM MSE from G.
- Diet-branch auxiliary CGM MSE on meal-event rows.
- Exercise-branch auxiliary CGM MSE on exercise-event rows.
- Omics semantic prediction loss on annotated rows.

Each CGM term uses a measured-value mask. The omics head predicts the cached Qwen3 embedding of standardized omics annotations from mean fused future latents. Missing omics targets are masked. A dataset without omics targets requires an explicit zero omics weight for an unaugmented run.

Default auxiliary weights are 0.25 for each branch and 0.1 for omics. These values are configurable in `training.loss_weights`.

Use `train.py --phase world_model` with the encoder-pretraining checkpoint.

## Phase 3: Sequential preference alignment

Alignment proceeds through professional, physiological and user preference stages. The ranker and G are updated while preceding encoders, transitions and fusion modules remain frozen.

For each shared context, the preferred and rejected candidates produce scores `score+` and `score-`. A frozen previous-stage ranker produces `ref+` and `ref-`. The preference objective is:

```text
margin = (score+ - score-) - (ref+ - ref-)
pair_loss = -logsigmoid(beta * margin)
total_loss = weighted_mean(pair_loss) + cgm_weight * factual_CGM_MSE
```

Default beta is 0.1 and CGM weight is 1.0. Copying the previous ranker is sufficient for reference scoring because all preceding modules are fixed. Candidate scoring skips unused glucose decoding; the factual event batch supplies the CGM regularizer.

Candidates replace only diet/exercise actions. They share the context's baseline metadata, CGM history and fixed `event_time`. The observed CGM trajectory is supplied separately for the factual action.

### Replay and user feedback

Replay combines all new pairs with prior-stage samples. For replay fraction `r`, the prior count is `round(new_count * r / (1-r))`. The default fraction is 0.25; rounding can affect the exact proportion in small datasets.

In the user stage, user-source pair weights are multiplied by `selection_count**(-0.5)` and the supplied `diversity_weight` in (0,1]. The weighted loss is normalized by total pair weight. Replay records retain their `source_stage`.

Physiological and user stages require a participant ID. Participant adaptation uses that participant's factual events and preference pairs, together with professional replay and the participant's earlier physiological/user pairs. The stage checkpoint chain is enforced by the training CLI.

Use `train.py --phase preference_alignment` for the sequential stages and `--phase participant_update` for subsequent adaptation.

## Inference export

`--export-inference` saves core weights with `include_training_heads=False`. Diet/exercise annotation heads, dual auxiliary decoders and the omics head are omitted. The exported state dictionary loads strictly into the corresponding inference model. Inference uses routine lifestyle and physiological inputs.

Checkpoints include the architecture version, model configuration, training phase, stage and history.

## Local model and agent interfaces

Model and processor loading uses configured local directories with `local_files_only=True`. Missing local paths produce configuration errors. Encoder injection interfaces support existing local deployments.

Generation and safety assessment use a locally deployed DeepSeek-R1 service. Safety retrieval extracts plan and profile terms and uses exact entity matching by default. Vector retrieval can be selected explicitly with a configured local embedding model.

Deterministic safety checks use their configured thresholds and required inputs. The glucose/vigorous-exercise check accepts an explicitly identified mg/dL reading. Private agent configuration is read from `kg_agents/config.json` or `XLIFE_CONFIG`; credentials can be provided through environment variables.
