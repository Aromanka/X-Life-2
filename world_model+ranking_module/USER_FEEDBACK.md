# Deployment personalisation

`train_user_feedback.py` updates the ranking network with feedback DPO (including
professional and physiological preference replay), and the glucose simulation
head with factual actions paired with subsequent glucose measurements. The
simulation loss uses the checkpoint's `predicted_cgm` supervision specification,
including its mask, scale, loss type and weight, through `supervision_loss`.
Other supervision heads, encoders, trajectory transformers, dynamics and gates
remain frozen. Direct readout updates `diet.decoder`, `exercise.decoder` and
`fusion.decoder`; autoregressive readout updates the corresponding `ar_decoders`.
Only modes represented in training observations receive simulation gradients.

Run from the repository root, using a physiological alignment checkpoint:

```bash
python 'world_model+ranking_module/train_user_feedback.py' \
  --config example_config.json \
  --init-checkpoint outputs/physiological/checkpoint.pt \
  --data data/user_feedback.json \
  --observation-data data/user_observations.json \
  --replay-data data/replay_pairs.json \
  --output outputs/personal
```

The new required `--observation-data` accepts JSON arrays or JSONL. Each row has:

| Field | Meaning |
| --- | --- |
| `observation_id` | Stable, nonempty ID of a factual action and measurement event, scoped to the participant. Identical duplicates are counted once; conflicts are rejected. |
| `state_id`, `split` | State identifier and explicit `train`, `validation` or `test` assignment. A state or participant/origin cannot cross splits or change context across any input file. |
| `context` | The same context schema used for feedback pairs: `subject_id`, `origin`, `history`, `history_mask`, `metadata`. History precedes the forecast origin. |
| `action` | The action actually executed: `mode` (`diet`, `exercise`, `joint`), `diet`, `exercise`, `onset`, and optional action text/images/type. This need not match a preferred prescription when execution differed. |
| `target` | Subsequent glucose values in chronological forecast order and in the same units as pretraining. Length equals the checkpoint/configured `horizon_steps` (36 by default). |
| `target_mask` | Equally sized Boolean or 0/1 array. A true entry requires a finite measured value. Unmeasured points can have null targets and false masks. At least one point must be measured per row. |

Keep future measurements outside `context` and `action`. Neither preferred nor
rejected alternatives are treated as executed actions. Physiological replay
observation IDs alone do not supply glucose targets. All feedback and observation
rows must belong to one participant; prior-stage preference replay can include
other participants. At least one training feedback pair and one training observation
are required. Optional `--replay-data` accepts a JSON or JSONL file containing preference pairs from any stages. Replay uses `user_feedback.replay_fraction` from configuration; omitting the file disables replay.

Each epoch visits all factual training observations once, independently of
feedback-pair subsampling and preference replay. Batches from both streams are
paired until both are exhausted; where both are present, their losses are added.
`--max-steps` truncates these optimizer steps. Validation observations are used
only for evaluation; test observations are excluded from updates and validation.
The history preserves `loss` as the preference loss and adds `physiological_loss`,
`observations` and `validation_physiological_loss`.

Resume with `--resume outputs/personal/checkpoint.pt` in place of
`--init-checkpoint`, supplying the same input files and training settings. Both
heads' optimizer states, the fixed ranking reference and random states are saved.
Resume starts at the next epoch, including after a truncated epoch. Checkpoints
retain `phase=physiological` for inference and set `alignment_stage=user_feedback`.
Old ranking-only feedback checkpoints lack this optimizer/data contract and cannot
resume through this path; start a new run from a physiological alignment checkpoint.

Regression checks (CPU, synthetic numeric actions and metadata, no HTTP service):

```bash
python 'world_model+ranking_module/tests/test_user_feedback.py'
```
