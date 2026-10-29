# Steps 9–10 implementation and validation (2026-10-29)

## Delivered

- Maintained v3 offline candidate flow: generation, strict quantitative validation, batched domain simulation, learned or explicitly random ranking, independent rule/semantic safety, safe Top 3. No safety-score ranking fallback.
- The demo context is an allowlist of patient ID/event origin/numeric metadata/measured history/mask. Original actions, action text/onset and all future labels are excluded before generation.
- All language/latent/embedding/annotation roles use JSON HTTP POST. Prompts, upstream model identities and optional local language-model loading are centralized in sibling extra_code. Numeric checkpoints retain their trained numeric action encoders.
- New safety package preserves deterministic policy methods from the original SafeguardAgent and adds canonical-factor adapters, user constraints and joint timing vetoes. Clinical conditions and fitness categories must be named explicitly; unnamed metadata array positions are never guessed. Backend failures and malformed reviews reject candidates.
- Knowledge extraction via HTTP, attributed graph snapshots, canonical entity normalization/deduplication, indexed role-specific retrieval, optional POST vector retrieval and lazy Neo4j snapshot adapter. Actual curated graph content has not been validated.
- Persistence, least-squares historical extrapolation, independent DLinear, optional labelled numeric action correction, fixed five-seed participant80/unseen20 protocol, purged seen holdout, missing-history/metadata/action reports, cost records and separate factual participant fine-tuning.
- Controlled configurations for patch length, state pooling, observation-gap feature and free autoregressive readout; no assumed improvement. CBG/night schema validation requires measured labels and recorded task anchors.
- Old v2 agents/local encoders/duplicate trainers/configs are preserved under extra_code/legacy_xlife and removed from maintained imports. Existing uncommitted v3 alignment work was retained.

## Executed locally

Interpreter: E:/code/Env/envs/ml_env/python.exe. Role service: real localhost HTTP POST server in mock mode; no upstream API calls and no server-local language-model weights loaded.

1. All 26 unittest checks passed, including immutable context, three modes, fewer than three/empty outputs, hard veto dominance, malformed candidates/reviews, service outages, learned-rank guard, migrated clinical rule, graph retrieval, remote latent identity/gradient and controlled readout variants.
2. Using data/xlife_cleaned_0207.json and existing test/step7_smoke/physiological/checkpoint.pt: diet, exercise and joint each generated four candidates and returned three safe results. Outputs: test/step9_10_smoke/{diet,exercise,joint}.json. These are explicitly test-only mock reviews/random rankings.
3. The same existing aligned checkpoint ran learned joint ranking, returning three safe results in test/step9_10_smoke/learned.json. Mock generation/review still makes this a test-only result, not an effectiveness assessment.
4. HTTP random latent action encoders completed two actual training steps over eight real records, saved/loaded a checkpoint and inferred three candidates. Outputs: test/step9_10_smoke/http_action_train and http_action_inference.json. This small training subset has no validation split and proves mechanics only.
5. Three real participants, seed 42 and one epoch exercised all four baseline/world configurations, separate participant adaptation and labelled DLinear action correction. Outputs: test/step9_10_smoke/final_baselines and action_baseline. No generalization or optimization claim follows from these checks.
6. Maintained Python files compiled; exp_1029_v2.sh passed Git Bash syntax checking.

## Pending external/data-dependent work

- Actual supplied upstream API and server-local test model generation/review; real latent weights and artifact identity; production graph import/index content and knowledge quality. Use exp_1029_v2.sh server-api or server-local, then inference.
- Ranking training on the user's server. Random ranking is an explicit smoke option only; use learned with the correct aligned checkpoint later.
- Full five-seed experiments and single-factor comparisons. Commands are in exp_1029_v2.sh experiments and ablations. Local smoke results do not satisfy the full research protocol.
- Verified CBG/night task definitions, bedtime/wake labels and task-specific training. Current implementation validates supplied definitions and does not claim overnight prediction capability.
- Full-parameter domain language-model SFT, additional sensor studies, RevIN/TCN/self-supervision and final scientific conclusions remain unverified/optional.
