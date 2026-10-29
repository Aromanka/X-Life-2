# HTTP inference contract

POST JSON to `XLIFE_LLM_URL/v1/{role}` (default localhost:8765). Roles: `candidate/prepare`, `safety/evaluate`, `generate`, `safety`, `action/explicit`, `action/factors`, `metadata/encode`, `embed`, `knowledge/extract`, `teacher`. Prescription prompts and upstream model selection belong to sibling `extra_code`. Action encoder training and its input instructions live in `action_vlm.py`. Importing the client performs no I/O.

Response envelope: `{"result":...,"provenance":{"role":...,"backend":...,"test_backend":boolean}}`. Backend failures do not silently produce mock success.

## Candidate

```json
{
  "mode": "joint",
  "diet": [45, 20, 12, 6, 350, 400],
  "exercise": [20, 70, null],
  "exercise_type": "walking",
  "intensity_scale": "unknown",
  "onset": [0, -20],
  "ingredients": ["rice", "vegetables"],
  "description": "Quantified meal followed by walking",
  "contraindications": [],
  "units": {
    "diet": ["g", "g", "g", "g", "g", "kcal"],
    "exercise": ["min", "kcal", "scale"],
    "onset": "min"
  }
}
```

Diet order: carbohydrate/protein/fat/fiber/mass/energy. Exercise: duration/energy/intensity. Null means unobserved, distinct from zero. Active diet requires all six quantities; exercise requires positive duration and type. Named-scale intensity stays in the audit record and is masked for numeric models with undefined training intensity scales. `[0,-20]` means meal at origin and exercise 20 minutes later. No portion multiplication is repeated.

Generation returns `{"candidates":[...]}`. Safety returns `{"is_safe":boolean,"reasons":[strings],"checks":[{"name":string,"passed":boolean}]}`; malformed or missing checks reject, and any failed check rejects. Requests include the measured history and model simulation explicitly identified as a prediction. External hard vetoes remain independent of semantic acceptance. All normalization, prescription validation, hard rules and semantic-result validation are implemented in sibling extra_code/services. The application only issues POST requests and consumes service decisions.

Constraints support `allergies`, `conditions`, `forbidden_exercise_types`, `max_carbs_g`, `max_energy_kcal`, `max_duration_min`, `min_exercise_delay_min`. Optional `glucose_unit` and `predicted_glucose_bounds={"unit":...,"min":...,"max":...}` require caller-verified matching units. No new clinical thresholds or unnamed clinical-array meanings are assumed.

`action/explicit` accepts text/structured action and returns `{"action":candidate}`. `action/factors` accepts domain/text/images and returns factors/backend_id; exercise additionally returns intensity_scale=MET. Separately fine-tuned Qwen2.5-VL models mean-pool input hidden states and predict structured physiological factors. The local projection consumes only these factors. See ACTION_ENCODERS.md for full-parameter supervision and image inputs. The former action/latent endpoint is retired; old HTTP checkpoints require retraining.

## Knowledge and task extensions

`knowledge_cli.py --input guideline.txt guideline.pdf --output graph.json` extracts via POST then normalizes/deduplicates attributed `head/relation/tail/source` triples. A provenance sidecar records extraction identities. Generation uses substring retrieval; safety uses exact entities. Vector retrieval is explicit and returns embedding provenance. The optional Neo4j adapter connects only when snapshot is requested. Missing graph data is visible in inference output.

`experiment.py --task-data windows.json --output outputs/task_validation` validates explicit CBG/night records. Required: task, subject_id, origin, history_minutes, target_minutes, history_points/target_points as timestamp-value pairs, glucose_unit, label_source. Night additionally requires bedtime equal to origin and wake_time matching target_minutes. This is data validation, not task training or performance validation.

Generated counterfactual actions have no observed future outcome, so the demonstration pipeline does not report prediction accuracy for them. Factual evaluation belongs to the trainer/experiment entry. Mock generation/review, random ranking and random latents prove interfaces only.

## External policy ownership (current)

`POST /v1/candidate/prepare` accepts `{candidate: ...}` and returns a canonical candidate plus its model action in the standard result/provenance envelope. All field, unit, schedule and numeric action decisions belong to extra_code/services/contracts.py. Invalid input is reported by that service as HTTP 422. X-Life-2/generation is a request wrapper only.

`POST /v1/safety/evaluate` accepts candidate/context/simulation/constraints/knowledge and returns the final is_safe decision, audit checks, vetoes and review provenance. All deterministic and semantic review behavior belongs to extra_code/services/safety. The existing `safety` role is an internal semantic-review operation used by that external service. X-Life-2/safety is a request wrapper only. Transport failures are recorded as safety_failed by the pipeline.

The numerical world model, checkpoint compatibility and tensor conversion remain local. They do not implement prescription policy. Tests of prescription rules and API shapes now live under extra_code; run `python -m unittest discover -s extra_code -p 'test_*.py'` from the workspace root.

Metadata encoding uses POST /v1/metadata/encode with canonical fields. It returns embedding/backend_id in the standard envelope. Qwen3 is frozen; the local alignment layer infers width. See METADATA.md for the shared dataset interface and checkpoint changes.
