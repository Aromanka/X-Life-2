# X-Life minimal dataset

It contains 3 pseudonymous participants and 6 observed event windows: 2 meals, 2 exercise records, and 2 nights.

## Contents

- `sample_data.json`: participant metadata, meals, exercise events, nocturnal CGM windows, labels and observed trajectories.
- `images/`: four actual meal photographs. Filenames are pseudonymous. JPEG metadata segments were removed while decoded pixels were verified identical to the source files.
- `manifest.json`: expected event counts and SHA-256 hashes for the four images.

## Event inventory

| Participant | Meal | Exercise | Night |
| --- | --- | --- | --- |
| `P001` | Dinner with two photos and 3-hour observed CGM | Jogging record approximately 70 minutes after the dinner record, with separate 3-hour observed post-record CGM | Rule-negative night |
| `P002` | Lunch with two photos and 3-hour observed CGM | — | Rule-positive night |
| `P003` | — | Cycling record with 3-hour observed post-record CGM | — |

All meal and exercise windows contain 12 CGM points in the hour before the record and 36 points in the three hours after it, sampled every 5 minutes. Both night windows contain 120 CGM points from 22:00 to 08:00. Values are in mmol/L. These are observed trajectories, not model predictions.

## Time and label definitions

Source application timestamps were stored as UTC without a timezone; CGM device timestamps were stored as Beijing local time. Application timestamps were converted to Beijing time before querying CGM. Each participant's dates were then shifted by a fixed private offset, preserving within-participant intervals and time of day. Public JSON contains no source account IDs, names, phone numbers, or original dates.

`recorded_at` is a proxy for meal or exercise timing. It is **not a verified first-bite or exercise-start time**. The dinner `P001` was followed by an exercise record, so its post-meal CGM is also exposed to that exercise; the effects cannot be separated causally from these observations. The `P002` lunch has observed post-record CGM but no completed app-generated after-meal analysis.

Night labels were recalculated using the project's rule: within 22:00–08:00, at least three consecutive readings on the same device below 3.9 mmol/L. They are **rule-derived CGM labels**, not clinician-adjudicated hypoglycemia events. No actual sleep onset/offset was verified; `cgm_sleep` denotes the night clock window. `metadata["Hypoglycemia (yes/no)"]` remains null because it represents clinical history, not the rule-derived night label.

## Data loading

The repository's `SampleImageDataset` reads the two `meals` entries and resolves `food_items[].img_names` when `image_dir` points to `images/`. It also converts top-level `exercise_events` with both CGM windows into standalone exercise samples; night labels remain metadata for downstream rule evaluation. The `P001` meal includes the `exercise_data_after_meal` field, and `cgm_pre_exercise` / `cgm_post_exercise` supply its observed exercise trajectory.

## Metadata

The `P003` application-to-CGM linkage uses a unique name and matching age; a phone identifier was unavailable. Clinical duration, complications, specific medication names and prior hypoglycemia history are unavailable and stored as null.
