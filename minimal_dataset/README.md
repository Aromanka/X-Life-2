# XLife minimal test dataset — 2025 summer cohort

This is a **local review package** prepared from the 2025 summer cohort. It contains 3 pseudonymous participants and 6 observed event windows: 2 meals, 2 exercise records, and 2 nights. The four meal photos were selected from `F:\XLife图像.zip` by exact archived filename and account folder. Each photo was visually checked against its recorded food names.

## Contents

- `sample_data.json`: patient objects compatible with the existing top-level `subject_id`, `metadata`, `meals`, and `nocturnal_hypoglycemia` layout. `exercise_events` and explicit labels/trajectories extend that layout.
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

## Compatibility

The repository's `SampleImageDataset` reads the two `meals` entries and can resolve `food_items[].img_names` when `image_dir` points to `images/`. It also converts top-level `exercise_events` with both CGM windows into standalone exercise samples; night labels remain metadata for downstream rule evaluation. The `P001` meal also populates the legacy `exercise_data_after_meal` field, and `cgm_pre_exercise` / `cgm_post_exercise` supply its observed exercise trajectory.

## Review before publication

This package is included in the repository as a code-verification fixture. Before using it in a journal supplement or other public release, confirm the cohort's permission to release individual photos and trajectories. The `P003` app-to-CGM identity link is based on unique name plus matching age because the app account lacks a phone number; confirm it before any broader release. Clinical duration, complications, specific medication names, and prior hypoglycemia history are unavailable and were left null rather than inferred.
