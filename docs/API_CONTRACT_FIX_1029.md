# Recorded API output diagnosis and fix, 2026-10-29

Inspected the downloaded `outputs/exp_1029_v2/{diet,exercise,joint}.json` files. All fifteen candidates were `invalid`, before model simulation and before safety review. This was a generation/extraction contract mismatch, not a medical safety rejection or a ranking failure. Generation provenance identifies the API backend rather than mock.

- Diet: irrelevant exercise fields were null, which the old validator rejected unconditionally. Additionally all candidates lacked units and had null active onsets.
- Exercise: factors were returned as top-level duration_min/energy_kcal rather than an array; units and active onsets were absent.
- Joint: extraction returned nested domain objects with descriptions/ingredients/contraindications under each domain. Four carried explicit units; one omitted onset units.

Changes: normalize inactive fields and explicit named-unit factor objects without inventing active measurements, schedules or ambiguous units. Flatten nested actions while preserving descriptions, ingredients and contraindications for safety review. Preserve pre-extraction and raw candidate records. Require units only for active branches, while continuing to require finite active schedules.

The external role server now validates real generation/extraction output using the shared pure-data contract before returning success, with up to three schema-guided attempts. Prompts, examples and repair instructions remain exclusively in extra_code. Missing active quantities/units/times are not silently manufactured by the client. Exhaustion is an explicit HTTP error; repair attempts and validation errors are recorded in provenance. The client timeout allows for repair attempts.

Random requirements now separate dietary energy from exercise duration and specify planned offsets explicitly. A diet request no longer receives an unrelated exercise duration, nor an exercise request an unrelated meal energy target. The CLI reports per-status counts and specific failure reasons.

Validation: the maintained 26 tests and six new external-server contract regression tests passed. Offline replay of downloaded joint candidates normalized four without semantic changes and simulated all four using the local step7 smoke checkpoint; a fixture reviewer selected three. The fifth joint candidate and existing diet/exercise candidates still require regeneration because required units/timing are missing. Original downloaded files were not changed. Compact diagnosis: sibling test/step9_contract_fix/diagnosis.json.

The real upstream API has not been rerun in this fix. Deploy the updated extra_code and X-Life-2 files, restart the role server, then rerun the original inference commands. Review any remaining failures using the newly printed status counts and reasons; Top 3 remains conditional on actual safety acceptance.
