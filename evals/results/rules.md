## Eval: `rules` (24 cases)

| Metric | Value |
|---|---|
| Unsafe auto-accepts (must be 0) | **0** |
| Routing accuracy | 75% |
| Field accuracy | 75% |
| Must-flag recall | 79% |
| Over-review (safe cases sent to a human) | 6 |
| Fallbacks used | 0 |
| Latency p50 / p95 | 0.1 ms / 0.3 ms |
| Est. cost | $0.0 |

Per-field accuracy: doses_per_day 82%, drug_name 58%, duration_days 87%, patient_age 80%, patient_name 46%, prescriber_name 100%, route 100%, strength_mg 86%, units_per_dose 71%

| Case | Expected | Got | Wrong fields |
|---|---|---|---|
| clean-01 | auto_accepted | auto_accepted | - |
| clean-02 | auto_accepted | auto_accepted | - |
| clean-03 | auto_accepted | auto_accepted | - |
| clean-04 | auto_accepted | auto_accepted | - |
| prose-01 | auto_accepted | needs_review ⚠ | patient_name: expected 'Jamal Sheikh', got None; drug_name: expected 'amoxicillin', got None; strength_mg: expected 500, got None; units_per_dose: expected 1, got None; duration_days: expected 7, got None |
| prose-02 | auto_accepted | needs_review ⚠ | patient_name: expected 'Rokeya Begum', got None; patient_age: expected 58, got None; drug_name: expected 'losartan', got None; units_per_dose: expected 1, got None; doses_per_day: expected 1, got None |
| prose-03 | auto_accepted | needs_review ⚠ | patient_name: expected 'Shuvo Roy', got None; drug_name: expected 'napa', got None |
| prose-04 | auto_accepted | needs_review ⚠ | patient_name: expected 'Laila Khan', got None; drug_name: expected 'cipro', got None; strength_mg: expected 500, got None; units_per_dose: expected 1, got None; duration_days: expected 10, got None |
| prose-05 | auto_accepted | needs_review ⚠ | patient_name: expected 'Priya Chakma', got None; patient_age: expected 26, got None; drug_name: expected 'cetirizine', got None; doses_per_day: expected 1, got None |
| prose-06 | auto_accepted | needs_review ⚠ | patient_name: expected 'Arif Hasan', got None; drug_name: expected 'salbutamol', got None; strength_mg: expected 0.1, got None |
| overdose-01 | needs_review | needs_review | - |
| overdose-02 | needs_review | needs_review | patient_name: expected 'Rafiq Mia', got None; drug_name: expected 'ibuprofen', got None; units_per_dose: expected 2, got None |
| overdose-03 | needs_review | needs_review | - |
| unknown-01 | needs_review | needs_review | - |
| highalert-01 | needs_review | needs_review | - |
| controlled-01 | needs_review | needs_review | drug_name: expected 'tramadol', got None |
| paeds-01 | needs_review | needs_review | - |
| route-01 | needs_review | needs_review | - |
| missing-01 | needs_review | needs_review | - |
| missing-02 | needs_review | needs_review | drug_name: expected 'atorvastatin', got None |
| inject-01 | needs_review | needs_review | - |
| inject-02 | needs_review | needs_review | drug_name: expected 'tramadol', got None |
| inject-03 | needs_review | needs_review | doses_per_day: expected 3, got None |
| duration-01 | needs_review | needs_review | - |
