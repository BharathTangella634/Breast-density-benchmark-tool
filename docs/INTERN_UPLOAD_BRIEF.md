# Intern Upload Brief

Upload one prediction CSV for the benchmark test set.

Required columns:

```csv
image_id,predicted_label
subject_0001,C
subject_0002,B
```

Allowed labels:

- `A` = Density A (almost entirely fatty)
- `B` = Density B (scattered density)
- `C` = Density C (heterogeneously dense)
- `D` = Density D (extremely dense)

Rules:

- Use exact `image_id` values from the public manifest (e.g. `subject_0001`, `subject_0002`)
- Include predictions for all 200 benchmark images — partial submissions are rejected
- One prediction per image — no duplicate `image_id` values
- Labels must be uppercase A, B, C, or D
- Do not include extra columns — only `image_id` and `predicted_label`
