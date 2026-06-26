# Intern Upload Brief

Upload one prediction CSV for the benchmark test set.

Required columns:

```csv
image_id,predicted_label
embed_0001,C
ibia_0001,B
```

Allowed labels:

- `A` = Density A (almost entirely fatty)
- `B` = Density B (scattered density)
- `C` = Density C (heterogeneously dense)
- `D` = Density D (extremely dense)

Rules:

- Use exact `image_id` values from the public manifest (e.g. `embed_0001`, `ibia_0001`)
- Include predictions for all 800 benchmark images — partial submissions are rejected
- One prediction per image — no duplicate `image_id` values
- Labels must be uppercase A, B, C, or D
- Do not include extra columns — only `image_id` and `predicted_label`
