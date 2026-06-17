# Intern Upload Brief

Upload one prediction CSV for the benchmark test set.

Required columns:

```csv
image_id,prediction
embed_0001,C
ibia_0001,B
```

Allowed labels:

- `A` = Density A
- `B` = Density B
- `C` = Density C
- `D` = Density D

Probability format is also accepted:

```csv
image_id,p0,p1,p2,p3
embed_0001,0.10,0.20,0.60,0.10
ibia_0001,0.05,0.75,0.15,0.05
```

Rules:

- Do not change `image_id`
- Do not skip rows
- Submit exactly one prediction per image
- Upload only the CSV, not the dataset
