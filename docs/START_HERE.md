# Breast Density Benchmark Tool: Start Here

## What to build first

Start with a private benchmark evaluator, not direct mammogram upload or a login system.

The website should let interns upload a small predictions CSV:

```csv
image_id,predicted_label
subject_0001,A
subject_0002,C
```

The backend compares those predictions with a private local ground-truth file that never leaves your machine or server.

For your current scope, build only these features:

- upload prediction CSV
- evaluate metrics
- save evaluation history
- show leaderboard with model name and accuracy

## Recommended architecture

- Frontend: React/Vite dashboard using Poppins and the palette `#14868C`, `#DAF3F4`, `#FDFCFC`.
- Backend: FastAPI evaluation API.
- Private storage: local folder or internal server path for benchmark manifest, labels, and optional image paths.
- Evaluation contract: interns submit predictions for fixed `image_id` values.
- Results: primary metric is macro F1; secondary metrics are accuracy, balanced accuracy, weighted F1, and optional quadratic kappa.

## First milestone

1. Create a benchmark split:
   - `image_id`
   - image path, kept private
   - true density label
   - optional patient/study identifiers for leakage checks

2. Publish only the `image_id` list or a feature file, depending on how interns are expected to run models.

3. Ask interns to upload predictions with exactly:

   ```csv
   image_id,predicted_label
   subject_0001,C
   subject_0002,B
   ```

4. Store each run result in history and update the leaderboard.

## Later milestones

- Leaderboard with model name, owner, date, macro F1, and secondary metrics.
- Model showcase page with method summaries.
- Optional model-container evaluation where interns upload a Docker image or inference script, and your server runs it against private images.
- Admin-only dataset split builder and ground-truth manager.

## Logo assets

Create the folder:

```bash
mkdir -p frontend/public/logos
```

Place these files in `frontend/public/logos/`:

- `tanuh_logo.png`
- `moe_logo.png`
- `iisc_logo.png`

The starter frontend already displays them side by side on a pale aqua strip.
