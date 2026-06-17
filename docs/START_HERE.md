# Breast Density Benchmark Tool: Start Here

## What to build first

Start with a private benchmark evaluator, not direct mammogram upload or a login system.

The website should let interns upload a small predictions CSV:

```csv
image_id,prediction
sample_001,A
sample_002,C
```

The backend compares those predictions with a private local ground-truth file that never leaves your machine or server. This is the cleanest first version because the EMBED images are large and should not be exposed through the public site.

For your current scope, build only these features:

- upload prediction CSV
- evaluate metrics
- save evaluation history
- show leaderboard with model name and accuracy

## Recommended architecture

- Frontend: React/Vite dashboard using Poppins and the palette `#14868C`, `#DAF3F4`, `#FDFCFC`.
- Backend: FastAPI evaluation API.
- Private storage: local folder or internal server path for selected EMBED manifest, labels, and optional image paths.
- Evaluation contract: interns submit predictions for fixed `image_id` values.
- Results: primary metric is macro F1; secondary metrics are accuracy, balanced accuracy, weighted F1, and optional quadratic kappa.

## First milestone

1. Filter the EMBED metadata using your criteria:
   - `FinalImageType = 2D`
   - `ViewPosition = CC/MLO`
   - `spot_mag` blank
   - `XRayTubeCurrent >= 100`
   - `BitsStored = 12.0`
   - `PatientSex = F`
   - `BreastImplantPresent = NO`

2. Create a benchmark split:
   - `image_id`
   - image path, kept private
   - true density label
   - optional patient/study identifiers for leakage checks

3. Publish only the `image_id` list or a feature file, depending on how interns are expected to run models.

4. Ask interns to upload predictions with exactly:

   ```csv
   image_id,prediction
   ```

5. Store each run result in history and update the leaderboard.

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
