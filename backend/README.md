# Backend

FastAPI service for private breast-density benchmark evaluation.

## Local setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
BENCHMARK_GROUND_TRUTH_CSV=/path/to/private/ground_truth.csv uvicorn app.main:app --reload
```

Optional history database location:

```bash
BENCHMARK_HISTORY_DB=/path/to/private/evaluation_history.db
```

The private ground-truth CSV should contain:

```csv
image_id,true_label
sample_001,A
sample_002,C
```

Intern prediction uploads should contain:

```csv
image_id,predicted_label
subject_0001,A
subject_0002,B
```

Probability uploads are also accepted:

```csv
image_id,p0,p1,p2,p3
sample_001,0.10,0.70,0.15,0.05
sample_002,0.02,0.08,0.80,0.10
```

## Create the selected EMBED manifest

```bash
python -m app.dataset_filter --metadata /path/to/embed_metadata.csv --output data/private/selected_manifest.csv
```

The first filter applies:

- `FinalImageType = 2D`
- `ViewPosition in CC, MLO`
- `spot_mag` blank
- `XRayTubeCurrent >= 100`
- `BitsStored = 12.0`
- `PatientSex = F`
- `BreastImplantPresent = NO`

## API summary

- `POST /api/evaluate`: upload a prediction CSV and save the result.
- `GET /api/history`: read the saved evaluation history.
- `GET /api/leaderboard`: read the best score per model.
