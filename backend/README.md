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
sample_001,A
sample_002,B
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
