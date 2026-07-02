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
subject_0001,A
subject_0002,C
```

Prediction uploads should contain:

```csv
image_id,predicted_label
subject_0001,A
subject_0002,B
```

## API summary

- `POST /api/evaluate`: upload a prediction CSV and save the result.
- `GET /api/history`: read the saved evaluation history.
- `GET /api/leaderboard`: read the best score per model.
