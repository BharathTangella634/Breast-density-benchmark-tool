# Breast-density-benchmark-tool

Evaluating mammography models for breast density classification.

This repository is starting as a private benchmark platform for breast density classification models.

Read [docs/START_HERE.md](docs/START_HERE.md) for the architecture and first milestones.

## What This Tool Does

This tool lets interns evaluate breast density classification models without uploading the full mammogram datasets to a public website.

The benchmark owner keeps the datasets and hidden labels private. Interns submit only a lightweight prediction CSV. The backend compares their predictions with the hidden answer key and reports the evaluation metrics.

Primary metric:

- Macro F1

Secondary metrics:

- Accuracy
- Balanced accuracy
- Weighted F1
- Quadratic weighted kappa

## Current Benchmark Data

The current benchmark preparation uses two local datasets:

- `EMBED`: 300 selected mammogram images
- `IBIA / a_data`: 500 selected mammogram images

Both datasets use density labels:

- `A` = Density A
- `B` = Density B
- `C` = Density C
- `D` = Density D

Generated benchmark files are in:

```text
data/benchmark_prep/
```

Public files to give interns:

- `data/benchmark_prep/embed_test_public.csv`
- `data/benchmark_prep/ibia_test_public.csv`

Private answer-key files to keep hidden:

- `data/benchmark_prep/embed_labels_private.csv`
- `data/benchmark_prep/ibia_labels_private.csv`

Do not give the private label CSVs to interns.

## Intern Submission Format

Interns should upload one prediction CSV:

```csv
image_id,prediction
embed_0001,C
ibia_0001,B
```

Allowed prediction values are only:

```text
A, B, C, D
```

Probability submissions are also accepted:

```csv
image_id,p0,p1,p2,p3
embed_0001,0.10,0.20,0.60,0.10
ibia_0001,0.05,0.75,0.15,0.05
```

Here `p0,p1,p2,p3` correspond to `A,B,C,D`.

## How Evaluation Works

The backend joins the intern submission with the hidden labels using `image_id`.

Example hidden labels:

```csv
image_id,true_label
embed_0001,C
ibia_0001,B
```

Example intern submission:

```csv
image_id,prediction
embed_0001,C
ibia_0001,A
```

The website computes the metrics, saves the run to evaluation history, and updates the leaderboard.

## Project Structure

- `backend/`: FastAPI API for evaluating prediction CSV files against hidden local labels.
- `frontend/`: React/Vite dashboard styled with the requested palette and logo strip.
- `scripts/prepare_benchmark_csvs.py`: creates EMBED and IBIA benchmark CSVs.
- `docs/`: intern instructions and project planning notes.

## Regenerate Benchmark CSVs

From the repository root:

```bash
python scripts/prepare_benchmark_csvs.py
```

This regenerates:

- `embed_labels_private.csv`
- `embed_test_public.csv`
- `ibia_labels_private.csv`
- `ibia_test_public.csv`

## Local Development

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
BENCHMARK_GROUND_TRUTH_CSV=/path/to/private_labels.csv uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```
