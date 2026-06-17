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

The current benchmark preparation uses two local datasets and creates an 800-image balanced benchmark:

- `EMBED`: 432 selected mammogram images
- `IBIA / a_data`: 368 selected mammogram images

Combined class balance:

- `A`: 200 images
- `B`: 200 images
- `C`: 200 images
- `D`: 200 images

Dataset contribution by class:

```text
EMBED: A=100, B=100, C=100, D=132
IBIA:  A=100, B=100, C=100, D=68
```

IBIA has only 68 available `D` images, so the remaining `D` examples are filled from EMBED.

Both datasets use density labels:

- `A` = Density A
- `B` = Density B
- `C` = Density C
- `D` = Density D

Generated benchmark files are in:

```text
data/benchmark_prep/
```

Public file to give interns:

- `data/benchmark_prep/benchmark_test_public.csv`

This is the only CSV interns need for the evaluation round.

Private answer-key file to keep hidden:

- `data/benchmark_prep/benchmark_labels_private.csv`

Do not give the private label CSV to interns.

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

- `benchmark_labels_private.csv`
- `benchmark_test_public.csv`

## Run Real Evaluation

Start the backend with the combined hidden labels:

```bash
cd backend
source .venv/bin/activate
BENCHMARK_GROUND_TRUTH_CSV=/home/tanuh/EMBED/Breast-density-benchmark-tool/data/benchmark_prep/benchmark_labels_private.csv uvicorn app.main:app --reload
```

Then start the frontend:

```bash
cd frontend
npm run dev
```

When an intern sends a prediction CSV, enter the model name in the website, upload the CSV, and click evaluate. The run will be saved in history and reflected in the leaderboard.

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
