# Breast Density Benchmark Tool

A private benchmark website for evaluating mammography breast-density classification models.

The benchmark data and true labels stay on the server. Users submit either a prediction CSV or an ONNX model, and the website evaluates the submission against the hidden benchmark set, stores the run, and updates the leaderboard.

## What This Tool Does

- Evaluates breast-density classification models for classes `A`, `B`, `C`, and `D`.
- Supports prediction CSV uploads.
- Supports ONNX model uploads for local server-side inference.
- Saves every evaluation run in a local SQLite history database.
- Shows a leaderboard with the best score per model.
- Keeps benchmark images and ground-truth labels private.

Primary metric:

- Macro F1

Secondary metrics:

- Accuracy
- Balanced accuracy
- Weighted F1
- Macro precision
- Macro recall
- Quadratic weighted kappa

## Current Benchmark Data

The current benchmark is a balanced 800-image test set prepared from EMBED and IBIA.

Class balance:

```text
A: 200 images
B: 200 images
C: 200 images
D: 200 images
```

Dataset contribution:

```text
EMBED: 432 images
IBIA:  368 images
Total: 800 images
```

Generated benchmark CSV files:

```text
data/benchmark_prep/benchmark_test_public.csv
data/benchmark_prep/benchmark_labels_private.csv
```

Give users only:

```text
data/benchmark_prep/benchmark_test_public.csv
```

Keep private:

```text
data/benchmark_prep/benchmark_labels_private.csv
data/private/
backend/.env
```

`data/private/` is ignored by git and is intended for local benchmark images, the SQLite history database, and other private runtime files.

## Submission Requirements

The website accepts one submission path at a time.

### Prediction CSV

Label format:

```csv
image_id,prediction
embed_0001,C
ibia_0001,B
```

Probability format:

```csv
image_id,p0,p1,p2,p3
embed_0001,0.10,0.20,0.60,0.10
ibia_0001,0.05,0.75,0.15,0.05
```

For probability CSVs, columns `p0,p1,p2,p3` correspond to `A,B,C,D`.

Rules:

- Keep `image_id` unchanged.
- Use labels only from `A`, `B`, `C`, `D`.
- Submit one prediction per benchmark image.
- Do not include true labels in the submitted CSV.

### ONNX Model

ONNX model requirements:

```text
File:   one standalone .onnx file containing the full inference pipeline
Input:  float32 grayscale mammogram tensor shaped [1,1,1024,1024]
Output: either 4 scores ordered A,B,C,D or class index 0=A,1=B,2=C,3=D
```

The ONNX file must contain all steps required for inference after receiving the benchmark input tensor, so the backend can run it directly and obtain the final A/B/C/D prediction. If the full pipeline cannot be exported into one ONNX file, submit a prediction CSV instead.

The backend runs inference on the private benchmark images, converts model outputs to predicted labels, evaluates the predictions, saves the run, and updates the leaderboard.

## How Evaluation Works

For CSV submissions:

1. The backend reads the uploaded prediction CSV.
2. It joins predictions with the hidden labels using `image_id`.
3. It computes macro F1, accuracy, balanced accuracy, weighted F1, precision, recall, and quadratic kappa.
4. It saves the run in SQLite.
5. The leaderboard updates with the best score per model.

For ONNX submissions:

1. The backend loads the uploaded `.onnx` model.
2. It reads private benchmark images from the configured image folder.
3. It resizes/preprocesses images to the expected tensor format.
4. It runs model inference locally.
5. It evaluates the generated predictions and saves the run like a CSV submission.

Uploading the same model name or same filename again creates a new run. The leaderboard still shows the best score per model and the run count increases.

## Project Structure

```text
backend/                  FastAPI evaluation API
backend/app/evaluation.py CSV metrics and validation
backend/app/onnx_inference.py ONNX inference on private images
backend/app/history.py    SQLite history and leaderboard
frontend/                 React/Vite website
scripts/                  Benchmark preparation utilities
docs/                     Planning and upload notes
data/benchmark_prep/      Public benchmark manifest and private labels
data/private/             Private local data, ignored by git
```

## Environment Variables

Create `backend/.env` locally:

```env
BENCHMARK_GROUND_TRUTH_CSV=/home/tanuh/EMBED/Breast-density-benchmark-tool/data/benchmark_prep/benchmark_labels_private.csv
BENCHMARK_HISTORY_DB=/home/tanuh/EMBED/Breast-density-benchmark-tool/data/private/evaluation_history.db
BENCHMARK_IMAGE_MANIFEST_CSV=/home/tanuh/EMBED/Breast-density-benchmark-tool/data/benchmark_prep/benchmark_test_public.csv
BENCHMARK_IMAGE_ROOT=/home/tanuh/EMBED/Breast-density-benchmark-tool/data/private/benchmark_test_data_png
BENCHMARK_ONNX_INPUT_SIZE=1024
BENCHMARK_ONNX_INPUT_CHANNELS=1
BENCHMARK_MAX_CSV_UPLOAD_MB=25
BENCHMARK_MAX_ONNX_UPLOAD_MB=750
BENCHMARK_ONNX_TIMEOUT_SECONDS=3600
```

Do not commit `backend/.env`.

## Run Locally

Backend:

From the repository root:

```bash
source backend/.venv/bin/activate
uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

Or from inside the `backend/` folder:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173/
```

The frontend expects the backend at:

```text
http://127.0.0.1:8000
```

## Prepare Private ONNX Images

If the benchmark test files are DICOMs, convert them to PNGs for ONNX inference:

```bash
python scripts/convert_benchmark_dicoms_to_png.py \
  --manifest data/benchmark_prep/benchmark_test_public.csv \
  --dicom-root /path/to/benchmark_test_data \
  --output-dir data/private/benchmark_test_data_png
```

The converted PNG folder should stay private and should not be pushed to GitHub.

## Git Notes

Safe to push:

```text
backend/app/
backend/requirements.txt
frontend/src/
frontend/public/
scripts/
docs/
README.md
.gitignore
data/benchmark_prep/benchmark_test_public.csv
```

Do not push:

```text
backend/.env
backend/.venv/
frontend/node_modules/
frontend/dist/
data/private/
submissions/
local SQLite databases
temporary lock files
```

Before committing, check:

```bash
git status
```

## Useful API Routes

```text
POST /api/evaluate       Evaluate prediction CSV
POST /api/evaluate-onnx  Evaluate ONNX model
GET  /api/leaderboard    Read best score per model
GET  /api/history        Read saved evaluation runs
```
