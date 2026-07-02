# Breast Density Benchmark Tool

A private benchmark website for evaluating mammography breast-density classification models.

The benchmark data and true labels stay on the server. Users submit either a prediction CSV or an ONNX model, and the website evaluates the submission against the hidden benchmark set, stores the run, and updates the leaderboard.

## What This Tool Does

- Evaluates breast-density classification models for classes `A`, `B`, `C`, and `D`.
- Supports prediction CSV uploads.
- Supports ONNX model uploads for local server-side inference.
- Saves every evaluation run in a database (SQLite locally, Cloud SQL MySQL in production).
- Shows a leaderboard with the best score per model.
- Keeps benchmark images and ground-truth labels private.

Primary metric:

- Macro F1

Secondary metrics:

- Accuracy
- Balanced accuracy
- Weighted F1
- Quadratic weighted kappa

## Current Benchmark Data

The current benchmark is a 200-image test set prepared from CR (clinic-collected) and IBIA datasets.

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

Required format:

```csv
image_id,predicted_label
subject_0001,C
subject_0002,B
```

Rules:

- Use exact `image_id` values from the public manifest.
- Use labels only from `A`, `B`, `C`, `D` (uppercase).
- Submit predictions for all 200 benchmark images. Partial submissions are rejected.
- One prediction per image. No duplicate `image_id` values.
- Only two columns: `image_id` and `predicted_label`.

### ONNX Model

ONNX model requirements:

```text
File:   one standalone .onnx file containing the full inference pipeline
Input:  float32 grayscale mammogram tensor shaped [1,1,1024,1024]
Output: either 4 scores ordered A,B,C,D or class index 0=A,1=B,2=C,3=D
```

The ONNX file must be self-contained with all weights embedded. If your export produces a separate `.data` file, re-export with embedded weights:

- PyTorch: `torch.onnx.export(model, ..., 'model.onnx')` without external data thresholds.
- ONNX: `onnx.save(model, 'model.onnx', save_as_external_data=False)`.

If the full pipeline cannot be exported into one ONNX file, submit a prediction CSV instead.

ONNX models are queued and evaluated one at a time. The model file is saved to disk during queuing, loaded for inference, and deleted after evaluation completes. The queue survives server restarts.

The backend runs inference on the private benchmark images, converts model outputs to predicted labels, evaluates the predictions, saves the run, and updates the leaderboard.

## How Evaluation Works

For CSV submissions:

1. The backend reads the uploaded prediction CSV.
2. It joins predictions with the hidden labels using `image_id`.
3. It computes macro F1, accuracy, balanced accuracy, weighted F1, and quadratic kappa.
4. It saves the run in SQLite.
5. The leaderboard updates with the best score per model.

For ONNX submissions:

1. The backend loads the uploaded `.onnx` model.
2. It reads private benchmark images from the configured image folder.
3. It resizes/preprocesses images to the expected tensor format.
4. It runs model inference locally.
5. It evaluates the generated predictions and saves the run like a CSV submission.

Each model name must be unique. Submitting a model name that already exists on the leaderboard is rejected. Choose a different name for each submission.

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

Copy `backend/.env.example` to `backend/.env` and fill in paths:

```env
BENCHMARK_GROUND_TRUTH_CSV=./data/benchmark_prep/benchmark_labels_private.csv
BENCHMARK_HISTORY_DB=./data/private/evaluation_history.db
BENCHMARK_IMAGE_MANIFEST_CSV=./data/benchmark_prep/benchmark_test_public.csv
BENCHMARK_IMAGE_ROOT=./data/private/benchmark_test_data_png
```

For production with Cloud SQL, add:

```env
BENCHMARK_DATABASE_URL=mysql://benchmark_user:password@CLOUD_SQL_IP:3306/benchmark_db
BENCHMARK_ALLOWED_ORIGINS=["https://yourdomain.com"]
```

When `BENCHMARK_DATABASE_URL` is set, the app uses Cloud SQL (MySQL) instead of SQLite. When unset, it uses the local SQLite file at `BENCHMARK_HISTORY_DB`.

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

## Deploy to GCloud VM

### 1. Create infrastructure

- Create a GCloud VM (Ubuntu recommended).
- Create a Cloud SQL MySQL instance (or use an existing one and create a new database).
- Create a GCloud Storage bucket and upload benchmark images and labels.

### 2. Initialize VM data

Upload benchmark data to a GCloud Storage bucket, then run the init script on the VM:

```bash
GCS_BUCKET=your-bucket-name bash scripts/init_vm.sh
```

This copies images and labels from the bucket to the VM's local filesystem.

### 3. Migrate existing results

To bring local SQLite evaluation results into Cloud SQL:

```bash
python scripts/migrate_sqlite_to_cloudsql.py \
  --sqlite-path data/private/evaluation_history.db \
  --database-url mysql://benchmark_user:pass@CLOUD_SQL_IP:3306/benchmark_db
```

### 4. Configure and run

Set `backend/.env` on the VM with the Cloud SQL connection string and data paths. Then:

```bash
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

Build the frontend for production:

```bash
cd frontend
VITE_API_BASE=https://yourdomain.com npm run build
```

Serve the `frontend/dist/` folder with nginx or similar.

### 5. Map domain

Point your domain's DNS A record to the VM's public IP. Configure nginx as a reverse proxy for the API and static file server for the frontend.

## Useful API Routes

```text
POST /api/evaluate       Evaluate prediction CSV (instant)
POST /api/submit-onnx    Submit ONNX model to queue
GET  /api/job/{job_id}   Poll ONNX job status and result
GET  /api/queue          Queue info (running, waiting, avg time)
GET  /api/leaderboard    Read best score per model
GET  /api/history        Read saved evaluation runs
```
