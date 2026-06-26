#!/usr/bin/env bash
set -euo pipefail

# VM initialization script for Breast Density Benchmark Tool.
# Copies benchmark data from GCloud Storage bucket to local VM filesystem.
# Idempotent — safe to re-run on VM rebuild.
#
# Usage:
#   bash scripts/init_vm.sh
#
# Required env vars:
#   GCS_BUCKET        — GCloud Storage bucket name (e.g. "my-benchmark-data")
#   GCS_DATA_PREFIX   — prefix inside bucket (default: "benchmark")
#   DATA_DIR          — local directory for data (default: "/opt/benchmark/data")

GCS_BUCKET="${GCS_BUCKET:?Set GCS_BUCKET to the GCloud Storage bucket name}"
GCS_DATA_PREFIX="${GCS_DATA_PREFIX:-benchmark}"
DATA_DIR="${DATA_DIR:-/opt/benchmark/data}"

echo "==> Initializing benchmark data from gs://${GCS_BUCKET}/${GCS_DATA_PREFIX}/"

# Create directories
mkdir -p "${DATA_DIR}/images"
mkdir -p "${DATA_DIR}/onnx_uploads"
mkdir -p "${DATA_DIR}/tensor_cache"

# Copy benchmark images
echo "==> Copying benchmark images..."
gsutil -m cp -r "gs://${GCS_BUCKET}/${GCS_DATA_PREFIX}/images/*" "${DATA_DIR}/images/"

# Copy ground truth labels
echo "==> Copying ground truth labels..."
gsutil cp "gs://${GCS_BUCKET}/${GCS_DATA_PREFIX}/benchmark_labels_private.csv" "${DATA_DIR}/benchmark_labels_private.csv"

# Copy public manifest
echo "==> Copying public manifest..."
gsutil cp "gs://${GCS_BUCKET}/${GCS_DATA_PREFIX}/benchmark_test_public.csv" "${DATA_DIR}/benchmark_test_public.csv"

echo "==> Data initialization complete."
echo "    Images:      ${DATA_DIR}/images/ ($(ls "${DATA_DIR}/images/" | wc -l) files)"
echo "    Labels:      ${DATA_DIR}/benchmark_labels_private.csv"
echo "    Manifest:    ${DATA_DIR}/benchmark_test_public.csv"
echo ""
echo "Set these in your backend .env:"
echo "    BENCHMARK_GROUND_TRUTH_CSV=${DATA_DIR}/benchmark_labels_private.csv"
echo "    BENCHMARK_IMAGE_MANIFEST_CSV=${DATA_DIR}/benchmark_test_public.csv"
echo "    BENCHMARK_IMAGE_ROOT=${DATA_DIR}/images"
echo "    BENCHMARK_ONNX_UPLOAD_DIR=${DATA_DIR}/onnx_uploads"
