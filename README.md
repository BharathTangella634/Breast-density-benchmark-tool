# Breast-density-benchmark-tool
Evaluating the models on Mammography images for Breast density classification

This repository is starting as a private benchmark platform for breast density classification models.

Read [docs/START_HERE.md](docs/START_HERE.md) for the architecture and first milestones.

## Starter structure

- `backend/`: FastAPI API for evaluating prediction CSV files against hidden local labels.
- `frontend/`: React/Vite dashboard styled with the requested palette and logo strip.
- `docs/`: project planning notes and implementation milestones.

## Core idea
On the webiste, lightweight prediction CSV will be submitted and the backend should compute:

- Primary: macro F1
- Secondary: accuracy, balanced accuracy, weighted F1
- Optional: quadratic kappa
