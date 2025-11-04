# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common commands

- Frontend (React + CRA)
  - Install: `cd frontend && npm install`
  - Dev server: `npm start` (proxies API to http://localhost:5000)
  - Build: `npm run build`
  - Tests (Jest via react-scripts):
    - Run all: `npm test`
    - Run a single test: `npm test -- -t "<pattern>"`

- Backend (Flask API + PyTorch)
  - Create venv and install deps:
    - Windows (PowerShell):
      - `cd backend`
      - `python -m venv venv`
      - `./venv/Scripts/Activate.ps1`
      - `pip install -r requirements.txt`
    - macOS/Linux:
      - `cd backend`
      - `python3 -m venv venv`
      - `source venv/bin/activate`
      - `pip install -r requirements.txt`
  - Run API: `python app.py` (defaults to port 5000)
  - Optional env vars before running (examples):
    - `JWT_SECRET_KEY` (e.g., set via `$env:JWT_SECRET_KEY="{{JWT_SECRET_KEY}}"` on PowerShell)
    - Mail (for Flask-Mail if used): `MAIL_SERVER`, `MAIL_PORT`, `MAIL_USE_TLS`, `MAIL_USE_SSL`, `MAIL_USERNAME`, `MAIL_PASSWORD`
    - Model switches: `DIP_THRESHOLD` (e.g. 0.6), `DIP_INVERT` ("1" to invert logits), `DIP_SWAP_LABELS` ("1" to swap reported labels)

- Run both concurrently
  - Terminal A (backend): set env as needed, then `cd backend` → activate venv → `python app.py`
  - Terminal B (frontend): `cd frontend` → `npm start`

- Model training (optional)
  - From `backend/`: `python train.py --data_dir "../data/datasets/your_dataset" --epochs 30 --batch_size 32 --img_size 224 --model_dir ..\models`
  - After training, a calibrated threshold is saved to `models/threshold.txt`; best weights to `models/best.pt`.
  - You can also trigger background training via API: `POST /train` with JSON `{ "data_dir": "...", "epochs": 20, "batch_size": 32, "img_size": 224 }`.

## Architecture overview

- Repo layout
  - `frontend/`: React application (Create React App) with Tailwind and Framer Motion. `package.json` sets `proxy` to `http://localhost:5000` for API calls during development.
  - `backend/`: Flask app exposing REST endpoints, local SQLite storage, and a PyTorch-based deepfake detector.
  - `models/`: persisted model artifacts (`best.pt`, `threshold.txt`).
  - `data/`: dataset notes and helpers.

- Backend API (key flows)
  - File ingest and inference: `POST /upload`
    - Accepts image or video, stores under `backend/uploads/`, runs model inference.
    - Images may return a saved Grad-CAM heatmap overlay path.
    - Videos are sampled into frames and averaged with temporal smoothing.
  - Results retrieval: `GET /result/<id>` → returns label, confidence, and any derived artifacts.
  - History: `GET /history` (optional JWT) → recent items from SQLite.
  - Dashboard: `GET /dashboard` (JWT required) → aggregate stats (totals, distribution, recents).
  - PDF report: `GET /report/<id>` → generates a report in `uploads/` and streams it.
  - Training: `POST /train` → spawns a background `train.py` process.

- Backend internals (big picture)
  - `app.py`: Flask app configuration (CORS, JWT, optional mail), route handlers, and model lifecycle. Uses environment flags (`DIP_THRESHOLD`, `DIP_INVERT`, `DIP_SWAP_LABELS`) to control inference behavior.
  - `deepfake_model.py`: Loads an EfficientNet-B3 head (optionally partial ResNet50) and applies TTA (rotations + hflip). Image preprocessing is robust to missing Albumentations. Optional Grad-CAM support if installed. For videos, per-frame probabilities are smoothed temporally before thresholding.
  - `database.py`: SQLite schema and helpers for history and user records (hashed passwords via `passlib`). Simple OTP remains for compatibility but registration defaults to verified.
  - `utils.py`: Video frame extraction (OpenCV), EXIF/metadata parsing for generator hints, heatmap overlay generation, and PDF report creation (ReportLab).
  - `train.py`: EfficientNet-B3 training pipeline with warmup/fine-tuning, mixup, optional focal loss, EMA, class-balanced sampling, cosine LR schedule, early stopping, and post-training threshold calibration written to `models/threshold.txt`.

- Frontend (big picture)
  - CRA-powered React UI with Tailwind; uses the CRA dev proxy to hit the Flask API at `http://localhost:5000` in development. Typical flow: upload → poll/display result → optional download report → show history/dashboard (JWT-aware endpoints).

## Notes & gotchas

- SQLite DB lives at `backend/dip_detector.db` and is initialized on app startup.
- Uploads are stored under `backend/uploads/`; heatmaps for images are saved alongside the original filename with `_heatmap` suffix.
- If `albumentations` or `pytorch-grad-cam` are missing, inference still works using fallback preprocessing and disables CAM generation.
- For production builds, the frontend outputs to `frontend/build/`; ensure CORS and API base URL are configured appropriately if not using the CRA dev proxy.
