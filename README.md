# DIP Detector

A full-stack project to detect whether an uploaded image or video is real or AI-generated (deepfake).

Repository structure:
- /frontend  → React + Tailwind + Framer Motion UI
- /backend   → Flask API + model + training pipeline
- /models    → Saved model checkpoints
- /data      → scripts for dataset / sample media

Quick start

Frontend
- cd frontend
- npm install
- npm start

Backend
- cd backend
- python -m venv venv
- venv/Scripts/activate (Windows) or source venv/bin/activate (macOS/Linux)
- pip install -r requirements.txt
- python app.py

Endpoints
- POST /upload → accepts image or video, saves file, runs prediction
- GET  /result/<id> → returns verdict + confidence + heatmap (image) / timeline (video)
- POST /train → starts training in background

Training
- python train.py --data_dir "../data/datasets/your_dataset" --epochs 30 --batch_size 32 --img_size 224
- Dataset folder must follow ImageFolder format:
  your_dataset/
    train/
      real/
      fake/
    val/
      real/
      fake/
- Use data/prepare_dataset.py for helper utilities and instructions.

Notes
- Models saved under /models
- Database: backend/dip_detector.db (SQLite)
