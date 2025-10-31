# Dataset preparation helper for DIP Detector

This script provides guidance and optional utilities to fetch and prepare a deepfake dataset.

Recommended options
1) FaceForensics++ (requires manual download/permissions)
2) Kaggle: Deepfake Detection Challenge (requires Kaggle account and kaggle CLI)

Expected folder structure (ImageFolder style):

your_dataset/
  train/
    real/
    fake/
  val/
    real/
    fake/

Usage examples
- Organize frames from videos into images per class (extract I-frames with OpenCV)
- Optionally downsample frames and ensure balanced classes

CLI
- Extract frames from a folder of videos into images per class:
  python prepare_dataset.py extract-frames --videos_dir path/to/videos --out_dir datasets/your_dataset/train --fps 1 --img_size 300 --label real

- Split into train/val (stratified by folder):
  python prepare_dataset.py split --root datasets/your_dataset --val_ratio 0.2

Note: Large public datasets may be hundreds of GB. Ensure you have disk space.
