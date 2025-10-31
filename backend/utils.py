import os
import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTS.union(VIDEO_EXTS)


def is_video_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def extract_video_frames(path: str, max_frames: int = 48, img_size: int = 224):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, frame_count - 1), num=min(max_frames, frame_count), dtype=int) if frame_count > 0 else []
    frames = []
    for target_idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def save_heatmap_overlay(image_path: str, heatmap: np.ndarray) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(os.path.dirname(image_path), f"{base}_heatmap.png")
    # Read original
    orig = Image.open(image_path).convert('RGB')
    orig_np = np.array(orig)
    hm = (heatmap * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    hm_color = cv2.resize(hm_color, (orig_np.shape[1], orig_np.shape[0]))
    overlay = (0.5 * orig_np + 0.5 * hm_color).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)
    return out_path


def generate_pdf_report(upload_dir: str, rec_id: int, file_path: str, label: str, confidence: float, ts: str) -> str:
    out_path = os.path.join(upload_dir, f"report_{rec_id}.pdf")
    c = canvas.Canvas(out_path, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2*cm, h-2*cm, "DEEP Detector Report")
    c.setFont("Helvetica", 12)
    c.drawString(2*cm, h-3*cm, f"Record ID: {rec_id}")
    c.drawString(2*cm, h-4*cm, f"File: {os.path.basename(file_path)}")
    c.drawString(2*cm, h-5*cm, f"Label: {label}")
    c.drawString(2*cm, h-6*cm, f"Confidence: {round(confidence*100,2)}%")
    c.drawString(2*cm, h-7*cm, f"Timestamp: {ts}")
    # Try to embed heatmap if exists
    base = os.path.splitext(os.path.basename(file_path))[0]
    heatmap_path = None
    for cand in os.listdir(upload_dir):
        if cand.startswith(base) and 'heatmap' in cand:
            heatmap_path = os.path.join(upload_dir, cand)
            break
    if heatmap_path and os.path.isfile(heatmap_path):
        try:
            c.drawImage(heatmap_path, 2*cm, 2*cm, width=10*cm, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass
    c.showPage()
    c.save()
    return out_path
