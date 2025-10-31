from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_mail import Mail, Message
import os
import sys
import uuid
import threading
import time
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from database import init_db, insert_record, get_record, get_all_records, create_user, create_user_verified, get_user_by_email, verify_user_otp, validate_login, dashboard_stats
from deepfake_model import DeepfakeModel
from utils import allowed_file, is_video_file, extract_video_frames, save_heatmap_overlay

UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), 'models')
DB_PATH = os.path.join(BASE_DIR, 'dip_detector.db')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'change-me')
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', '587'))
app.config['MAIL_USE_TLS'] = str(os.environ.get('MAIL_USE_TLS', 'true')).lower() == 'true'
app.config['MAIL_USE_SSL'] = str(os.environ.get('MAIL_USE_SSL', 'false')).lower() == 'true'
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = app.config.get('MAIL_USERNAME')
mail = Mail(app)
jwt = JWTManager(app)

# Initialize DB
init_db(DB_PATH)
# Load model once
model = DeepfakeModel(model_dir=MODEL_DIR)

@app.route('/')
def index():
    return jsonify({"status": "ok", "service": "DIP Detector API"})

@app.route('/auth/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    email = (data.get('email') or '').strip(); password = str(data.get('password') or '')
    if not email or not password:
        return jsonify({"error": "email and password required"}), 400
    if len(password) < 6:
        return jsonify({"error": "password must be at least 6 characters"}), 400
    if get_user_by_email(DB_PATH, email):
        return jsonify({"error": "User exists"}), 400
    try:
        create_user_verified(DB_PATH, email, password)
        return jsonify({"status": "created"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# OTP verification no longer required; keep endpoint for compatibility
@app.route('/auth/verify', methods=['POST'])
def verify():
    return jsonify({"status": "disabled"})

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    email = (data.get('email') or '').strip(); password = str(data.get('password') or '')
    if not validate_login(DB_PATH, email, password):
        return jsonify({"error": "Invalid credentials or not verified"}), 401
    token = create_access_token(identity=email)
    return jsonify({"access_token": token})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    uid = str(uuid.uuid4())
    filename = secure_filename(f.filename)
    ext = os.path.splitext(filename)[1].lower()
    save_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
    f.save(save_path)

    try:
        if is_video_file(save_path):
            frames = extract_video_frames(save_path, max_frames=48)
            if len(frames) == 0:
                return jsonify({"error": "Could not read video frames"}), 400
            frame_scores, final_score, label = model.predict_video(frames)
            record_id = insert_record(DB_PATH, save_path, label, float(final_score))
            return jsonify({
                "id": record_id,
                "file_id": uid,
                "label": label,
                "confidence": round(float(final_score) * 100, 2),
                "timeline": frame_scores
            })
        else:
            score, label, heatmap = model.predict_image_file(save_path, return_cam=True)
            heatmap_path = None
            if heatmap is not None:
                heatmap_path = save_heatmap_overlay(save_path, heatmap)
            record_id = insert_record(DB_PATH, save_path, label, float(score))
            resp = {
                "id": record_id,
                "file_id": uid,
                "label": label,
                "confidence": round(float(score) * 100, 2)
            }
            if heatmap_path:
                resp["heatmap_path"] = os.path.basename(heatmap_path)
            return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/result/<int:rec_id>', methods=['GET'])
def result(rec_id):
    rec = get_record(DB_PATH, rec_id)
    if not rec:
        return jsonify({"error": "Not found"}), 404
    file_path, label, conf, ts = rec

    resp = {
        "id": rec_id,
        "file_path": file_path,
        "label": label,
        "confidence": round(float(conf) * 100, 2),
        "timestamp": ts
    }

    if not is_video_file(file_path):
        # Try to find heatmap in uploads dir
        base = os.path.splitext(os.path.basename(file_path))[0]
        for cand in os.listdir(UPLOAD_DIR):
            if cand.startswith(base) and 'heatmap' in cand:
                resp['heatmap_path'] = cand
                break
    else:
        # For videos, provide frame timeline if cached in sidecar json (optional)
        pass

    return jsonify(resp)

@app.route('/history', methods=['GET'])
@jwt_required(optional=True)
def history():
    rows = get_all_records(DB_PATH, limit=int(request.args.get('limit', 100)))
    items = []
    for (rid, file_path, label, conf, ts) in rows:
        items.append({
            'id': rid,
            'file_path': file_path,
            'label': label,
            'confidence': float(conf),
            'timestamp': ts,
        })
    return jsonify({'items': items})

@app.route('/dashboard', methods=['GET'])
@jwt_required()
def dashboard():
    _ = get_jwt_identity()
    stats = dashboard_stats(DB_PATH)
    return jsonify(stats)

@app.route('/report/<int:rec_id>', methods=['GET'])
# Optional auth; leave open for quick download
def report(rec_id):
    from utils import generate_pdf_report
    rec = get_record(DB_PATH, rec_id)
    if not rec:
        return jsonify({"error": "Not found"}), 404
    file_path, label, conf, ts = rec
    out_path = generate_pdf_report(UPLOAD_DIR, rec_id, file_path, label, float(conf), ts)
    return send_file(out_path, as_attachment=True, download_name=f"dip_report_{rec_id}.pdf")

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)

@app.route('/train', methods=['POST'])
@jwt_required(optional=True)
def train():
    """Starts training in a background thread using train.py"""
    args = request.get_json(silent=True) or {}
    data_dir = args.get('data_dir', os.path.join(os.path.dirname(BASE_DIR), 'data', 'datasets'))
    epochs = int(args.get('epochs', 20))
    batch_size = int(args.get('batch_size', 32))
    img_size = int(args.get('img_size', 224))

    def _bg_train():
        import subprocess, sys
        cmd = [sys.executable, os.path.join(BASE_DIR, 'train.py'),
               '--data_dir', data_dir,
               '--epochs', str(epochs),
               '--batch_size', str(batch_size),
               '--img_size', str(img_size),
               '--model_dir', MODEL_DIR]
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            pass

    threading.Thread(target=_bg_train, daemon=True).start()
    return jsonify({"status": "started", "params": {"data_dir": data_dir, "epochs": epochs, "batch_size": batch_size, "img_size": img_size}})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
