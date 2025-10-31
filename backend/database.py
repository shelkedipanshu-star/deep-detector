import os
import sqlite3
from datetime import datetime

def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            ts TEXT NOT NULL
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            verified INTEGER DEFAULT 0,
            otp TEXT,
            otp_expiry TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def insert_record(db_path: str, file_path: str, label: str, confidence: float) -> int:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    ts = datetime.utcnow().isoformat()
    c.execute(
        "INSERT INTO history (file_path, label, confidence, ts) VALUES (?, ?, ?, ?)",
        (file_path, label, float(confidence), ts),
    )
    conn.commit()
    rec_id = c.lastrowid
    conn.close()
    return rec_id


def get_record(db_path: str, rec_id: int):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT file_path, label, confidence, ts FROM history WHERE id = ?", (rec_id,))
    row = c.fetchone()
    conn.close()
    return row


def get_all_records(db_path: str, limit: int = 100):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, file_path, label, confidence, ts FROM history ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

from passlib.hash import pbkdf2_sha256
from datetime import datetime, timedelta

def _truncate_password(pw: str) -> str:
    # bcrypt only uses first 72 bytes; enforce truncation deterministically
    if not isinstance(pw, str):
        pw = str(pw)
    b = pw.encode('utf-8')
    if len(b) <= 72:
        return pw
    b = b[:72]
    # ensure valid utf-8 string; drop partial codepoints if needed
    return b.decode('utf-8', errors='ignore')


def create_user(db_path: str, email: str, password: str, otp: str):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    pw = _truncate_password(password)
    ph = pbkdf2_sha256.hash(pw)
    exp = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
    c.execute("INSERT INTO users (email, password_hash, verified, otp, otp_expiry) VALUES (?,?,?,?,?)",
              (email, ph, 0, otp, exp))
    conn.commit(); conn.close()


def create_user_verified(db_path: str, email: str, password: str):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    pw = _truncate_password(password)
    ph = pbkdf2_sha256.hash(pw)
    c.execute("INSERT INTO users (email, password_hash, verified, otp, otp_expiry) VALUES (?,?,?,?,?)",
              (email, ph, 1, None, None))
    conn.commit(); conn.close()


def get_user_by_email(db_path: str, email: str):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, email, password_hash, verified, otp, otp_expiry FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    return row


def verify_user_otp(db_path: str, email: str, otp: str) -> bool:
    user = get_user_by_email(db_path, email)
    if not user:
        return False
    uid, em, ph, ver, uotp, expiry = user
    try:
        if uotp == otp and datetime.utcnow() <= datetime.fromisoformat(expiry):
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("UPDATE users SET verified=1, otp=NULL, otp_expiry=NULL WHERE email=?", (email,))
            conn.commit(); conn.close()
            return True
    except Exception:
        return False
    return False


def validate_login(db_path: str, email: str, password: str) -> bool:
    user = get_user_by_email(db_path, email)
    if not user:
        return False
    uid, em, ph, ver, uotp, expiry = user
    if ver != 1:
        return False
    pw = _truncate_password(password)
    return pbkdf2_sha256.verify(pw, ph)


def dashboard_stats(db_path: str):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(*), AVG(confidence) FROM history")
    total, avg_conf = c.fetchone()
    c.execute("SELECT label, COUNT(*) FROM history GROUP BY label")
    dist = c.fetchall()
    c.execute("SELECT id, file_path, label, confidence, ts FROM history ORDER BY id DESC LIMIT 10")
    recent = c.fetchall()
    conn.close()
    return {
        'total': int(total or 0),
        'avg_confidence': float(avg_conf or 0.0),
        'distribution': {k: v for k, v in dist},
        'recent': [
            {'id': rid, 'file_path': fp, 'label': lab, 'confidence': float(cf), 'timestamp': ts}
            for (rid, fp, lab, cf, ts) in recent
        ]
    }
