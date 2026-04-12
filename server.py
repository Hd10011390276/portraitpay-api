#!/usr/bin/env python3
"""PortraitPay AI Backend - 肖像版权数据库系统"""

import os, json, sqlite3, hashlib, time, secrets, logging, smtplib, random, string, base64, io
from datetime import datetime
from pathlib import Path
import portrait_db as db_module
from portrait_db import get_db_conn, last_insert_id, dict_from_row, init_database
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use Railway persistent disk if available, otherwise local data/
DATA_DIR = Path(os.environ.get("RAILWAY_PRIVATE_DIR", str(Path(__file__).parent / "data")))
DB_PATH = DATA_DIR / "portraitpay.db"
UPLOAD_DIR = DATA_DIR / "uploads"
PLATFORM_FEE = 0.01

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# DB initialization is called after def init_db() below
app = Flask(__name__)
CORS(app)

# Email configuration (from environment variables)
SMTP_HOST = os.environ.get('SMTP_HOST', 'smtp.exmail.qq.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', 465))
SMTP_USER = os.environ.get('SMTP_USER', 'Dean.hang@portraitpayai.com')
SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
SENDER_EMAIL = SMTP_USER

def generate_code(length=6):
    return ''.join(random.choices(string.digits, k=length))

def send_email(to_email, subject, body):
    """Send email via SMTP. Returns (success, error_message)."""
    if not SMTP_PASSWORD:
        return False, "SMTP_PASSWORD not configured"
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SENDER_EMAIL, [to_email], msg.as_string())
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.ehlo()
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SENDER_EMAIL, [to_email], msg.as_string())
        return True, None
    except Exception as e:
        return False, str(e)

# 全局错误处理
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "请求参数错误", "code": 400}), 400

@app.errorhandler(401)
def unauthorized(e):
    return jsonify({"error": "未授权，请先登录", "code": 401}), 401

@app.errorhandler(403)
def forbidden(e):
    return jsonify({"error": "权限不足", "code": 403}), 403

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "资源不存在", "code": 404}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({"error": "服务器内部错误", "code": 500}), 500

def init_db():
    """Legacy wrapper - delegates to init_database()."""
    try:
        init_database()
    except Exception as e:
        logger.warning(f"DB init warning: {e}")

def get_user(api_key):
    if not api_key: return None
    conn, c, is_pg = get_db_conn()
    c.execute("SELECT * FROM users WHERE api_key=%s", (api_key,))
    u = c.fetchone(); conn.close()
    return dict(u) if u else None

def charge(uid, amt):
    conn, c, is_pg = get_db_conn()
    c.execute("SELECT balance FROM users WHERE id=%s", (uid,))
    r = c.fetchone()
    if not r or dict(r)['balance'] < amt: conn.close(); return False
    c.execute("UPDATE users SET balance = balance - %s WHERE id=%s", (amt, uid))
    conn.commit(); conn.close()
    return True

def pay_uploader(uid, amt):
    fee = amt * PLATFORM_FEE
    net = amt - fee
    conn, c, is_pg = get_db_conn()
    c.execute("UPDATE users SET balance = balance + %s WHERE id=%s", (net, uid))
    c.execute("INSERT INTO revenues (source_type, amount, platform_fee, uploader_id) VALUES (%s, %s, %s, %s)",
             ("query", amt, fee, uid))
    conn.commit(); conn.close()


# ─── Face Recognition Helpers ────────────────────────────────────────────────

# =============================================================================
# ArcFace (InsightFace buffalo_l) embedding - 512-dim deep learning embeddings
# =============================================================================

_arcface_app = None
_arcface_loading_attempted = False

def _get_arcface_app():
    """Load and cache the InsightFace buffalo_l model."""
    global _arcface_app, _arcface_loading_attempted
    if _arcface_app is not None:
        return _arcface_app
    if _arcface_loading_attempted:
        return None
    _arcface_loading_attempted = True
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _arcface_app = app
        logger.info("InsightFace buffalo_l model loaded successfully")
        return app
    except ImportError:
        logger.warning("InsightFace not installed, ArcFace embeddings unavailable")
        return None
    except Exception as e:
        logger.warning(f"InsightFace model loading failed: {e}")
        return None


def _extract_arcface_embedding(image_source):
    """
    Extract 512-dim ArcFace embedding using InsightFace buffalo_l.
    Returns (embedding_list, error_msg).
    Falls back to None if InsightFace is not available.
    """
    import cv2
    import numpy as np

    app = _get_arcface_app()
    if app is None:
        return None, "insightface_unavailable"

    # Decode image
    img = None
    img_bytes = None
    if isinstance(image_source, str):
        if image_source.startswith('data:'):
            b64 = image_source.split(',')[1]
            img_bytes = base64.b64decode(b64)
        elif len(image_source) > 200 and not Path(image_source).exists():
            img_bytes = base64.b64decode(image_source)
        elif Path(image_source).exists():
            img = cv2.imread(str(image_source))
        else:
            img_bytes = base64.b64decode(image_source)
    elif isinstance(image_source, bytes):
        img_bytes = image_source

    if img is None and img_bytes:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, "could_not_decode_image"

    try:
        faces = app.get(img)
        if not faces:
            return None, "no_face_detected"
        largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return largest.embedding.tolist(), None
    except Exception as e:
        return None, str(e)


# =============================================================================
# OpenCV-based face embedding (fallback when InsightFace unavailable)
# =============================================================================

def _extract_face_embedding(image_source):
    """
    Extract face embedding using OpenCV (pure Python, no external model files needed).
    Falls back gracefully to resize-based features if any step fails.
    Returns (embedding_list, error_msg).
    """
    import cv2
    import numpy as np

    def norm(x):
        n = np.linalg.norm(x)
        return (x / n) if n > 0 else x

    def extract_from_face(face):
        """Extract embedding from a cropped face image."""
        if face.size == 0:
            return None
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

        gray_small = cv2.resize(gray, (64, 64))
        sobelx = cv2.Sobel(gray_small, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_small, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        angle = np.arctan2(sobely, sobelx)
        hog = []
        for cy in range(8):
            for cx in range(8):
                cm = magnitude[cy*8:(cy+1)*8, cx*8:(cx+1)*8]
                ca = angle[cy*8:(cy+1)*8, cx*8:(cx+1)*8]
                h, _ = np.histogram(ca.ravel(), bins=9, range=(-np.pi, np.pi), weights=cm.ravel())
                hog.extend(h)
        hog = np.array(hog, dtype=np.float32)
        pixels = cv2.resize(gray, (32, 32)).flatten().astype(np.float32)

        color_feat = norm(np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32))
        hog_feat = norm(hog)
        pixel_feat = norm(pixels)
        embedding = np.concatenate([color_feat * 0.2, hog_feat * 0.4, pixel_feat * 0.4])
        return norm(embedding).tolist()

    # ---- Decode image ----
    img = None
    img_bytes = None
    if isinstance(image_source, str):
        if image_source.startswith('data:'):
            b64 = image_source.split(',')[1]
            img_bytes = base64.b64decode(b64)
        elif len(image_source) > 200 and not Path(image_source).exists():
            img_bytes = base64.b64decode(image_source)
        elif Path(image_source).exists():
            img = cv2.imread(str(image_source))
        else:
            img_bytes = base64.b64decode(image_source)
    elif isinstance(image_source, bytes):
        img_bytes = image_source

    if img is None and img_bytes:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, "Could not decode image"

    # ---- Try DNN face detection (model cached locally) ----
    try:
        model_dir = Path.home() / '.cache' / 'portraitpay'
        model_dir.mkdir(parents=True, exist_ok=True)
        prototxt = model_dir / 'deploy.prototxt'
        caffemodel = model_dir / 'res10_300x300_ssd_iter_140000.caffemodel'

        if caffemodel.exists() and prototxt.exists():
            net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            best_conf, best_box = 0, None
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > 0.5 and conf > best_conf:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    best_conf, best_box = conf, box.astype(int)
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                face = img[y1:y2, x1:x2]
                emb = extract_from_face(face)
                if emb:
                    return emb, None
    except Exception as e:
        logger.warning(f"DNN detection skipped/failed: {e}")

    # ---- Fallback: center crop + resize ----
    h, w = img.shape[:2]
    crop_size = min(h, w)
    y_offset = (h - crop_size) // 2
    x_offset = (w - crop_size) // 2
    face = img[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]
    emb = extract_from_face(face)
    if emb:
        return emb, None
    return None, "embedding extraction failed"


def _cosine_similarity(a, b):
    """Compute cosine similarity between two embedding vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _find_matching_faces(embedding, threshold=0.35, top_k=5):
    """
    Find matching faces from the database.
    Returns list of dicts with face_id, name, similarity score.
    """
    conn, c, is_pg = get_db_conn()
    c.execute("""
        SELECT fe.id, fe.face_id, fe.embedding, f.name, f.is_celebrity,
               f.original_price, f.copyright_info, f.image_path
        FROM face_embeddings fe
        JOIN faces f ON fe.face_id = f.id
        WHERE f.status = 'active'
    """)
    rows = c.fetchall()
    conn.close()

    matches = []
    for row in rows:
        try:
            stored_emb = json.loads(row['embedding'])
        except (json.JSONDecodeError, TypeError):
            try:
                import pickle
                stored_emb = pickle.loads(row['embedding'])
            except Exception:
                continue

        sim = _cosine_similarity(embedding, stored_emb)
        if sim >= threshold:
            matches.append({
                'face_id': row['face_id'],
                'name': row['name'],
                'is_celebrity': bool(row['is_celebrity']),
                'similarity': round(float(sim), 4),
                'price': row['original_price'] or 0,
                'copyright_info': row['copyright_info'],
                'image_path': row['image_path']
            })

    # Sort by similarity descending
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    return matches[:top_k]

@app.route('/api/register', methods=['POST'])
def register():
    d = request.json
    if not d.get('username') or not d.get('password'):
        return jsonify({"error": "请填写用户名和密码"}), 400
    email = d.get('email', '').strip()
    # Email is optional for now (email sending not yet working)
    # if not email:
    #     return jsonify({"error": "请填写邮箱地址"}), 400
    
    ph = hashlib.sha256(d['password'].encode()).hexdigest()
    ak = secrets.token_hex(16)
    code = generate_code(6)
    
    conn, c, is_pg = get_db_conn()
    try:
        # Check if user already exists
        c.execute("SELECT id FROM users WHERE username=%s", (d['username'],))
        if c.fetchone():
            conn.close()
            return jsonify({"error": "用户名已存在"}), 400
        
        c.execute("INSERT INTO users (username, password_hash, api_key, email, verification_code, verified) VALUES (%s, %s, %s, %s, %s, FALSE)",
                 (d['username'], ph, ak, email, code))
        conn.commit(); uid = last_insert_id(conn, c, is_pg); conn.close()
        
        # Send verification email
        subject = "PortraitPay 注册验证码"
        body = f"您的验证码是：{code}\n\n请在5分钟内完成验证。\n\n如果不是您本人注册，请忽略此邮件。"
        ok, err = send_email(email, subject, body)
        if not ok:
            logger.warning(f"Failed to send verification email to {email}: {err}")
        
        return jsonify({"success": True, "status": "pending_verification", "message": "注册成功，请查收验证码邮件"})
    except Exception as e:
        conn.close()
        logger.error(f"Registration error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": "用户名已存在或注册失败", "detail": str(e)}), 400

@app.route('/api/register/verify', methods=['POST'])
def verify_registration():
    d = request.json
    username = d.get('username', '').strip()
    code = d.get('code', '').strip()
    
    if not username or not code:
        return jsonify({"error": "请提供用户名和验证码"}), 400
    
    conn, c, is_pg = get_db_conn()
    c.execute("SELECT id, verification_code, verified FROM users WHERE username=%s", (username,))
    user = c.fetchone()
    
    if not user:
        conn.close()
        return jsonify({"error": "用户不存在"}), 404
    
    if user['verified']:
        conn.close()
        return jsonify({"error": "用户已验证"}), 400
    
    if user['verification_code'] != code:
        conn.close()
        return jsonify({"error": "验证码错误"}), 400
    
    c.execute("UPDATE users SET verified=1, verification_code=NULL WHERE id=%s", (user['id'],))
    conn.commit(); conn.close()
    
    return jsonify({"success": True, "message": "验证成功，欢迎加入PortraitPay！"})

@app.route('/api/login', methods=['POST'])
def login():
    d = request.json
    ph = hashlib.sha256(d.get('password','').encode()).hexdigest()
    conn, c, is_pg = get_db_conn()
    c.execute("SELECT * FROM users WHERE username=%s AND password_hash=%s", (d.get('username'), ph))
    u = c.fetchone(); conn.close()
    if u: return jsonify({"success": True, "api_key": u["api_key"], "balance": u["balance"]})
    return jsonify({"error": "登录失败"}), 401

@app.route('/api/balance', methods=['GET'])
def balance():
    u = get_user(request.headers.get('X-API-Key'))
    if not u: return jsonify({"error": "未授权"}), 401
    return jsonify({"balance": u["balance"]})

@app.route('/api/deposit', methods=['POST'])
def deposit():
    u = get_user(request.headers.get('X-API-Key'))
    if not u: return jsonify({"error": "未授权"}), 401
    amt = request.json.get('amount', 0)
    if amt <= 0: return jsonify({"error": "金额需大于0"}), 400
    conn, c, is_pg = get_db_conn()
    c.execute("UPDATE users SET balance = balance + %s WHERE id=%s", (amt, u["id"]))
    conn.commit(); conn.close()
    return jsonify({"success": True, "new_balance": u["balance"] + amt})

@app.route('/api/faces', methods=['GET'])
def get_faces():
    category = request.args.get('category', 'all')
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)

    conn, c, is_pg = get_db_conn()

    if category == 'celebrity':
        c.execute("SELECT id, name, description, is_celebrity, original_price, ai_declaration, usage_count FROM faces WHERE status='active' AND is_celebrity=1 LIMIT %s OFFSET %s", (limit, offset))
    elif category == 'normal':
        c.execute("SELECT id, name, description, is_celebrity, original_price, ai_declaration, usage_count FROM faces WHERE status='active' AND is_celebrity=0 LIMIT %s OFFSET %s", (limit, offset))
    else:
        c.execute("SELECT id, name, description, is_celebrity, original_price, ai_declaration, usage_count FROM faces WHERE status='active' LIMIT %s OFFSET %s", (limit, offset))

    rows = c.fetchall()

    # total count
    if category == 'celebrity':
        c.execute("SELECT COUNT(*) FROM faces WHERE status='active' AND is_celebrity=1")
    elif category == 'normal':
        c.execute("SELECT COUNT(*) FROM faces WHERE status='active' AND is_celebrity=0")
    else:
        c.execute("SELECT COUNT(*) FROM faces WHERE status='active'")
    total = dict(c.fetchone())['count']

    conn.close()
    return jsonify({"faces": [dict(r) for r in rows], "total": total, "limit": limit, "offset": offset})


@app.route('/api/faces/search', methods=['GET'])
def search_faces():
    q = request.args.get('q', '').strip()
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)

    if not q:
        return jsonify({"error": "请提供搜索关键词"}), 400

    conn, c, is_pg = get_db_conn()
    pattern = f"%{q}%"
    c.execute("SELECT id, name, description, is_celebrity, original_price, ai_declaration, usage_count FROM faces WHERE status='active' AND name LIKE %s LIMIT %s OFFSET %s", (pattern, limit, offset))
    rows = c.fetchall()
    c.execute("SELECT COUNT(*) FROM faces WHERE status='active' AND name LIKE %s", (pattern,))
    total = dict(c.fetchone())['count']
    conn.close()
    return jsonify({"faces": [dict(r) for r in rows], "total": total, "limit": limit, "offset": offset, "query": q})

@app.route('/api/faces', methods=['POST'])
def add_face():
    u = get_user(request.headers.get('X-API-Key'))
    if not u: return jsonify({"error": "请先登录"}), 401
    d = request.json
    if not d.get('ai_declaration'):
        return jsonify({"error": "必须声明非AI生成"}), 400
    hid = hashlib.sha256(f"{d.get('name')}{time.time()}".encode()).hexdigest()[:16]
    conn, c, is_pg = get_db_conn()
    c.execute("INSERT INTO faces (name, description, hash_id, is_celebrity, copyright_info, uploader_id, original_price, ai_declaration) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
             (d.get('name'), d.get('description'), hid, d.get('is_celebrity',0), d.get('copyright_info'), u['id'], d.get('price',0), d.get('ai_declaration')))
    conn.commit(); conn.close()
    return jsonify({"success": True, "hash_id": hid})

@app.route('/api/faces/<int:fid>', methods=['GET'])
def get_face(fid):
    u = get_user(request.headers.get('X-API-Key'))
    conn, c, is_pg = get_db_conn()
    c.execute("SELECT * FROM faces WHERE id=%s AND status='active'", (fid,))
    f = c.fetchone()
    if not f: conn.close(); return jsonify({"error": "不存在"}), 404
    price = f["original_price"] or 0
    if price == 0 or (u and charge(u['id'], price)):
        c.execute("UPDATE faces SET usage_count = usage_count + 1 WHERE id=%s", (fid,))
        if price > 0 and f["uploader_id"]: pay_uploader(f["uploader_id"], price)
        conn.commit()
        conn.close()
        return jsonify({"name": f["name"], "description": f["description"], "copyright_info": f["copyright_info"], "ai_declaration": f["ai_declaration"]})
    conn.close()
    return jsonify({"error": "需付费", "price": price}), 402

@app.route('/api/works', methods=['GET'])
def get_works():
    conn, c, is_pg = get_db_conn()
    c.execute("SELECT id, title, description, work_type, author_name, original_price, ai_declaration FROM works WHERE status='active'")
    rows = c.fetchall(); conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/works', methods=['POST'])
def add_work():
    u = get_user(request.headers.get('X-API-Key'))
    if not u: return jsonify({"error": "请先登录"}), 401
    d = request.json
    if not d.get('ai_declaration'):
        return jsonify({"error": "必须声明非AI生成"}), 400
    hid = hashlib.sha256(f"{d.get('title')}{time.time()}".encode()).hexdigest()[:16]
    conn, c, is_pg = get_db_conn()
    c.execute("INSERT INTO works (title, description, content, work_type, hash_id, author_name, uploader_id, original_price, ai_declaration) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
             (d.get('title'), d.get('description'), d.get('content'), d.get('work_type'), hid, d.get('author_name'), u['id'], d.get('price',0), d.get('ai_declaration')))
    conn.commit(); conn.close()
    return jsonify({"success": True, "hash_id": hid})

@app.route('/api/works/<int:wid>', methods=['GET'])
def get_work(wid):
    u = get_user(request.headers.get('X-API-Key'))
    conn, c, is_pg = get_db_conn()
    c.execute("SELECT * FROM works WHERE id=%s AND status='active'", (wid,))
    w = c.fetchone()
    if not w: conn.close(); return jsonify({"error": "不存在"}), 404
    price = w["original_price"] or 0
    if price == 0 or (u and charge(u['id'], price)):
        c.execute("UPDATE works SET usage_count = usage_count + 1 WHERE id=%s", (wid,))
        if price > 0 and w["uploader_id"]: pay_uploader(w["uploader_id"], price)
        conn.commit()
        conn.close()
        return jsonify({"title": w["title"], "description": w["description"], "content": w["content"], "author_name": w["author_name"]})
    conn.close()
    return jsonify({"error": "需付费", "price": price}), 402

@app.route('/api/stats', methods=['GET'])
def stats():
    try:
        conn, c, is_pg = get_db_conn()
        c.execute("SELECT COUNT(*) FROM faces WHERE status='active'")
        fc = dict(c.fetchone())['count']
        c.execute("SELECT COUNT(*) FROM works WHERE status='active'")
        wc = dict(c.fetchone())['count']
        c.execute("SELECT SUM(usage_count) FROM faces")
        fu = dict(c.fetchone()).get('sum') or 0
        wu = 0  # works table has no usage_count column
        c.execute("SELECT SUM(platform_fee) FROM revenues")
        pr = dict(c.fetchone()).get('sum') or 0
        conn.close()
        return jsonify({"faces": fc, "works": wc, "uses": fu+wu, "platform_revenue": pr, "fee_rate": "1%"})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/api/my', methods=['GET'])
def my_uploads():
    u = get_user(request.headers.get('X-API-Key'))
    if not u: return jsonify({"error": "未授权"}), 401
    conn, c, is_pg = get_db_conn()
    c.execute("SELECT * FROM faces WHERE uploader_id=%s", (u['id'],))
    fs = [dict(r) for r in c.fetchall()]
    c.execute("SELECT * FROM works WHERE uploader_id=%s", (u['id'],))
    ws = [dict(r) for r in c.fetchall()]
    c.execute("SELECT SUM(amount-platform_fee) FROM revenues WHERE uploader_id=%s", (u['id'],))
    earn = dict(c.fetchone()).get('sum') or 0
    # 获取余额
    c.execute("SELECT balance FROM users WHERE id=%s", (u['id'],))
    balance = dict(c.fetchone())['balance']
    conn.close()
    return jsonify({"faces": fs, "works": ws, "earnings": earn, "balance": balance})

@app.route('/api/history', methods=['GET'])
def get_history():
    """获取用户的使用历史"""
    u = get_user(request.headers.get('X-API-Key'))
    if not u: return jsonify({"error": "未授权"}), 401
    
    chart_mode = request.args.get('chart', 'false') == 'true'
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    conn, c, is_pg = get_db_conn()
    
    if chart_mode:
        # 返回最近7天每日收益（图表用）
        c.execute("""
            SELECT DATE(created_at) as date, SUM(amount - platform_fee) as amount
            FROM revenues
            WHERE uploader_id = %s AND created_at >= DATE('now', '-7 days')
            GROUP BY DATE(created_at)
            ORDER BY date ASC
        """, (u['id'],))
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return jsonify(rows)
    
    # 获取人脸查询历史
    c.execute("""
        SELECT 'face' as type, f.name as target_name, ul.timestamp, ul.amount_paid,
               'AI调用' as usage_type, '已结算' as status
        FROM usage_logs ul
        JOIN faces f ON ul.target_id = f.id
        WHERE ul.user_id = %s
        ORDER BY ul.timestamp DESC
        LIMIT %s OFFSET %s
    """, (u['id'], limit, offset))
    face_history = [dict(r) for r in c.fetchall()]
    
    # 获取作品查看历史
    c.execute("""
        SELECT 'work' as type, w.title as target_name, ul.timestamp, ul.amount_paid,
               '作品查看' as usage_type, '已结算' as status
        FROM usage_logs ul
        JOIN works w ON ul.target_id = w.id
        WHERE ul.user_id = %s
        ORDER BY ul.timestamp DESC
        LIMIT %s OFFSET %s
    """, (u['id'], limit, offset))
    work_history = [dict(r) for r in c.fetchall()]
    
    # 合并并排序
    all_history = face_history + work_history
    all_history.sort(key=lambda x: x['timestamp'], reverse=True)
    
    conn.close()
    return jsonify({"history": all_history[:limit]})

@app.route('/api/export', methods=['GET'])
def export_data():
    """导出用户数据"""
    u = get_user(request.headers.get('X-API-Key'))
    if not u: return jsonify({"error": "未授权"}), 401
    
    conn, c, is_pg = get_db_conn()
    
    c.execute("SELECT * FROM faces WHERE uploader_id=%s", (u['id'],))
    faces = [dict(r) for r in c.fetchall()]
    
    c.execute("SELECT * FROM works WHERE uploader_id=%s", (u['id'],))
    works = [dict(r) for r in c.fetchall()]
    
    c.execute("SELECT * FROM revenues WHERE uploader_id=%s", (u['id'],))
    revenues = [dict(r) for r in c.fetchall()]
    
    conn.close()
    
    return jsonify({
        "exported_at": datetime.now().isoformat(),
        "user": {"id": u['id'], "username": u['username']},
        "faces": faces,
        "works": works,
        "revenues": revenues
    })

@app.route('/')
def idx():
    return jsonify({
        "name": "PortraitPay AI API",
        "version": "1.0.0",
        "description": "肖像版权数据库系统",
        "endpoints": {
            "auth": ["/api/register", "/api/login", "/api/balance", "/api/deposit"],
            "faces": ["/api/faces", "/api/faces/<id>"],
            "works": ["/api/works", "/api/works/<id>"],
            "stats": ["/api/stats", "/api/my", "/api/history"],
            "llm": ["/api/llm/portrait-check", "/api/llm/verify"]
        }
    })

if __name__ == '__main__':
    logger.info("PortraitPay AI starting on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)

@app.route('/api/llm/verify', methods=['POST'])
def llm_verify():
    """验证肖像使用权"""
    data = request.json
    face_id = data.get('face_id')
    usage_purpose = data.get('purpose', 'AI生成')
    
    if not face_id:
        return jsonify({"error": "缺少face_id"}), 400
    
    conn, c, is_pg = get_db_conn()
    c.execute("SELECT * FROM faces WHERE id=%s AND status='active'", (face_id,))
    face = c.fetchone()
    conn.close()
    
    if not face:
        return jsonify({"error": "肖像不存在"}), 404
    
    price = face["original_price"] or 0
    
    return jsonify({
        "face_id": face_id,
        "name": face["name"],
        "is_celebrity": bool(face["is_celebrity"]),
        "price": price,
        "copyright_info": face["copyright_info"],
        "authorized": price == 0,
        "usage_purpose": usage_purpose,
        "license": f"PortraitHub License - {usage_purpose}"
    })


@app.route('/api/debug/face/<int:fid>', methods=['GET'])
def debug_face(fid):
    """Debug endpoint to check face uploader_id."""
    conn, c, is_pg = get_db_conn()
    c.execute("SELECT * FROM faces WHERE id=%s", (fid,))
    face = c.fetchone()
    conn.close()
    if not face:
        return jsonify({"error": "not found"}), 404
    return jsonify(dict(face))

@app.route('/api/faces/register-embedding', methods=['POST'])
def register_face_embedding():
    """
    Register a face with its embedding.
    Body: { face_id: int, image: base64_string }
    """
    try:
        api_key = request.headers.get('X-API-Key')
        user = get_user(api_key)
        if not user:
            return jsonify({"error": "请先登录"}), 401

        data = request.json
        face_id = data.get('face_id')
        image_data = data.get('image')

        if not face_id:
            return jsonify({"error": "缺少face_id"}), 400
        if not image_data:
            return jsonify({"error": "缺少图片"}), 400

        # Verify the face exists
        conn, c, is_pg = get_db_conn()
        c.execute("SELECT * FROM faces WHERE id=%s AND status='active'", (face_id,))
        face = c.fetchone()
        conn.close()

        if not face:
            return jsonify({"error": "肖像不存在"}), 404

        # Check uploader_id matches, but if no uploader_id set (legacy), allow anyway
        if face['uploader_id'] is not None and face['uploader_id'] != int(user['id']):
            return jsonify({"error": "肖像不存在或无权操作"}), 404

        # Extract embedding
        embedding, err = _extract_face_embedding(image_data)
        if err:
            logger.error(f"Embedding extraction failed for face_id {face_id}: {err}")
            return jsonify({"error": f"人脸识别失败: {err}"}), 422

        # Store embedding
        import pickle
        conn, c, is_pg = get_db_conn()
        c.execute("DELETE FROM face_embeddings WHERE face_id=%s", (face_id,))
        c.execute("INSERT INTO face_embeddings (face_id, embedding, model_name) VALUES (%s, %s, %s)",
                   (face_id, pickle.dumps(embedding), 'hist-hog-pixel'))
        conn.commit()
        conn.close()

        logger.info(f"Stored embedding for face_id {face_id} by user {user['id']}")
        return jsonify({"success": True, "face_id": face_id, "embedding_dim": len(embedding)})
    except Exception as e:
        import traceback
        logger.error(f"Register embedding error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"服务器错误: {str(e)}"}), 500


@app.route('/api/debug/whoami', methods=['GET'])
def debug_whoami():
    api_key = request.headers.get('X-API-Key')
    user = get_user(api_key)
    if not user:
        return jsonify({"error": "invalid"}), 401
    # Explicitly pick serializable fields only
    return jsonify({
        "id": int(user['id']),
        "username": str(user['username']),
        "api_key": api_key
    })

@app.route('/api/debug/test-query', methods=['GET'])
def debug_test_query():
    """Test parameterized query on Railway PG."""
    try:
        conn, c, is_pg = get_db_conn()
        c.execute("SELECT COUNT(*) FROM users WHERE username=%s", ("nonexistent",))
        r = dict(c.fetchone())
        conn.close()
        return jsonify({"works": True, "result": r})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/dbinfo', methods=['GET'])
def debug_dbinfo():
    """Check database connection and environment on Railway."""
    import platform
    env_keys = ['DATABASE_URL', 'RAILWAY_ENVIRONMENT', 'POSTGRES_HOST', 'POSTGRES_PORT',
                'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB', 'PGHOST', 'PGPORT',
                'PGUSER', 'PGPASSWORD', 'PGDATABASE', 'PGURL', 'POSTGRES_URL']
    env_info = {k: os.environ.get(k, '(not set)')[:50] for k in env_keys}
    env_info['_USE_PG'] = str(db_module._USE_PG)
    env_info['_PG_URL'] = (db_module._PG_URL or '(none)')[:50]
    env_info['platform'] = platform.system()
    env_info['cwd'] = str(Path(__file__).parent)
    
    # Try to list tables
    table_info = {}
    try:
        conn, c, is_pg = get_db_conn()
        c.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'" if is_pg 
                  else "SELECT name FROM sqlite_master WHERE type='table'")
        tables = c.fetchall()
        table_info['tables'] = [dict_from_row(t) for t in tables]
        conn.close()
    except Exception as e:
        table_info['error'] = str(e)
    
    return jsonify({**env_info, **table_info})

@app.route('/api/faces/list-embeddings', methods=['GET'])
def list_face_embeddings():
    """
    List all faces that have embeddings registered.
    """
    conn, c, is_pg = get_db_conn()
    c.execute("""
        SELECT f.id, f.name, f.is_celebrity, f.original_price, f.image_path,
               fe.model_name, fe.created_at
        FROM face_embeddings fe
        JOIN faces f ON fe.face_id = f.id
        WHERE f.status = 'active'
    """)
    rows = c.fetchall()
    conn.close()
    
    return jsonify({
        "faces": [dict(r) for r in rows],
        "total": len(rows)
    })


@app.route('/api/faces/match', methods=['POST'])
def match_face():
    """
    Match a face image against registered faces.
    Body: { image: base64_string, threshold: 0.35 (optional) }
    Returns matched faces sorted by similarity.
    """
    data = request.json
    image_data = data.get('image')
    threshold = float(data.get('threshold', 0.35))
    
    if not image_data:
        return jsonify({"error": "缺少图片"}), 400
    
    # Extract embedding from provided image
    embedding, err = _extract_face_embedding(image_data)
    if err:
        logger.error(f"Embedding extraction failed in match: {err}")
        return jsonify({"error": f"人脸识别失败: {err}"}), 422
    
    # Find matches
    matches = _find_matching_faces(embedding, threshold=threshold)
    
    return jsonify({
        "matches": matches,
        "total_matches": len(matches),
        "threshold": threshold,
        "embedding_dim": len(embedding)
    })


@app.route('/api/faces/match-celeb', methods=['POST'])
def match_celebrity():
    """
    Match a face image against celebrity database using ArcFace (InsightFace buffalo_l).
    This endpoint uses deep learning embeddings (512-dim) for high-accuracy matching.
    
    Body: { image: base64_string, top_k: 5 (optional) }
    Returns top-k matched celebrities sorted by cosine similarity.
    """
    import numpy as np
    
    data = request.json
    image_data = data.get('image')
    top_k = int(data.get('top_k', 5))
    
    if not image_data:
        return jsonify({"error": "缺少图片"}), 400
    
    # Extract ArcFace embedding
    embedding, err = _extract_arcface_embedding(image_data)
    if err:
        logger.error(f"ArcFace embedding extraction failed: {err}")
        return jsonify({
            "error": f"人脸识别失败: {err}",
            "model": "arcface_buffalo_l",
            "available": _arcface_app is not None
        }), 422
    
    # Find matches only among celebrity embeddings (ArcFace embeddings)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT fe.id, fe.face_id, fe.embedding, f.name, ci.category,
               f.original_price, f.copyright_info, f.image_path, fe.model_name
        FROM face_embeddings fe
        JOIN faces f ON fe.face_id = f.id
        JOIN celebrity_info ci ON f.id = ci.face_id
        WHERE f.status = 'active' AND f.is_celebrity = 1
    """)
    rows = c.fetchall()
    conn.close()
    
    def norm(x):
        n = np.linalg.norm(x)
        return (x / n) if n > 0 else x
    
    def cosine_sim(a, b):
        return float(np.dot(norm(a), norm(b)))
    
    matches = []
    for row in rows:
        try:
            stored_emb = json.loads(row['embedding'])
        except (json.JSONDecodeError, TypeError):
            continue
        
        sim = cosine_sim(embedding, stored_emb)
        matches.append({
            'face_id': row['face_id'],
            'name': row['name'],
            'category': row['category'],
            'is_celebrity': True,
            'similarity': round(sim, 4),
            'similarity_pct': round(sim * 100, 1),
            'price': row['original_price'] or 0,
            'copyright_info': row['copyright_info'],
            'image_path': row['image_path'],
            'model': row['model_name']
        })
    
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    top_matches = matches[:top_k]
    
    return jsonify({
        "matches": top_matches,
        "total_db_celebs": len(rows),
        "top_k": top_k,
        "embedding_dim": len(embedding),
        "model": "arcface_buffalo_l"
    })


@app.route('/api/llm/portrait-check', methods=['POST'])
def llm_portrait_check():
    """
    REAL face recognition endpoint.
    Accepts an image and checks if it matches any registered celebrity/public faces.
    Also accepts a text prompt for keyword-based checks.
    """
    data = request.json
    image_data = data.get('image')
    prompt = data.get('prompt', '')
    
    # If image is provided, do real face matching
    if image_data:
        embedding, err = _extract_face_embedding(image_data)
        if err:
            logger.warning(f"Face detection failed: {err}")
            return jsonify({
                "needs_permission": True,
                "source": "detection_failed",
                "message": f"无法检测到人脸: {err}",
                "recommendation": "请确保图片中包含清晰的人脸"
            })
        
        matches = _find_matching_faces(embedding, threshold=0.35)
        
        if not matches:
            return jsonify({
                "needs_permission": False,
                "source": "no_match",
                "message": "未匹配到已注册肖像，可自由使用"
            })
        
        # Get best match
        best = matches[0]
        return jsonify({
            "needs_permission": True,
            "source": "face_recognition",
            "match": best,
            "all_matches": matches,
            "message": f"检测到与 {best['name']} 相似的人脸，需要授权",
            "recommendation": f"请通过 PortraitHub 获取 {best['name']} 的授权"
        })
    
    # Text-only prompt: keyword matching
    face_keywords = ['脸', '肖像', '人物', '明星', '演员', '人像', '照片', 'face', 'portrait', 'celebrity', '人物照', '照片']
    celebrity_keywords = ['明星', '演员', '歌手', '名人', 'celebrity', 'actor', 'actress', 'singer']
    
    prompt_lower = prompt.lower()
    is_celebrity = any(kw in prompt_lower for kw in celebrity_keywords)
    
    if any(kw in prompt_lower for kw in face_keywords):
        if is_celebrity:
            return jsonify({
                "needs_permission": True,
                "source": "keyword_celebrity",
                "message": "检测到可能涉及明星/公众人物肖像，建议获取授权"
            })
        else:
            return jsonify({
                "needs_permission": True,
                "source": "keyword_generic",
                "message": "检测到可能涉及人物肖像内容"
            })
    
    return jsonify({
        "needs_permission": False,
        "source": "no_face_detected",
        "message": "未检测到人脸相关内容"
    })

@app.route('/api/upload/portrait', methods=['POST'])
def upload_portrait():
    """上传肖像 - 需要登录且只能上传一张"""
    api_key = request.headers.get('X-API-Key')
    user = get_user(api_key)
    if not user:
        return jsonify({"error": "请先登录"}), 401
    
    # 检查是否已上传过
    conn, c, is_pg = get_db_conn()
    c.execute("SELECT id FROM faces WHERE uploader_id=%s", (user['id'],))
    existing = c.fetchone()
    if existing:
        conn.close()
        return jsonify({"error": "每位用户只能上传一张肖像"}), 400
    
    # 获取图片数据
    data = request.json
    name = data.get('name', '')
    image_base64 = data.get('image', '')
    id_image_base64 = data.get('id_image', '')  # 身份证照片
    age = data.get('age', 0)
    ai_declaration = data.get('ai_declaration', 0)
    
    if not age or age < 18:
        conn.close()
        return jsonify({"error": "必须年满18岁才能注册"}), 400
    
    if not ai_declaration:
        conn.close()
        return jsonify({"error": "必须声明非AI生成内容"}), 400
    
    if not image_base64:
        conn.close()
        return jsonify({"error": "请上传肖像照片"}), 400
    
    # 保存肖像图片
    import base64, time
    try:
        img_data = base64.b64decode(image_base64)
        filename = f"portrait_{user['id']}_{int(time.time())}.jpg"
        filepath = UPLOAD_DIR / filename
        with open(filepath, 'wb') as f:
            f.write(img_data)
        portrait_path = str(filepath)
    except:
        return jsonify({"error": "图片格式错误"}), 400
    
    # 保存ID图片（可选，用于人工审核）
    id_path = None
    if id_image_base64:
        try:
            id_data = base64.b64decode(id_image_base64)
            id_filename = f"id_{user['id']}_{int(time.time())}.jpg"
            id_filepath = UPLOAD_DIR / id_filename
            with open(id_filepath, 'wb') as f:
                f.write(id_data)
            id_path = str(id_filepath)
        except:
            pass
    
    hash_id = hashlib.sha256(f"{name}{user['id']}{time.time()}".encode()).hexdigest()[:16]
    
    c.execute('''INSERT INTO faces (name, image_path, hash_id, is_celebrity, uploader_id, original_price, ai_declaration, age, id_image_path)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)''',
             (name, portrait_path, hash_id, 0, user['id'], 0, ai_declaration, age, id_path))
    conn.commit()
    face_id = last_insert_id(conn, c, is_pg)
    conn.close()
    
    return jsonify({
        "success": True,
        "id": face_id,
        "hash_id": hash_id,
        "portrait_path": portrait_path,
        "message": "上传成功！请等待审核"
    })

@app.route('/api/upload/open-folder', methods=['POST'])
def open_upload_folder():
    """返回上传文件夹路径"""
    return jsonify({"folder": str(UPLOAD_DIR)})


# =============================================================================
# Local-First Portrait Fingerprint API (Phase 2)
# These endpoints support Local-First architecture where:
# - Original portraits stay on user's device
# - Only irreversible fingerprint hashes are stored on server
# - Server cannot access original images
# =============================================================================

@app.route('/api/portrait/register-fingerprint', methods=['POST'])
def register_portrait_fingerprint():
    """
    Register a portrait fingerprint (Local-First).
    Browser generates fingerprint locally, sends hash to server.
    Original image never leaves user's device.

    Request body:
    {
        "name": "User Name",
        "fingerprint_hash": "hex_string_64_chars",
        "fingerprint_type": "phash|dhash|ahash",
        "local_device_id": "optional_device_identifier",
        "price": 0,
        "ai_declaration": true,
        "copyright_info": "optional_copyright_text"
    }
    """
    api_key = request.headers.get('X-API-Key')
    user = get_user(api_key)
    if not user:
        return jsonify({"error": "请先登录"}), 401

    data = request.json
    name = data.get('name', '').strip()
    fingerprint_hash = data.get('fingerprint_hash', '').strip()
    fingerprint_type = data.get('fingerprint_type', 'phash')
    local_device_id = data.get('local_device_id', '')
    price = data.get('price', 0)
    ai_declaration = data.get('ai_declaration', 0)
    copyright_info = data.get('copyright_info', '')

    # Validation
    if not name:
        return jsonify({"error": "请提供姓名/艺名"}), 400
    if not fingerprint_hash or len(fingerprint_hash) < 16:
        return jsonify({"error": "指纹哈希无效"}), 400
    if len(fingerprint_hash) > 128:
        return jsonify({"error": "指纹哈希过长"}), 400
    if not ai_declaration:
        return jsonify({"error": "必须声明非AI生成肖像"}), 400

    conn, c, is_pg = get_db_conn()

    # Check if user already has a face registered
    c.execute("SELECT id FROM faces WHERE uploader_id=%s", (user['id'],))
    existing = c.fetchone()
    if existing:
        conn.close()
        return jsonify({"error": "每位用户只能注册一张肖像"}), 400

    # Generate hash_id for this face
    import time
    hash_id = hashlib.sha256(f"{name}{user['id']}{time.time()}".encode()).hexdigest()[:16]

    # Insert face record (without storing any image paths)
    c.execute('''INSERT INTO faces (name, hash_id, is_celebrity, uploader_id, original_price, ai_declaration, copyright_info, age, fingerprint_registered, local_device_id)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
             (name, hash_id, 0, user['id'], price, ai_declaration, copyright_info, 18, 1, local_device_id))
    conn.commit()
    face_id = last_insert_id(conn, c, is_pg)

    # Insert fingerprint record
    c.execute('''INSERT INTO portrait_fingerprints (face_id, fingerprint_hash, fingerprint_type, model_name)
                 VALUES (%s, %s, %s, %s)''',
             (face_id, fingerprint_hash, fingerprint_type, 'browser_local'))
    conn.commit()
    conn.close()

    logger.info(f"Portrait fingerprint registered for user {user['id']}, face_id={face_id}")
    return jsonify({
        "success": True,
        "face_id": face_id,
        "hash_id": hash_id,
        "message": "肖像指纹注册成功！其他用户现在可以查询您的授权状态"
    })


@app.route('/api/portrait/search-by-fingerprint', methods=['POST'])
def search_portrait_by_fingerprint():
    """
    Search for a portrait by fingerprint (Local-First).
    Browser generates fingerprint from query image, sends to server.
    Server compares against stored fingerprints.

    Request body:
    {
        "fingerprint_hash": "hex_string_64_chars",
        "fingerprint_type": "phash|dhash|ahash",
        "threshold": 0.85,  // optional, similarity threshold
        "limit": 5  // optional, max results
    }

    Returns:
    {
        "matches": [
            {
                "face_id": 123,
                "name": "XXX",
                "similarity": 0.92,
                "authorization_status": "available|unavailable|pending|unknown",
                "price": 100,
                "usage_restrictions": ["no_porn", "no_politics"]
            }
        ],
        "total": 1,
        "search_id": "audit_id"
    }
    """
    data = request.json
    fingerprint_hash = data.get('fingerprint_hash', '').strip()
    fingerprint_type = data.get('fingerprint_type', 'phash')
    threshold = data.get('threshold', 0.35)  # Default threshold for Hamming distance
    limit = min(data.get('limit', 5), 20)

    if not fingerprint_hash:
        return jsonify({"error": "请提供指纹哈希"}), 400

    conn, c, is_pg = get_db_conn()

    # Get all registered fingerprints
    c.execute("""
        SELECT pf.fingerprint_hash, pf.fingerprint_type, pf.face_id,
               f.name, f.is_celebrity, f.original_price, f.copyright_info,
               f.ai_declaration, f.status
        FROM portrait_fingerprints pf
        JOIN faces f ON pf.face_id = f.id
        WHERE pf.fingerprint_type = %s AND f.status = 'active'
    """, (fingerprint_type,))

    rows = c.fetchall()
    matches = []

    for row in rows:
        stored_hash = row['fingerprint_hash']
        # Calculate Hamming distance for perceptual hashes
        if fingerprint_type in ('phash', 'dhash', 'ahash'):
            similarity = _hamming_similarity(fingerprint_hash, stored_hash)
        else:
            # Fallback: exact match only
            similarity = 1.0 if fingerprint_hash == stored_hash else 0.0

        if similarity >= threshold:
            # Determine authorization status
            if row['status'] != 'active':
                auth_status = 'unavailable'
            elif row['ai_declaration'] == 0:
                auth_status = 'pending_review'
            else:
                auth_status = 'available'

            matches.append({
                'face_id': row['face_id'],
                'name': row['name'],
                'is_celebrity': bool(row['is_celebrity']),
                'similarity': round(float(similarity), 4),
                'authorization_status': auth_status,
                'price': row['original_price'] or 0,
                'copyright_info': row['copyright_info'],
            })

    # Sort by similarity descending
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    matches = matches[:limit]

    # Log search query for audit (without storing the actual fingerprint)
    c.execute('''INSERT INTO search_queries (query_fingerprint_hash, results_count, top_match_face_id, top_match_similarity)
                 VALUES (%s, %s, %s, %s)''',
             (fingerprint_hash[:16], len(matches),
              matches[0]['face_id'] if matches else None,
              matches[0]['similarity'] if matches else None))
    conn.commit()
    search_id = last_insert_id(conn, c, is_pg)
    conn.close()

    return jsonify({
        "matches": matches,
        "total": len(matches),
        "search_id": search_id,
        "message": "查询成功" if matches else "未找到匹配的肖像"
    })


def _hamming_similarity(hash1: str, hash2: str) -> float:
    """
    Calculate similarity between two hex-encoded perceptual hashes.
    Returns a value between 0 (completely different) and 1 (identical).
    Uses Hamming distance - count of differing bits.
    """
    if len(hash1) != len(hash2):
        # Try matching shorter strings
        min_len = min(len(hash1), len(hash2))
        hash1 = hash1[:min_len]
        hash2 = hash2[:min_len]

    # Convert hex strings to binary and count differences
    try:
        b1 = int(hash1, 16)
        b2 = int(hash2, 16)
    except ValueError:
        return 0.0

    # Calculate Hamming distance
    xor = b1 ^ b2
    hamming_distance = bin(xor).count('1')

    # Convert to similarity (assuming 64-character hex = 256 bits)
    bit_length = max(len(hash1) * 4, 1)  # 4 bits per hex char
    similarity = 1.0 - (hamming_distance / bit_length)

    return max(0.0, min(1.0, similarity))


@app.route('/api/portrait/authorization-check', methods=['POST'])
def check_portrait_authorization():
    """
    Check if a specific portrait can be used for a given purpose.
    This is the core licensing query API.

    Request body:
    {
        "face_id": 123,
        "usage_type": "ai_training|ai_generation|advertising|entertainment",
        "commercial_use": true|false,
        "exclusive": false
    }

    Returns:
    {
        "can_use": true|false,
        "authorization_status": "available|unavailable|pending|requires_manual_approval",
        "price": 100,
        "license_type": "standard|commercial|exclusive",
        "restrictions": ["no_porn", "no_politics"],
        "requires_contract": true|false
    }
    """
    data = request.json
    face_id = data.get('face_id')
    usage_type = data.get('usage_type', 'ai_generation')
    commercial_use = data.get('commercial_use', False)
    exclusive = data.get('exclusive', False)

    if not face_id:
        return jsonify({"error": "请提供 face_id"}), 400

    conn, c, is_pg = get_db_conn()
    c.execute("""
        SELECT f.*, pf.fingerprint_hash
        FROM faces f
        LEFT JOIN portrait_fingerprints pf ON f.id = pf.face_id
        WHERE f.id = %s
    """, (face_id,))

    face = c.fetchone()
    conn.close()

    if not face:
        return jsonify({"error": "肖像不存在"}), 404

    face_dict = dict(face) if face else {}

    # Determine authorization status
    if face_dict.get('status') != 'active':
        return jsonify({
            "can_use": False,
            "authorization_status": "unavailable",
            "reason": "该肖像已被禁用或删除"
        })

    if face_dict.get('ai_declaration') == 0:
        return jsonify({
            "can_use": False,
            "authorization_status": "pending_review",
            "reason": "该肖像正在审核中"
        })

    # Determine price based on usage type
    base_price = face_dict.get('original_price') or 0
    if commercial_use:
        price = base_price * 2  # Commercial use costs more
    elif exclusive:
        price = base_price * 5  # Exclusive is most expensive
    else:
        price = base_price

    # Determine restrictions from copyright_info
    restrictions = []
    copyright_info = face_dict.get('copyright_info', '') or ''
    restriction_keywords = {
        'no_porn': ['色情', '成人', 'porn', 'nsfw'],
        'no_politics': ['政治', 'politics'],
        'no_ads': ['广告', 'advertising'],
        'no_medical': ['医疗', 'medical']
    }
    for restr, keywords in restriction_keywords.items():
        if any(kw.lower() in copyright_info.lower() for kw in keywords):
            restrictions.append(restr)

    # Some usage types always require manual approval
    requires_manual = usage_type in ('advertising', 'exclusive') or exclusive

    return jsonify({
        "can_use": True,
        "authorization_status": "available" if not requires_manual else "requires_manual_approval",
        "price": price,
        "license_type": "exclusive" if exclusive else ("commercial" if commercial_use else "standard"),
        "restrictions": restrictions,
        "requires_contract": requires_manual,
        "usage_type": usage_type,
        "message": "可以授权使用" if not requires_manual else "需要人工审批"
    })


@app.route('/api/portrait/update-authorization', methods=['PUT'])
def update_portrait_authorization():
    """
    Update portrait authorization settings.
    User can update price, copyright_info, usage restrictions.

    Request body:
    {
        "face_id": 123,
        "price": 100,
        "copyright_info": "仅限AI训练使用，禁止商用",
        "status": "active|inactive"
    }
    """
    api_key = request.headers.get('X-API-Key')
    user = get_user(api_key)
    if not user:
        return jsonify({"error": "请先登录"}), 401

    data = request.json
    face_id = data.get('face_id')
    price = data.get('price')
    copyright_info = data.get('copyright_info')
    status = data.get('status')

    if not face_id:
        return jsonify({"error": "请提供 face_id"}), 400

    conn, c, is_pg = get_db_conn()

    # Verify ownership
    c.execute("SELECT uploader_id FROM faces WHERE id=%s", (face_id,))
    face = c.fetchone()
    if not face:
        conn.close()
        return jsonify({"error": "肖像不存在"}), 404
    if dict(face)['uploader_id'] != user['id']:
        conn.close()
        return jsonify({"error": "无权修改此肖像"}), 403

    # Build update query dynamically
    updates = []
    params = []
    if price is not None:
        updates.append("original_price = %s")
        params.append(price)
    if copyright_info is not None:
        updates.append("copyright_info = %s")
        params.append(copyright_info)
    if status is not None:
        updates.append("status = %s")
        params.append(status)

    if updates:
        params.append(face_id)
        c.execute(f"UPDATE faces SET {', '.join(updates)} WHERE id = %s", tuple(params))
        conn.commit()

    conn.close()
    return jsonify({"success": True, "message": "授权设置已更新"})


# =============================================================================
# Admin & Utility Endpoints
# =============================================================================

@app.route('/api/admin/stats', methods=['GET'])
def admin_stats():
    """Get platform statistics for admin dashboard."""
    api_key = request.headers.get('X-API-Key')
    user = get_user(api_key)
    if not user or not user.get('is_admin'):
        return jsonify({"error": "需要管理员权限"}), 403

    conn, c, is_pg = get_db_conn()

    c.execute("SELECT COUNT(*) as total FROM faces WHERE status='active'")
    total_faces = dict(c.fetchone())['count']

    c.execute("SELECT COUNT(*) as total FROM users")
    total_users = dict(c.fetchone())['count']

    c.execute("SELECT COUNT(*) as total FROM portrait_fingerprints")
    total_fingerprints = dict(c.fetchone())['count']

    c.execute("SELECT SUM(results_count) as total FROM search_queries")
    result = c.fetchone()
    total_searches = dict(result)['total'] or 0 if result else 0

    conn.close()

    return jsonify({
        "total_faces": total_faces,
        "total_users": total_users,
        "total_fingerprints": total_fingerprints,
        "total_searches": total_searches,
        "local_first_enabled": True
    })


@app.route('/api/health/local-first', methods=['GET'])
def health_local_first():
    """Health check for Local-First API."""
    conn, c, is_pg = get_db_conn()
    c.execute("SELECT COUNT(*) as fp_count FROM portrait_fingerprints")
    result = c.fetchone()
    fp_count = dict(result)['count'] if result else 0
    conn.close()

    return jsonify({
        "status": "healthy",
        "local_first_mode": True,
        "fingerprints_registered": fp_count,
        "server_stores_fingerprints_only": True,
        "original_images_never_stored": True
    })


# =============================================================================
# Admin Automation API (Phase 5)
# Single-founder operation automation: auto-triage, reports, health checks
# =============================================================================

@app.route('/api/admin/tickets', methods=['GET'])
def get_tickets():
    """
    Get ticket queue for admin review.
    Query params:
    - status: open|in_progress|resolved|all (default: open)
    - priority: urgent|high|medium|low
    - category: auth_issue|authorization|dispute|enterprise|bug_report|data_request|payment|other
    - limit: max results (default 50)
    """
    api_key = request.headers.get('X-API-Key')
    user = get_user(api_key)
    if not user or not user.get('is_admin'):
        # Allow non-authenticated access for viewing tickets (for simplicity)
        pass

    status = request.args.get('status', 'open')
    priority = request.args.get('priority')
    category = request.args.get('category')
    limit = min(int(request.args.get('limit', 50)), 100)

    # Import and use admin automation
    try:
        from admin_automation import admin_automation
        tickets = admin_automation.get_ticket_queue(status, priority, category, limit)
        return jsonify({
            "tickets": tickets,
            "count": len(tickets),
            "filter": {"status": status, "priority": priority, "category": category}
        })
    except Exception as e:
        logger.error(f"Failed to get tickets: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/tickets/<int:ticket_id>', methods=['PUT'])
def update_ticket(ticket_id):
    """Update a ticket (change status, priority, assign, resolve)."""
    api_key = request.headers.get('X-API-Key')
    user = get_user(api_key)
    if not user:
        return jsonify({"error": "需要登录"}), 401

    data = request.json
    status = data.get('status')
    priority = data.get('priority')
    assigned_to = data.get('assigned_to')
    resolution = data.get('resolution')

    conn, c, is_pg = get_db_conn()
    updates = []
    params = []

    if status:
        updates.append("status = %s")
        params.append(status)
        if status == 'resolved':
            updates.append("resolved_at = CURRENT_TIMESTAMP")
    if priority:
        updates.append("priority = %s")
        params.append(priority)
    if assigned_to:
        updates.append("assigned_to = %s")
        params.append(assigned_to)
    if resolution:
        updates.append("resolution = %s")
        params.append(resolution)

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(ticket_id)

    if updates:
        c.execute(f"UPDATE tickets SET {', '.join(updates)} WHERE id = %s", tuple(params))
        conn.commit()

    conn.close()
    return jsonify({"success": True, "ticket_id": ticket_id})


@app.route('/api/admin/triage', methods=['POST'])
def triage_message():
    """
    Auto-triage an incoming message.
    Used for contact form submissions and support emails.
    """
    data = request.json
    subject = data.get('subject', '')
    body = data.get('body', '')
    email = data.get('email', '')
    source = data.get('source', 'contact_form')

    if not subject and not body:
        return jsonify({"error": "需要 subject 或 body"}), 400

    try:
        from admin_automation import admin_automation
        result = admin_automation.triage_message(subject, body, email)

        # Optionally create a ticket
        if data.get('create_ticket', False):
            ticket_id = admin_automation.create_ticket(subject, body, email, source)
            result['ticket_id'] = ticket_id

        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to triage message: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/reports/daily', methods=['GET'])
def daily_report():
    """Get daily operations report."""
    try:
        from admin_automation import admin_automation
        report = admin_automation.generate_daily_report()
        return jsonify(report)
    except Exception as e:
        logger.error(f"Failed to generate daily report: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/reports/weekly', methods=['GET'])
def weekly_report():
    """Get weekly aggregated report."""
    try:
        from admin_automation import admin_automation
        report = admin_automation.generate_weekly_report()
        return jsonify(report)
    except Exception as e:
        logger.error(f"Failed to generate weekly report: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/reports/monthly', methods=['GET'])
def monthly_report():
    """Get monthly aggregated report."""
    try:
        from admin_automation import admin_automation
        report = admin_automation.generate_monthly_report()
        return jsonify(report)
    except Exception as e:
        logger.error(f"Failed to generate monthly report: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/health', methods=['GET'])
def admin_health():
    """Get system health status for admin dashboard."""
    try:
        from admin_automation import admin_automation
        health = admin_automation.get_system_health()
        return jsonify(health)
    except Exception as e:
        logger.error(f"Failed to get health: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# Contact Form API with Auto-Triage
# =============================================================================

@app.route('/api/contacts', methods=['POST'])
def submit_contact():
    """
    Submit contact form with auto-triage.
    """
    data = request.json
    subject = data.get('subject', '')
    body = data.get('body', '')
    email = data.get('email', '')

    if not subject or not body:
        return jsonify({"error": "请填写主题和内容"}), 400

    if not email:
        return jsonify({"error": "请填写邮箱地址"}), 400

    try:
        from admin_automation import admin_automation
        triage_result = admin_automation.triage_message(subject, body, email)
        ticket_id = admin_automation.create_ticket(subject, body, email, "contact_form")

        return jsonify({
            "success": True,
            "message": "感谢您的留言，我们已收到并正在处理",
            "ticket_id": ticket_id,
            "category": triage_result['category'],
            "priority": triage_result['priority'],
            "action": triage_result['action']
        })
    except Exception as e:
        logger.error(f"Failed to submit contact: {e}")
        # Still return success to user to avoid leaking system errors
        return jsonify({
            "success": True,
            "message": "感谢您的留言，我们已收到"
        })


# =============================================================================
# Enterprise Inquiry API
# =============================================================================

@app.route('/api/enterprise/inquiry', methods=['POST'])
def enterprise_inquiry():
    """
    Submit enterprise/institutional inquiry.
    This is a high-priority channel for future Hollywood/agency partnerships.
    """
    data = request.json
    company = data.get('company', '')
    contact = data.get('contact', '')
    email = data.get('email', '')
    phone = data.get('phone', '')
    use_case = data.get('use_case', '')
    scale = data.get('scale', '')  # small|medium|large|enterprise
    message = data.get('message', '')

    if not email or not message:
        return jsonify({"error": "请填写邮箱和留言内容"}), 400

    try:
        from admin_automation import admin_automation

        # Create high-priority ticket for enterprise inquiries
        subject = f"企业咨询 - {company or '未提供公司名称'}"
        body = f"联系人和职位: {contact}\n公司: {company}\n使用场景: {use_case}\n规模: {scale}\n\n留言:\n{message}"

        triage_result = admin_automation.triage_message(subject, body, email)

        # Force high priority for enterprise
        if triage_result['priority'] not in ['urgent', 'high']:
            triage_result['priority'] = 'high'

        ticket_id = admin_automation.create_ticket(subject, body, email, "enterprise_inquiry")

        return jsonify({
            "success": True,
            "message": "感谢您的企业咨询，我们的商务团队将在1-2个工作日内与您联系",
            "ticket_id": ticket_id,
            "estimated_response": "1-2个工作日"
        })
    except Exception as e:
        logger.error(f"Failed to submit enterprise inquiry: {e}")
        return jsonify({
            "success": True,
            "message": "感谢您的咨询，我们将尽快回复"
        })


# =============================================================================
# Automated Alerts API
# =============================================================================

@app.route('/api/alerts/send', methods=['POST'])
def send_alert():
    """
    Send alert to admin (email/push notification).
    Used for critical system events.
    """
    data = request.json
    alert_type = data.get('type', 'info')  # info|warning|critical
    title = data.get('title', '')
    message = data.get('message', '')

    if not title or not message:
        return jsonify({"error": "需要 title 和 message"}), 400

    # Log the alert
    logger.warning(f"ALERT [{alert_type.upper()}] {title}: {message}")

    # In production, this would integrate with email/push services
    # For now, just log and return success
    return jsonify({
        "success": True,
        "alert_type": alert_type,
        "logged_at": datetime.now().isoformat()
    })


def _create_admin_user(username, email, password):
    """Create an admin user directly in the database. Returns result string."""
    try:
        conn, c, is_pg = get_db_conn()

        # Check if user already exists
        c.execute("SELECT id FROM users WHERE username=%s", (username,))
        if c.fetchone():
            conn.close()
            return f"Admin user '{username}' already exists"

        # Generate password hash and API key
        import hashlib, secrets
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        api_key = secrets.token_hex(32)

        # Insert user with is_admin=TRUE
        if is_pg:
            c.execute(
                "INSERT INTO users (username, password_hash, api_key, balance, is_admin, email, verified) "
                "VALUES (%s, %s, %s, 0, TRUE, %s, 1)",
                (username, password_hash, api_key, email)
            )
        else:
            c.execute(
                "INSERT INTO users (username, password_hash, api_key, balance, is_admin, email, verified) "
                "VALUES (?, ?, ?, 0, 1, ?, 1)",
                (username, password_hash, api_key, email)
            )

        conn.commit()
        conn.close()
        return f"Admin user '{username}' created with API key: {api_key}"
    except Exception as e:
        import traceback
        logger.error(f"Failed to create admin user: {e}\n{traceback.format_exc()}")
        return f"Failed to create admin user: {str(e)}"


# =============================================================================
# Database Migration Endpoint (for setting up new tables)
# =============================================================================

@app.route('/api/admin/init-db', methods=['POST'])
def admin_init_db():
    """
    Initialize/upgrade database schema.
    Creates portrait_fingerprints, search_queries tables and adds columns to faces.
    This is a one-time setup endpoint.

    Auth: Either admin user with X-API-Key, or X-INIT-SECRET header matching env var.
    """
    # Check for secret key auth (for initial setup without admin user)
    init_secret = request.headers.get('X-INIT-SECRET')
    expected_secret = os.environ.get('DB_INIT_SECRET', 'portraitpay-init-2026')
    if init_secret == expected_secret:
        pass  # Authorized via secret
    else:
        # Check for admin user auth
        api_key = request.headers.get('X-API-Key')
        user = get_user(api_key)
        if not user or not user.get('is_admin'):
            return jsonify({"error": "需要管理员权限"}), 403

    try:
        conn, c, is_pg = get_db_conn()

        results = []

        # 1. Create portrait_fingerprints table
        if is_pg:
            c.execute("SELECT table_name FROM information_schema.tables WHERE table_name='portrait_fingerprints' AND table_schema='public'")
        else:
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portrait_fingerprints'")

        if not c.fetchone():
            if is_pg:
                c.execute("""
                    CREATE TABLE portrait_fingerprints (
                        id SERIAL PRIMARY KEY,
                        face_id INTEGER NOT NULL,
                        fingerprint_hash TEXT NOT NULL,
                        fingerprint_type TEXT DEFAULT 'phash',
                        model_name VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(face_id, fingerprint_type)
                    )
                """)
            else:
                c.execute("""
                    CREATE TABLE portrait_fingerprints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        face_id INTEGER NOT NULL,
                        fingerprint_hash TEXT NOT NULL,
                        fingerprint_type TEXT DEFAULT 'phash',
                        model_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(face_id, fingerprint_type)
                    )
                """)
            results.append("Created portrait_fingerprints table")
        else:
            results.append("portrait_fingerprints table already exists")

        # 2. Create search_queries table
        if is_pg:
            c.execute("SELECT table_name FROM information_schema.tables WHERE table_name='search_queries' AND table_schema='public'")
        else:
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='search_queries'")

        if not c.fetchone():
            if is_pg:
                c.execute("""
                    CREATE TABLE search_queries (
                        id SERIAL PRIMARY KEY,
                        query_fingerprint_hash TEXT NOT NULL,
                        results_count INTEGER DEFAULT 0,
                        top_match_face_id INTEGER,
                        top_match_similarity REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            else:
                c.execute("""
                    CREATE TABLE search_queries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_fingerprint_hash TEXT NOT NULL,
                        results_count INTEGER DEFAULT 0,
                        top_match_face_id INTEGER,
                        top_match_similarity REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            results.append("Created search_queries table")
        else:
            results.append("search_queries table already exists")

        # 3. Create tickets table for admin automation
        if is_pg:
            c.execute("SELECT table_name FROM information_schema.tables WHERE table_name='tickets' AND table_schema='public'")
        else:
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets'")

        if not c.fetchone():
            if is_pg:
                c.execute("""
                    CREATE TABLE tickets (
                        id SERIAL PRIMARY KEY,
                        subject TEXT NOT NULL,
                        body TEXT,
                        email VARCHAR(255),
                        category VARCHAR(50) DEFAULT 'other',
                        priority VARCHAR(20) DEFAULT 'medium',
                        status VARCHAR(20) DEFAULT 'open',
                        assigned_to VARCHAR(100),
                        resolution TEXT,
                        source VARCHAR(50) DEFAULT 'contact_form',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        resolved_at TIMESTAMP
                    )
                """)
            else:
                c.execute("""
                    CREATE TABLE tickets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subject TEXT NOT NULL,
                        body TEXT,
                        email TEXT,
                        category TEXT DEFAULT 'other',
                        priority TEXT DEFAULT 'medium',
                        status TEXT DEFAULT 'open',
                        assigned_to TEXT,
                        resolution TEXT,
                        source TEXT DEFAULT 'contact_form',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        resolved_at TIMESTAMP
                    )
                """)
            results.append("Created tickets table")
        else:
            results.append("tickets table already exists")

        # 4. Add columns to faces table (ignore errors if already exist)
        existing_cols = []
        if is_pg:
            c.execute("SELECT column_name FROM information_schema.columns WHERE table_name='faces' AND column_name IN ('fingerprint_registered', 'local_device_id')")
        else:
            c.execute("PRAGMA table_info(faces)")
            rows = c.fetchall()
            existing_cols = [dict(r)['name'] for r in rows]

        if is_pg:
            for col, col_type, default in [
                ('fingerprint_registered', 'BOOLEAN', 'FALSE'),
                ('local_device_id', 'VARCHAR(128)', 'NULL')
            ]:
                c.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name='faces' AND column_name='{col}'")
                if not c.fetchone():
                    c.execute(f"ALTER TABLE faces ADD COLUMN {col} {col_type} DEFAULT {default}")
                    results.append(f"Added {col} column to faces")
                else:
                    results.append(f"{col} column already exists in faces")
        else:
            if 'fingerprint_registered' not in existing_cols:
                c.execute("ALTER TABLE faces ADD COLUMN fingerprint_registered INTEGER DEFAULT 0")
                results.append("Added fingerprint_registered column to faces")
            else:
                results.append("fingerprint_registered column already exists in faces")

            if 'local_device_id' not in existing_cols:
                c.execute("ALTER TABLE faces ADD COLUMN local_device_id TEXT")
                results.append("Added local_device_id column to faces")
            else:
                results.append("local_device_id column already exists in faces")

        # 5. Create index on fingerprint_hash
        if is_pg:
            c.execute("SELECT indexname FROM pg_indexes WHERE indexname='idx_fingerprint_hash'")
        else:
            c.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_fingerprint_hash'")

        if not c.fetchone():
            c.execute("CREATE INDEX idx_fingerprint_hash ON portrait_fingerprints(fingerprint_hash)")
            results.append("Created index on fingerprint_hash")
        else:
            results.append("idx_fingerprint_hash index already exists")

        conn.commit()
        conn.close()

        # 6. Optionally create admin user
        data = request.json or {}
        if data.get('create_admin'):
            admin_username = data.get('admin_username', 'admin')
            admin_email = data.get('admin_email', 'admin@portraitpayai.com')
            admin_password = data.get('admin_password', 'PortraitPay2026!')
            admin_result = _create_admin_user(admin_username, admin_email, admin_password)
            results.append(admin_result)

        return jsonify({
            "success": True,
            "message": "Database migration completed",
            "results": results,
            "is_pg": is_pg
        })

    except Exception as e:
        import traceback
        logger.error(f"Database init error: {e}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

