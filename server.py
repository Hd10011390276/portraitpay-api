#!/usr/bin/env python3
"""PortraitPay AI Backend - 肖像版权数据库系统"""

import os, json, sqlite3, hashlib, time, secrets, logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "portraitpay.db"
UPLOAD_DIR = DATA_DIR / "uploads"
PLATFORM_FEE = 0.01

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
CORS(app)

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
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        api_key TEXT UNIQUE,
        balance REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_admin INTEGER DEFAULT 0
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        image_path TEXT,
        hash_id TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_celebrity INTEGER DEFAULT 0,
        copyright_info TEXT,
        usage_count INTEGER DEFAULT 0,
        revenue REAL DEFAULT 0.0,
        uploader_id INTEGER,
        original_price REAL DEFAULT 0.0,
        ai_declaration INTEGER DEFAULT 0,
        status TEXT DEFAULT 'active'
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS works (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        content TEXT,
        work_type TEXT,
        hash_id TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        author_name TEXT,
        usage_count INTEGER DEFAULT 0,
        revenue REAL DEFAULT 0.0,
        uploader_id INTEGER,
        original_price REAL DEFAULT 0.0,
        ai_declaration INTEGER DEFAULT 0,
        status TEXT DEFAULT 'active'
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS usage_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_type TEXT,
        target_id INTEGER,
        action TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        user_id INTEGER,
        amount_paid REAL DEFAULT 0.0
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS revenues (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_type TEXT,
        source_id INTEGER,
        amount REAL,
        platform_fee REAL,
        uploader_id INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute("SELECT COUNT(*) FROM users WHERE is_admin=1")
    if c.fetchone()[0] == 0:
        h = hashlib.sha256("admin123".encode()).hexdigest()
        c.execute("INSERT INTO users (username, password_hash, is_admin, balance, api_key) VALUES (?, ?, 1, 1000.0, ?)",
                 ("admin", h, secrets.token_hex(16)))
    conn.commit()
    conn.close()

init_db()

def get_user(api_key):
    if not api_key: return None
    conn = sqlite3.connect(str(DB_PATH)); conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE api_key=?", (api_key,))
    u = c.fetchone(); conn.close()
    return dict(u) if u else None

def charge(uid, amt):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT balance FROM users WHERE id=?", (uid,))
    r = c.fetchone()
    if not r or r[0] < amt: conn.close(); return False
    c.execute("UPDATE users SET balance = balance - ? WHERE id=?", (amt, uid))
    conn.commit(); conn.close()
    return True

def pay_uploader(uid, amt):
    fee = amt * PLATFORM_FEE
    net = amt - fee
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("UPDATE users SET balance = balance + ? WHERE id=?", (net, uid))
    c.execute("INSERT INTO revenues (source_type, amount, platform_fee, uploader_id) VALUES (?, ?, ?, ?)",
             ("query", amt, fee, uid))
    conn.commit(); conn.close()

@app.route('/api/register', methods=['POST'])
def register():
    d = request.json
    if not d.get('username') or not d.get('password'):
        return jsonify({"error": "请填写用户名和密码"}), 400
    ph = hashlib.sha256(d['password'].encode()).hexdigest()
    ak = secrets.token_hex(16)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, api_key) VALUES (?, ?, ?)",
                 (d['username'], ph, ak))
        conn.commit(); uid = c.lastrowid; conn.close()
        return jsonify({"success": True, "api_key": ak, "message": "注册成功"})
    except: conn.close(); return jsonify({"error": "用户名已存在"}), 400

@app.route('/api/login', methods=['POST'])
def login():
    d = request.json
    ph = hashlib.sha256(d.get('password','').encode()).hexdigest()
    conn = sqlite3.connect(str(DB_PATH)); conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password_hash=?", (d.get('username'), ph))
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
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("UPDATE users SET balance = balance + ? WHERE id=?", (amt, u["id"]))
    conn.commit(); conn.close()
    return jsonify({"success": True, "new_balance": u["balance"] + amt})

@app.route('/api/faces', methods=['GET'])
def get_faces():
    conn = sqlite3.connect(str(DB_PATH)); conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id, name, description, is_celebrity, original_price, ai_declaration, usage_count FROM faces WHERE status='active'")
    rows = c.fetchall(); conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/faces', methods=['POST'])
def add_face():
    u = get_user(request.headers.get('X-API-Key'))
    if not u: return jsonify({"error": "请先登录"}), 401
    d = request.json
    if not d.get('ai_declaration'):
        return jsonify({"error": "必须声明非AI生成"}), 400
    hid = hashlib.sha256(f"{d.get('name')}{time.time()}".encode()).hexdigest()[:16]
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("INSERT INTO faces (name, description, hash_id, is_celebrity, copyright_info, uploader_id, original_price, ai_declaration) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
             (d.get('name'), d.get('description'), hid, d.get('is_celebrity',0), d.get('copyright_info'), u['id'], d.get('price',0), d.get('ai_declaration')))
    conn.commit(); conn.close()
    return jsonify({"success": True, "hash_id": hid})

@app.route('/api/faces/<int:fid>', methods=['GET'])
def get_face(fid):
    u = get_user(request.headers.get('X-API-Key'))
    conn = sqlite3.connect(str(DB_PATH)); conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM faces WHERE id=? AND status='active'", (fid,))
    f = c.fetchone()
    if not f: conn.close(); return jsonify({"error": "不存在"}), 404
    price = f["original_price"] or 0
    if price == 0 or (u and charge(u['id'], price)):
        c.execute("UPDATE faces SET usage_count = usage_count + 1 WHERE id=?", (fid,))
        if price > 0 and f["uploader_id"]: pay_uploader(f["uploader_id"], price)
        conn.commit()
        conn.close()
        return jsonify({"name": f["name"], "description": f["description"], "copyright_info": f["copyright_info"], "ai_declaration": f["ai_declaration"]})
    conn.close()
    return jsonify({"error": "需付费", "price": price}), 402

@app.route('/api/works', methods=['GET'])
def get_works():
    conn = sqlite3.connect(str(DB_PATH)); conn.row_factory = sqlite3.Row
    c = conn.cursor()
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
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("INSERT INTO works (title, description, content, work_type, hash_id, author_name, uploader_id, original_price, ai_declaration) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
             (d.get('title'), d.get('description'), d.get('content'), d.get('work_type'), hid, d.get('author_name'), u['id'], d.get('price',0), d.get('ai_declaration')))
    conn.commit(); conn.close()
    return jsonify({"success": True, "hash_id": hid})

@app.route('/api/works/<int:wid>', methods=['GET'])
def get_work(wid):
    u = get_user(request.headers.get('X-API-Key'))
    conn = sqlite3.connect(str(DB_PATH)); conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM works WHERE id=? AND status='active'", (wid,))
    w = c.fetchone()
    if not w: conn.close(); return jsonify({"error": "不存在"}), 404
    price = w["original_price"] or 0
    if price == 0 or (u and charge(u['id'], price)):
        c.execute("UPDATE works SET usage_count = usage_count + 1 WHERE id=?", (wid,))
        if price > 0 and w["uploader_id"]: pay_uploader(w["uploader_id"], price)
        conn.commit()
        conn.close()
        return jsonify({"title": w["title"], "description": w["description"], "content": w["content"], "author_name": w["author_name"]})
    conn.close()
    return jsonify({"error": "需付费", "price": price}), 402

@app.route('/api/stats', methods=['GET'])
def stats():
    conn = sqlite3.connect(str(DB_PATH)); c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM faces WHERE status='active'")
    fc = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM works WHERE status='active'")
    wc = c.fetchone()[0]
    c.execute("SELECT SUM(usage_count) FROM faces")
    fu = c.fetchone()[0] or 0
    c.execute("SELECT SUM(usage_count) FROM works")
    wu = c.fetchone()[0] or 0
    c.execute("SELECT SUM(platform_fee) FROM revenues")
    pr = c.fetchone()[0] or 0
    conn.close()
    return jsonify({"faces": fc, "works": wc, "uses": fu+wu, "platform_revenue": pr, "fee_rate": "1%"})

@app.route('/api/my', methods=['GET'])
def my_uploads():
    u = get_user(request.headers.get('X-API-Key'))
    if not u: return jsonify({"error": "未授权"}), 401
    conn = sqlite3.connect(str(DB_PATH)); conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM faces WHERE uploader_id=?", (u['id'],))
    fs = [dict(r) for r in c.fetchall()]
    c.execute("SELECT * FROM works WHERE uploader_id=?", (u['id'],))
    ws = [dict(r) for r in c.fetchall()]
    c.execute("SELECT SUM(amount-platform_fee) FROM revenues WHERE uploader_id=?", (u['id'],))
    earn = c.fetchone()[0] or 0
    # 获取余额
    c.execute("SELECT balance FROM users WHERE id=?", (u['id'],))
    balance = c.fetchone()[0] or 0
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
    
    conn = sqlite3.connect(str(DB_PATH)); conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    if chart_mode:
        # 返回最近7天每日收益（图表用）
        c.execute("""
            SELECT DATE(created_at) as date, SUM(amount - platform_fee) as amount
            FROM revenues
            WHERE uploader_id = ? AND created_at >= DATE('now', '-7 days')
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
        WHERE ul.user_id = ?
        ORDER BY ul.timestamp DESC
        LIMIT ? OFFSET ?
    """, (u['id'], limit, offset))
    face_history = [dict(r) for r in c.fetchall()]
    
    # 获取作品查看历史
    c.execute("""
        SELECT 'work' as type, w.title as target_name, ul.timestamp, ul.amount_paid,
               '作品查看' as usage_type, '已结算' as status
        FROM usage_logs ul
        JOIN works w ON ul.target_id = w.id
        WHERE ul.user_id = ?
        ORDER BY ul.timestamp DESC
        LIMIT ? OFFSET ?
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
    
    conn = sqlite3.connect(str(DB_PATH)); conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT * FROM faces WHERE uploader_id=?", (u['id'],))
    faces = [dict(r) for r in c.fetchall()]
    
    c.execute("SELECT * FROM works WHERE uploader_id=?", (u['id'],))
    works = [dict(r) for r in c.fetchall()]
    
    c.execute("SELECT * FROM revenues WHERE uploader_id=?", (u['id'],))
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

@app.route('/api/llm/portrait-check', methods=['POST'])
def llm_portrait_check():
    """AI模型调用的肖像检查接口"""
    data = request.json
    prompt = data.get('prompt', '')
    
    # 检查是否涉及人脸/肖像
    face_keywords = ['脸', '肖像', '人物', '明星', '演员', '人像', '照片', 'face', 'portrait', 'celebrity']
    
    needs_check = any(kw in prompt.lower() for kw in face_keywords)
    
    if not needs_check:
        return jsonify({"needs_permission": False, "message": "不涉及肖像"})
    
    # 返回需要授权提示
    return jsonify({
        "needs_permission": True,
        "message": "检测到可能涉及肖像的内容，请先查询PortraitHub获取授权",
        "api_endpoint": "/api/faces",
        "license_info": "请联系肖像所有者获取授权"
    })

@app.route('/api/llm/verify', methods=['POST'])
def llm_verify():
    """验证肖像使用权"""
    data = request.json
    face_id = data.get('face_id')
    usage_purpose = data.get('purpose', 'AI生成')
    
    if not face_id:
        return jsonify({"error": "缺少face_id"}), 400
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM faces WHERE id=? AND status='active'", (face_id,))
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

@app.route('/api/upload/portrait', methods=['POST'])
def upload_portrait():
    """上传肖像 - 需要登录且只能上传一张"""
    api_key = request.headers.get('X-API-Key')
    user = get_user(api_key)
    if not user:
        return jsonify({"error": "请先登录"}), 401
    
    # 检查是否已上传过
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT id FROM faces WHERE uploader_id=?", (user['id'],))
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
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
             (name, portrait_path, hash_id, 0, user['id'], 0, ai_declaration, age, id_path))
    conn.commit()
    face_id = c.lastrowid
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

