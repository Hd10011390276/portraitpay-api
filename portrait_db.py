"""Database layer: SQLite for local dev, PostgreSQL for Railway production."""
import os
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# ── PostgreSQL detection ──────────────────────────────────────────────────────
_PG_URL = None
_USE_PG = False

_db_url = os.environ.get("DATABASE_URL")
if not _db_url and os.environ.get("RAILWAY_ENVIRONMENT"):
    for _key in ("POSTGRES_URL", "PG_URL", "DATABASE_PUBLIC_URL"):
        _db_url = os.environ.get(_key)
        if _db_url:
            break

if not _db_url:
    _host = os.environ.get("POSTGRES_HOST")
    _port = os.environ.get("POSTGRES_PORT", "5432")
    _user = os.environ.get("POSTGRES_USER", "postgres")
    _pass = os.environ.get("POSTGRES_PASSWORD", "")
    _db   = os.environ.get("POSTGRES_DB", "railway")
    if _host and _pass:
        _db_url = f"postgresql://{_user}:{_pass}@{_host}:{_port}/{_db}"

if _db_url:
    try:
        import psycopg2
        import psycopg2.extras
        _PG_URL = _db_url
        _USE_PG = True
        logger.info(f"PostgreSQL: detected ({_db_url[:40]}...)")
    except ImportError:
        logger.warning("psycopg2 not installed; falling back to SQLite")
        _PG_URL = None

# ── Local paths ───────────────────────────────────────────────────────────────
_DATA_DIR = Path(os.environ.get("RAILWAY_PRIVATE_DIR", str(Path(__file__).parent / "data")))
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DATA_DIR / "portraitpay.db"

# ── Core helpers ──────────────────────────────────────────────────────────────
def get_db_conn():
    """Return (conn, cursor, is_pg). Caller must commit() and close()."""
    if _USE_PG and _PG_URL:
        conn = psycopg2.connect(_PG_URL, sslmode="require")
        conn.cursor_factory = psycopg2.extras.RealDictCursor
        return conn, conn.cursor(), True
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn, conn.cursor(), False

def last_insert_id(_conn, _cursor, _is_pg):
    if _is_pg:
        _cursor.execute("SELECT lastval()")
        return _cursor.fetchone()[0]
    _cursor.execute("SELECT last_insert_rowid()")
    return _cursor.fetchone()[0]

def dict_from_row(row):
    """Alias for row_to_dict for backward compatibility."""
    return row_to_dict(row)

def row_to_dict(row):
    if row is None:
        return None
    if isinstance(row, dict):
        return dict(row)
    try:
        return dict(zip(row.keys(), [row[k] for k in row.keys()]))
    except Exception:
        return str(row)

# Backward compatibility alias
dict_from_row = row_to_dict

# ── Schema definitions ────────────────────────────────────────────────────────
_PG_SCHEMA = {
    "users": """id SERIAL PRIMARY KEY, username VARCHAR(50) UNIQUE NOT NULL,
        password_hash VARCHAR(256) NOT NULL, api_key VARCHAR(128) UNIQUE,
        balance FLOAT DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_admin BOOLEAN DEFAULT FALSE, email VARCHAR(255),
        verification_code VARCHAR(10), verified BOOLEAN DEFAULT FALSE""",
    "faces": """id SERIAL PRIMARY KEY, name VARCHAR(100) NOT NULL,
        hash_id VARCHAR(64) UNIQUE, image_path VARCHAR(512), id_image_path VARCHAR(512),
        is_celebrity BOOLEAN DEFAULT FALSE, uploader_id INTEGER REFERENCES users(id),
        price FLOAT DEFAULT 5.0, original_price FLOAT, revenue FLOAT DEFAULT 0,
        usage_count INTEGER DEFAULT 0, age INTEGER DEFAULT 0,
        ai_declaration BOOLEAN DEFAULT FALSE, copyright_info TEXT,
        status VARCHAR(20) DEFAULT 'active', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP""",
    "face_embeddings": """id SERIAL PRIMARY KEY, face_id INTEGER REFERENCES faces(id),
        embedding TEXT NOT NULL, model_name VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP""",
    "works": """id SERIAL PRIMARY KEY, work_name VARCHAR(200), face_id INTEGER,
        creator_id INTEGER REFERENCES users(id), image_path VARCHAR(512),
        prompt TEXT, style VARCHAR(50), thumbnail_path VARCHAR(512),
        status VARCHAR(20) DEFAULT 'pending', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP""",
    "transactions": """id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id),
        type VARCHAR(20) NOT NULL, amount FLOAT NOT NULL,
        status VARCHAR(20) DEFAULT 'pending', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        face_id INTEGER, work_id INTEGER, description TEXT""",
    "revenues": """id SERIAL PRIMARY KEY, source_type VARCHAR(20) NOT NULL,
        amount FLOAT NOT NULL, platform_fee FLOAT NOT NULL,
        uploader_id INTEGER REFERENCES users(id),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP""",
}

_SQLITE_SCHEMA = {
    k: v.replace("SERIAL", "INTEGER PRIMARY KEY AUTOINCREMENT")
       .replace(" BOOLEAN DEFAULT FALSE", " INTEGER DEFAULT 0")
       .replace(" VARCHAR", " TEXT")
       .replace(" FLOAT", " REAL")
       .replace(" TIMESTAMP", " TIMESTAMP")
       .replace(" REFERENCES users(id)", "")
       .replace(" REFERENCES faces(id)", "")
       .replace("REFERENCES users(id)", "")
       .replace("REFERENCES faces(id)", "")
    for k, v in _PG_SCHEMA.items()
}
# Fix remaining PG types for SQLite
_SQLITE_SCHEMA["users"] = """id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
        api_key TEXT UNIQUE, balance REAL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_admin INTEGER DEFAULT 0, email TEXT,
        verification_code TEXT, verified INTEGER DEFAULT 0"""
_SQLITE_SCHEMA["faces"] = """id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, hash_id TEXT UNIQUE, image_path TEXT, id_image_path TEXT,
        is_celebrity INTEGER DEFAULT 0, uploader_id INTEGER,
        price REAL DEFAULT 5.0, original_price REAL, revenue REAL DEFAULT 0,
        usage_count INTEGER DEFAULT 0, age INTEGER DEFAULT 0,
        ai_declaration INTEGER DEFAULT 0, copyright_info TEXT,
        status TEXT DEFAULT 'active', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"""
_SQLITE_SCHEMA["face_embeddings"] = """id INTEGER PRIMARY KEY AUTOINCREMENT,
        face_id INTEGER, embedding TEXT NOT NULL, model_name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"""
_SQLITE_SCHEMA["works"] = """id INTEGER PRIMARY KEY AUTOINCREMENT,
        work_name TEXT, face_id INTEGER, creator_id INTEGER, image_path TEXT,
        prompt TEXT, style TEXT, thumbnail_path TEXT,
        status TEXT DEFAULT 'pending', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"""
_SQLITE_SCHEMA["transactions"] = """id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, type TEXT NOT NULL, amount REAL NOT NULL,
        status TEXT DEFAULT 'pending', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        face_id INTEGER, work_id INTEGER, description TEXT"""
_SQLITE_SCHEMA["revenues"] = """id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_type TEXT NOT NULL, amount REAL NOT NULL, platform_fee REAL NOT NULL,
        uploader_id INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"""

# ── Init ──────────────────────────────────────────────────────────────────────
def init_database():
    if _USE_PG and _PG_URL:
        import psycopg2
        conn = psycopg2.connect(_PG_URL, sslmode="require")
        cur = conn.cursor()
        for _tbl, _cols in _PG_SCHEMA.items():
            cur.execute(f"CREATE TABLE IF NOT EXISTS {_tbl} ({_cols})")
        conn.commit()
        conn.close()
    else:
        conn = sqlite3.connect(str(_DB_PATH))
        cur = conn.cursor()
        for _tbl, _cols in _SQLITE_SCHEMA.items():
            cur.execute(f"CREATE TABLE IF NOT EXISTS {_tbl} ({_cols})")
        conn.commit()
        conn.close()
    logger.info("Database initialized")
