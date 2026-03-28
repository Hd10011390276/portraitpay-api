"""Database abstraction layer - supports both SQLite (local) and PostgreSQL (Railway)."""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Detect Railway PostgreSQL automatically
def _detect_postgres_url():
    """Auto-detect Railway PostgreSQL from environment or proxy hostname pattern."""
    if os.environ.get('DATABASE_URL'):
        return os.environ['DATABASE_URL']
    
    if os.environ.get('RAILWAY_ENVIRONMENT'):
        for key in ['POSTGRES_URL', 'PG_URL', 'DATABASE_PUBLIC_URL']:
            if os.environ.get(key):
                return os.environ[key]
    
    pg_host = os.environ.get('POSTGRES_HOST')
    pg_port = os.environ.get('POSTGRES_PORT', '5432')
    pg_user = os.environ.get('POSTGRES_USER', 'postgres')
    pg_pass = os.environ.get('POSTGRES_PASSWORD', '')
    pg_db = os.environ.get('POSTGRES_DB', 'railway')
    
    if pg_host and pg_pass:
        return f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
    
    return None

_postgres_url = _detect_postgres_url()
_use_postgres = False

if _postgres_url:
    try:
        import psycopg2
        import psycopg2.extras
        _use_postgres = True
        logger.info(f"PostgreSQL detected: {_postgres_url[:40]}...")
    except ImportError:
        logger.warning("psycopg2 not installed, falling back to SQLite")
        _postgres_url = None

# SQLite paths
_local_dir = Path(os.environ.get('RAILWAY_PRIVATE_DIR', str(Path(__file__).parent / 'data')))
_local_dir.mkdir(parents=True, exist_ok=True)
_db_path = _local_dir / 'portraitpay.db'

def get_db_conn():
    """Get a DB connection and cursor (not using a context manager).
    Returns (conn, cursor, is_postgres).
    Caller must call conn.commit() and conn.close() when done.
    """
    if _use_postgres and _postgres_url:
        conn = psycopg2.connect(_postgres_url, sslmode='require')
        conn.row_factory = psycopg2.extras.RealDictCursor
        cursor = conn.cursor()
        return conn, cursor, True
    else:
        import sqlite3
        conn = sqlite3.connect(str(_db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        return conn, cursor, False

class DbRow:
    """Dict-like wrapper for sqlite3.Row to match psycopg2 RealDictCursor behavior."""
    __slots__ = ('_row',)
    def __init__(self, row):
        self._row = row
    def __getitem__(self, key):
        return self._row[key]
    def __setitem__(self, key, val):
        self._row[key] = val
    def __delitem__(self, key):
        del self._row[key]
    def __contains__(self, key):
        return key in self._row.keys()
    def __iter__(self):
        return iter(self._row.keys())
    def __len__(self):
        return len(self._row.keys())
    def __repr__(self):
        return str(dict(self))
    def __str__(self):
        return str(dict(self))
    def __eq__(self, other):
        if isinstance(other, dict):
            return dict(self) == other
        return dict(self) == dict(other)
    def __hash__(self):
        return hash(self._row['id']) if 'id' in self._row.keys() else id(self)
    def get(self, key, default=None):
        try:
            return self._row[key]
        except (KeyError, IndexError, TypeError):
            return default
    def keys(self):
        return self._row.keys()
    def values(self):
        return [self._row[k] for k in self._row.keys()]
    def items(self):
        return [(k, self._row[k]) for k in self._row.keys()]

def _wrap_row(row):
    """Wrap sqlite3.Row in DbRow for consistent dict-like behavior."""
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    return DbRow(row)

def get_db():
    """Context manager: yields (conn, cursor, is_postgres)."""
    conn, cursor, is_postgres = get_db_conn()
    try:
        yield conn, cursor, is_postgres
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def last_insert_id(conn, cursor, is_postgres):
    """Get the last inserted row ID using conn."""
    if is_postgres:
        cursor.execute("SELECT lastval()")
        return cursor.fetchone()[0]
    else:
        cursor.execute("SELECT last_insert_rowid()")
        return cursor.fetchone()[0]

def dict_from_row(row):
    """Convert a DbRow/sqlite3.Row/RealDictRow to a plain dict."""
    if row is None:
        return None
    if isinstance(row, dict):
        return dict(row)
    try:
        return dict(zip(row.keys(), [row[k] for k in row.keys()]))
    except:
        return str(row)

def init_database():
    """Create all tables."""
    import sqlite3
    
    common_tables = {
        'users': '''id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash VARCHAR(256) NOT NULL,
            api_key VARCHAR(128) UNIQUE,
            balance FLOAT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_admin BOOLEAN DEFAULT FALSE,
            email VARCHAR(255),
            verification_code VARCHAR(10),
            verified BOOLEAN DEFAULT FALSE''',
        'faces': '''id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            hash_id VARCHAR(64) UNIQUE,
            image_path VARCHAR(512),
            id_image_path VARCHAR(512),
            is_celebrity BOOLEAN DEFAULT FALSE,
            uploader_id INTEGER REFERENCES users(id),
            price FLOAT DEFAULT 5.0,
            original_price FLOAT,
            revenue FLOAT DEFAULT 0,
            usage_count INTEGER DEFAULT 0,
            age INTEGER DEFAULT 0,
            ai_declaration BOOLEAN DEFAULT FALSE,
            copyright_info TEXT,
            status VARCHAR(20) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP''',
        'face_embeddings': '''id SERIAL PRIMARY KEY,
            face_id INTEGER REFERENCES faces(id),
            embedding TEXT NOT NULL,
            model_name VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP''',
        'works': '''id SERIAL PRIMARY KEY,
            work_name VARCHAR(200),
            face_id INTEGER REFERENCES faces(id),
            creator_id INTEGER REFERENCES users(id),
            image_path VARCHAR(512),
            prompt TEXT,
            style VARCHAR(50),
            thumbnail_path VARCHAR(512),
            status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP''',
        'transactions': '''id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            type VARCHAR(20) NOT NULL,
            amount FLOAT NOT NULL,
            status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            face_id INTEGER,
            work_id INTEGER,
            description TEXT''',
    }

    sqlite_tables = {
        'users': '''id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            api_key TEXT UNIQUE,
            balance REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_admin INTEGER DEFAULT 0,
            email TEXT,
            verification_code TEXT,
            verified INTEGER DEFAULT 0''',
        'faces': '''id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            hash_id TEXT UNIQUE,
            image_path TEXT,
            id_image_path TEXT,
            is_celebrity INTEGER DEFAULT 0,
            uploader_id INTEGER,
            price REAL DEFAULT 5.0,
            original_price REAL,
            revenue REAL DEFAULT 0,
            usage_count INTEGER DEFAULT 0,
            age INTEGER DEFAULT 0,
            ai_declaration INTEGER DEFAULT 0,
            copyright_info TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP''',
        'face_embeddings': '''id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER,
            embedding TEXT NOT NULL,
            model_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP''',
        'works': '''id INTEGER PRIMARY KEY AUTOINCREMENT,
            work_name TEXT,
            face_id INTEGER,
            creator_id INTEGER,
            image_path TEXT,
            prompt TEXT,
            style TEXT,
            thumbnail_path TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP''',
        'transactions': '''id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            face_id INTEGER,
            work_id INTEGER,
            description TEXT''',
    }

    if _use_postgres and _postgres_url:
        import psycopg2
        conn = psycopg2.connect(_postgres_url, sslmode='require')
        c = conn.cursor()
        for table, cols in common_tables.items():
            c.execute(f'CREATE TABLE IF NOT EXISTS {table} ({cols})')
        conn.commit()
        conn.close()
        logger.info("PostgreSQL tables created/verified")
    else:
        conn = sqlite3.connect(str(_db_path))
        c = conn.cursor()
        for table, cols in sqlite_tables.items():
            c.execute(f'CREATE TABLE IF NOT EXISTS {table} ({cols})')
        conn.commit()
        conn.close()
        logger.info("SQLite tables created/verified")
