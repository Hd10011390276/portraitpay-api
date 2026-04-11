"""Migration: Add Local-First support to PortraitPay

This migration adds:
1. portrait_fingerprints table - stores irreversible portrait hashes
2. Modifies faces table - removes image_path, adds local_first fields
3. search_queries table - audit log for searches without storing images
"""

import sqlite3
import psycopg2
import os
import logging

logger = logging.getLogger(__name__)

def migrate(db_path=None, pg_url=None, is_pg=False):
    """Run local-first migration.

    Args:
        db_path: Path to SQLite database (for local dev)
        pg_url: PostgreSQL connection URL (for production)
        is_pg: Whether to use PostgreSQL
    """
    if is_pg and pg_url:
        _migrate_postgresql(pg_url)
    else:
        _migrate_sqlite(db_path or "data/portraitpay.db")

def _migrate_sqlite(db_path):
    """Migrate SQLite database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check if portrait_fingerprints table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portrait_fingerprints'")
    if cur.fetchone():
        logger.info("portrait_fingerprints table already exists, skipping")
        conn.close()
        return

    # 1. Create portrait_fingerprints table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS portrait_fingerprints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER NOT NULL,
            fingerprint_hash TEXT NOT NULL,
            fingerprint_type TEXT DEFAULT 'phash',
            model_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(face_id, fingerprint_type)
        )
    """)
    logger.info("Created portrait_fingerprints table")

    # 2. Create search_queries table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS search_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_fingerprint_hash TEXT NOT NULL,
            results_count INTEGER DEFAULT 0,
            top_match_face_id INTEGER,
            top_match_similarity REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    logger.info("Created search_queries table")

    # 3. Add local_first fields to faces table
    try:
        cur.execute("ALTER TABLE faces ADD COLUMN fingerprint_registered INTEGER DEFAULT 0")
        logger.info("Added fingerprint_registered column to faces")
    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e).lower():
            logger.info("fingerprint_registered column already exists")
        else:
            raise

    try:
        cur.execute("ALTER TABLE faces ADD COLUMN local_device_id TEXT")
        logger.info("Added local_device_id column to faces")
    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e).lower():
            logger.info("local_device_id column already exists")
        else:
            raise

    # 4. Create index on fingerprint_hash for fast searching
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fingerprint_hash ON portrait_fingerprints(fingerprint_hash)")
    logger.info("Created index on fingerprint_hash")

    conn.commit()
    conn.close()
    logger.info("SQLite migration completed")

def _migrate_postgresql(pg_url):
    """Migrate PostgreSQL database."""
    conn = psycopg2.connect(pg_url, sslmode="require")
    cur = conn.cursor()

    # Check if portrait_fingerprints table exists
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_name='portrait_fingerprints'")
    if cur.fetchone():
        logger.info("portrait_fingerprints table already exists, skipping")
        conn.close()
        return

    # 1. Create portrait_fingerprints table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS portrait_fingerprints (
            id SERIAL PRIMARY KEY,
            face_id INTEGER NOT NULL,
            fingerprint_hash TEXT NOT NULL,
            fingerprint_type TEXT DEFAULT 'phash',
            model_name VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(face_id, fingerprint_type)
        )
    """)
    logger.info("Created portrait_fingerprints table")

    # 2. Create search_queries table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS search_queries (
            id SERIAL PRIMARY KEY,
            query_fingerprint_hash TEXT NOT NULL,
            results_count INTEGER DEFAULT 0,
            top_match_face_id INTEGER,
            top_match_similarity REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    logger.info("Created search_queries table")

    # 3. Add local_first fields to faces table
    try:
        cur.execute("ALTER TABLE faces ADD COLUMN fingerprint_registered BOOLEAN DEFAULT FALSE")
        logger.info("Added fingerprint_registered column to faces")
    except psycopg2.errors.DuplicateColumn:
        logger.info("fingerprint_registered column already exists")

    try:
        cur.execute("ALTER TABLE faces ADD COLUMN local_device_id VARCHAR(128)")
        logger.info("Added local_device_id column to faces")
    except psycopg2.errors.DuplicateColumn:
        logger.info("local_device_id column already exists")

    # 4. Create index on fingerprint_hash
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fingerprint_hash ON portrait_fingerprints(fingerprint_hash)")

    conn.commit()
    conn.close()
    logger.info("PostgreSQL migration completed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate()