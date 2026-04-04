#!/usr/bin/env python3
"""
Test face matching using ArcFace embeddings.
Downloads a test image, computes embedding, matches against celebrity DB.

Usage:
    conda run -n base python test_face_matching.py [image_url]
    conda run -n base python test_face_matching.py --celebrity "Name"
"""

import sys
import json
import sqlite3
import math
import urllib.request
import urllib.parse
import numpy as np

SCRIPT_DIR = __file__.rsplit('/', 1)[0]
DB_PATH = f'{SCRIPT_DIR}/data/portraitpay.db'


def load_model():
    """Load InsightFace buffalo_l model."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def download_image(url):
    """Download image from URL."""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read(), None
    except Exception as e:
        return None, str(e)


def compute_embedding(app, image_data):
    """Compute face embedding from image bytes."""
    import cv2
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "decode_failed"
    faces = app.get(img)
    if not faces:
        return None, "no_face_detected"
    largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return largest.embedding, None


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_celebrity_embeddings():
    """Load all celebrity embeddings from database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT f.id, f.name, f.category, fe.embedding, ci.category as celeb_category
        FROM faces f
        JOIN face_embeddings fe ON f.id = fe.face_id
        JOIN celebrity_info ci ON f.id = ci.face_id
        WHERE f.is_celebrity=1 AND f.status='active'
    """)
    rows = c.fetchall()
    conn.close()
    return [
        {
            'id': r['id'],
            'name': r['name'],
            'category': r['celeb_category'],
            'embedding': json.loads(r['embedding'])
        }
        for r in rows
    ]


def match_face(embedding, celebrities, top_n=5):
    """Match a face embedding against the celebrity database."""
    results = []
    for celeb in celebrities:
        sim = cosine_similarity(np.array(embedding), np.array(celeb['embedding']))
        results.append({
            'name': celeb['name'],
            'category': celeb['category'],
            'similarity': float(sim),
            'celeb_id': celeb['id']
        })
    
    # Sort by similarity (descending)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]


def test_with_image_url(app, url):
    """Test face matching with an image URL."""
    print(f"\n📥 Downloading test image: {url}")
    img_data, err = download_image(url)
    if err:
        print(f"❌ Failed to download image: {err}")
        return
    
    print(f"✅ Image downloaded ({len(img_data)} bytes)")
    
    embedding, err = compute_embedding(app, img_data)
    if err:
        print(f"❌ Failed to compute embedding: {err}")
        return
    
    print(f"✅ 512-dim embedding computed")
    
    celebrities = load_celebrity_embeddings()
    print(f"\n🔍 Matching against {len(celebrities)} celebrities...\n")
    
    matches = match_face(embedding, celebrities, top_n=5)
    
    print("=" * 50)
    print("  🎯 TOP 5 FACE MATCHES")
    print("=" * 50)
    for i, m in enumerate(matches, 1):
        sim_pct = m['similarity'] * 100
        bar_len = int(sim_pct / 2)
        bar = '█' * bar_len + '░' * (50 - bar_len)
        print(f"  {i}. {m['name']} ({m['category']})")
        print(f"     [{bar[:25]}] {sim_pct:.1f}%")
        print()
    
    return matches


def test_with_celebrity_name(app, name):
    """Test: get embedding for a specific celebrity and find similar faces."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT f.id, f.name, fe.embedding
        FROM faces f
        JOIN face_embeddings fe ON f.id = fe.face_id
        WHERE f.name=? AND f.is_celebrity=1
    """, (name,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        print(f"❌ Celebrity '{name}' not found in database")
        return
    
    embedding = json.loads(row['embedding'])
    print(f"\n🔍 Finding faces similar to {name} (ID={row['id']})...\n")
    
    celebrities = load_celebrity_embeddings()
    matches = match_face(embedding, celebrities, top_n=6)
    
    print("=" * 50)
    print(f"  🎯 FACES SIMILAR TO {name}")
    print("=" * 50)
    for i, m in enumerate(matches, 1):
        if m['name'] == name:
            continue  # Skip the same person
        sim_pct = m['similarity'] * 100
        print(f"  {i}. {m['name']} ({m['category']}) - {sim_pct:.1f}%")
    
    return matches


def main():
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    app = load_model()
    print("✅ InsightFace buffalo_l model loaded")
    
    if len(sys.argv) > 1 and sys.argv[1] == '--celebrity':
        if len(sys.argv) < 3:
            print("Usage: python test_face_matching.py --celebrity 'Name'")
            return
        test_with_celebrity_name(app, sys.argv[2])
    
    elif len(sys.argv) > 1:
        url = sys.argv[1]
        test_with_image_url(app, url)
    
    else:
        # Default: test with Elon Musk image
        print("\n🧪 Running default test: Elon Musk image from Wikipedia")
        url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Elon_Musk_%2854816836217%29_%28cropped_2%29_%28b%29.jpg/440px-Elon_Musk_%2854816836217%29_%28cropped_2%29_%28b%29.jpg'
        test_with_image_url(app, url)
        
        print("\n" + "="*50)
        print("🧪 Also testing self-match for 马斯克...")
        test_with_celebrity_name(app, '马斯克')


if __name__ == "__main__":
    main()
