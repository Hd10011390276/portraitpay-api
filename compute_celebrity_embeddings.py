#!/usr/bin/env python3
"""
Compute ArcFace (buffalo_l) embeddings for celebrities.
Uses insightface + onnxruntime in conda base environment.

Usage:
    conda run -n base python compute_celebrity_embeddings.py run [--limit N]
    conda run -n base python compute_celebrity_embeddings.py test
"""

import os
import sys
import json
import sqlite3
import time
import logging
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / "data" / "portraitpay.db"

# Wikipedia title mapping for celebrities whose names don't match
WIKI_TITLE_MAP = {
    # Business
    '马斯克': 'Elon_Musk',
    '马云': 'Jack_Ma',
    '马化腾': 'Pony_Ma',
    '刘强东': 'Liu_Qiangdong',
    '张一鸣': 'Zhang_Yiming',
    '王健林': 'Wang_Jianlin',
    '王兴': 'Wang_Xing',  # Meituan founder - skip
    '雷军': 'Lei_Jun',
    '董明珠': 'Dong_Mingzhu',
    '贝索斯': 'Jeff_Bezos',
    '比尔·盖茨': 'Bill_Gates',
    '蒂姆·库克': 'Tim_Cook',
    '扎克伯格': 'Mark_Zuckerberg',
    '巴菲特': 'Warren_Buffett',
    '索罗斯': 'George_Soros',
    '许家印': 'Evergrande',
    
    # Sports - Football
    '梅西': 'Lionel_Messi',
    'C罗': 'Cristiano_Ronaldo',
    '姆巴佩': 'Kylian_Mbappe',
    '哈兰德': 'Erling_Haaland',
    '贝克汉姆': 'David_Beckham',
    '孙兴慜': 'Son_Heung-min',
    '武磊': 'Wu_Lei_(footballer)',
    
    # Sports - Basketball
    '易建联': 'Yi_Jianlian',
    '科比·布莱恩特': 'Kobe_Bryant',
    '勒布朗·詹姆斯': 'LeBron_James',
    '斯蒂芬·库里': 'Stephen_Curry',
    '迈克尔·乔丹': 'Michael_Jordan',
    
    # Sports - Tennis
    '德约科维奇': 'Novak_Djokovic',
    '费德勒': 'Roger_Federer',
    '纳达尔': 'Rafael_Nadal',
    
    # Sports - F1 / Other
    '汉密尔顿': 'Lewis_Hamilton',
    '维斯塔潘': 'Max_Verstappen',
    '谷爱凌': 'Eileen_Gu',
    '全红婵': 'Quan_Hongchan',
    
    # Sports - Football
    '内马尔': 'Neymar',
    
    # Sports - Other
    '舒马赫': 'Michael_Schumacher',
    '郎平': 'Lang_Ping',
    '苏炳添': 'Su_Bingtian',
    
    # Korean actors
    '马东锡': 'Ma_Dong-seok',
    '宋康昊': 'Song_Kang-ho',
    '金秀贤': 'Kim_Soo-hyun',
    '全智贤': 'Jun_Ji-hyun',
    '裴斗娜': 'Bae_Doona',
    
    # Chinese actors
    '周润发': 'Chow_Yun-fat',
    '邓超': 'Deng_Chao',
    '沈腾': 'Shen_Teng',
    '马丽': None,  # No usable image on English Wikipedia
    '贾玲': None,  # No English Wikipedia article
    'Chris Evans': 'Chris_Evans_(actor)',
    '许家印': None,  # No usable face image in Wikipedia
}


def get_wiki_image(name, retry_original=True):
    """
    Get face image from Wikipedia for a celebrity.
    Returns (image_bytes, error_msg).
    """
    # Skip celebrities without Wikipedia mappings
    wiki_title = WIKI_TITLE_MAP.get(name)
    if wiki_title is None:
        return None, 'no_wiki_mapping'
    
    try:
        title_encoded = urllib.parse.quote(wiki_title.replace(' ', '_').replace('·', '_').encode('utf-8'))
        
        # Step 1: Get image URL from Wikipedia REST API
        api_url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{title_encoded}'
        req = urllib.request.Request(api_url, headers={
            'User-Agent': 'PortraitPay/1.0 (face-recognition-research)',
            'Accept': 'application/json',
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        
        # Get thumbnail or original
        img_url = None
        if 'thumbnail' in data:
            img_url = data['thumbnail']['source']
        elif 'originalimage' in data:
            img_url = data['originalimage']['source']
        
        if not img_url:
            return None, 'no_image_in_response'
        
        # Step 2: Download the image
        img_req = urllib.request.Request(img_url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        })
        with urllib.request.urlopen(img_req, timeout=20) as resp:
            img_data = resp.read()
        
        return img_data, None
        
    except urllib.error.HTTPError as e:
        if e.code == 404 and retry_original:
            # Try the original image path directly (some articles have images but no thumbnail)
            wiki_title = WIKI_TITLE_MAP.get(name, name)
            orig_url = f"https://upload.wikimedia.org/wikipedia/commons/{wiki_title.replace('%C3%A9', '%C3%A9').replace(' ', '_')}.jpg"
            try:
                img_req = urllib.request.Request(orig_url, headers={
                    'User-Agent': 'Mozilla/5.0',
                })
                with urllib.request.urlopen(img_req, timeout=20) as resp:
                    return resp.read(), None
            except:
                pass
        return None, f'http_{e.code}'
    except Exception as e:
        return None, str(e)


def get_face_model():
    """Load InsightFace buffalo_l model."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def compute_embedding(app, image_data):
    """Compute face embedding from image bytes using InsightFace."""
    import cv2
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "decode_failed"
    faces = app.get(img)
    if not faces:
        return None, "no_face_detected"
    largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return largest.embedding.tolist(), None


def get_celebrities_without_embeddings():
    """Get all celebrities that need embedding computation."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT f.id, f.name, ci.category
        FROM faces f
        JOIN celebrity_info ci ON f.id = ci.face_id
        LEFT JOIN face_embeddings fe ON f.id = fe.face_id
        WHERE fe.id IS NULL AND f.status='active' AND f.is_celebrity=1
    """)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_embedding(face_id, embedding, model_name="buffalo_l"):
    """Save embedding to database."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT id FROM face_embeddings WHERE face_id=?", (face_id,))
    existing = c.fetchone()
    if existing:
        c.execute("""
            UPDATE face_embeddings 
            SET embedding=?, model_name=?, created_at=CURRENT_TIMESTAMP
            WHERE face_id=?
        """, (json.dumps(embedding), model_name, face_id))
    else:
        c.execute("""
            INSERT INTO face_embeddings (face_id, embedding, model_name)
            VALUES (?, ?, ?)
        """, (face_id, json.dumps(embedding), model_name))
    conn.commit()
    conn.close()


def run_embedding_computation(limit=None):
    """Main: compute embeddings for all celebrities."""
    app = get_face_model()
    logger.info("InsightFace buffalo_l model loaded successfully")
    
    celebrities = get_celebrities_without_embeddings()
    if limit:
        celebrities = celebrities[:limit]
    
    logger.info(f"Computing embeddings for {len(celebrities)} celebrities...")
    
    results = {"success": 0, "failed": 0, "no_image": 0, "no_face": 0}
    
    for i, celeb in enumerate(celebrities):
        name = celeb['name']
        face_id = celeb['id']
        
        # Download image from Wikipedia
        img_data, err = get_wiki_image(name)
        if err:
            logger.warning(f"  [{i+1}/{len(celebrities)}] ✗ {name}: {err}")
            results["no_image"] += 1
            continue
        
        # Compute embedding
        embedding, emb_err = compute_embedding(app, img_data)
        
        if embedding:
            save_embedding(face_id, embedding, model_name="buffalo_l")
            logger.info(f"  [{i+1}/{len(celebrities)}] ✓ {name}: 512-dim embedding saved")
            results["success"] += 1
        else:
            logger.warning(f"  [{i+1}/{len(celebrities)}] ✗ {name}: {emb_err}")
            results["failed"] += 1
        
        # Rate limit to respect Wikipedia
        time.sleep(1.5)
    
    logger.info(f"Embedding computation complete: {results}")
    
    # Print summary
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("""
        SELECT COUNT(*) FROM face_embeddings fe 
        JOIN faces f ON fe.face_id=f.id 
        WHERE f.is_celebrity=1
    """)
    total = c.fetchone()[0]
    conn.close()
    logger.info(f"Total celebrity embeddings now in DB: {total}")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
        run_embedding_computation(limit)
    else:
        print(__doc__)
