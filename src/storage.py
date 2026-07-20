import cv2
import os
import sqlite3
import base64
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

DB_PATH = "data/profiles.db"

def init_directories(data_dir="data"):
    """Create data directory if not exists."""
    os.makedirs(data_dir, exist_ok=True)

def _get_connection():
    """Creates a new SQLite connection with WAL mode enabled for concurrent read performance."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    """Initializes the SQLite3 database schema for storing profiles and embeddings."""
    init_directories()
    conn = _get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            # Create profiles table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                label_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """)
            # Create embeddings table (one student can have multiple face vectors for higher accuracy)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER,
                vector BLOB NOT NULL,
                FOREIGN KEY (profile_id) REFERENCES profiles(label_id) ON DELETE CASCADE
            )
            """)
            # Cache version counter for cross-worker cache invalidation
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_version (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL DEFAULT 0
            )
            """)
            cursor.execute("INSERT OR IGNORE INTO cache_version (id, version) VALUES (1, 0)")
    finally:
        conn.close()

def get_cache_version() -> int:
    """Returns the current cache version counter. Microsecond-fast single integer read."""
    if not os.path.exists(DB_PATH):
        return 0
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version FROM cache_version WHERE id = 1")
        row = cursor.fetchone()
        return row[0] if row else 0
    except Exception:
        return 0
    finally:
        if conn:
            conn.close()

def increment_cache_version():
    """Increments the cache version counter, signaling all workers to reload embeddings."""
    conn = None
    try:
        conn = _get_connection()
        with conn:
            conn.execute("UPDATE cache_version SET version = version + 1 WHERE id = 1")
    except Exception as e:
        print(f"Failed to increment cache version: {e}")
    finally:
        if conn:
            conn.close()

def decode_base64_image(base64_data: str) -> np.ndarray:
    """Decodes a base64 encoded BGR image."""
    if ',' in base64_data:
        base64_data = base64_data.split(',')[1]
        
    try:
        image_bytes = base64.b64decode(base64_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

def register_user(label_id: int, name: str) -> bool:
    """Inserts or replaces a student profile in the database."""
    conn = None
    try:
        conn = _get_connection()
        with conn:
            cursor = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT OR REPLACE INTO profiles (label_id, name, created_at) VALUES (?, ?, ?)",
                (label_id, name, now)
            )
        return True
    except Exception as e:
        print(f"Failed to register user in SQLite: {e}")
        return False
    finally:
        if conn:
            conn.close()

def save_embedding(label_id: int, embedding: np.ndarray) -> bool:
    """Serializes the float vector and saves it into the database."""
    conn = None
    try:
        # Flatten and cast to float32
        vec_f32 = embedding.flatten().astype(np.float32)
        # Convert NumPy array to raw binary bytes
        vector_bytes = vec_f32.tobytes()
        
        conn = _get_connection()
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO embeddings (profile_id, vector) VALUES (?, ?)",
                (label_id, sqlite3.Binary(vector_bytes))
            )
        # Signal all workers to reload their embedding cache
        increment_cache_version()
        return True
    except Exception as e:
        print(f"Failed to save embedding in SQLite: {e}")
        return False
    finally:
        if conn:
            conn.close()

def load_embeddings() -> Dict[int, List[np.ndarray]]:
    """Loads all student face vectors from the database and deserializes them."""
    embeddings_map = {}
    if not os.path.exists(DB_PATH):
        return {}
        
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT profile_id, vector FROM embeddings")
        rows = cursor.fetchall()
        
        for profile_id, vector_bytes in rows:
            # Deserialization: Convert raw bytes back to 128-dim NumPy float32 array
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            
            if profile_id not in embeddings_map:
                embeddings_map[profile_id] = []
            embeddings_map[profile_id].append(vector)
            
    except Exception as e:
        print(f"Failed to load embeddings: {e}")
    finally:
        if conn:
            conn.close()
        
    return embeddings_map

def load_profiles() -> Dict[int, dict]:
    """Loads all student profiles (metadata)."""
    profiles = {}
    if not os.path.exists(DB_PATH):
        return {}
        
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT label_id, name, created_at FROM profiles")
        rows = cursor.fetchall()
        
        for label_id, name, created_at in rows:
            profiles[label_id] = {
                "name": name,
                "created_at": created_at
            }
    except Exception as e:
        print(f"Failed to load profiles: {e}")
    finally:
        if conn:
            conn.close()
    return profiles

def delete_user(label_id: int) -> bool:
    """Deletes a student profile and all associated face vectors (Cascade delete)."""
    conn = None
    try:
        conn = _get_connection()
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM profiles WHERE label_id = ?", (label_id,))
            # Also clean embeddings table (SQLite handles cascade delete if configured, manual cleanup just in case)
            cursor.execute("DELETE FROM embeddings WHERE profile_id = ?", (label_id,))
        # Signal all workers to reload their embedding cache
        increment_cache_version()
        return True
    except Exception as e:
        print(f"Failed to delete user: {e}")
        return False
    finally:
        if conn:
            conn.close()
