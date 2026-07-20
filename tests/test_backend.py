import os
import sys
import io
import unittest
import unittest.mock
import numpy as np
import sqlite3

# Dynamically add the 'src' directory to Python Path to resolve modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import storage
from recognizer import SFRecognizer
from cache import EmbeddingCache
from app import app

class TestFaceRecognitionBackend(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Set test API key on Flask app config
        app.config["FACE_API_KEY"] = "test-key"
        # Redirect database path to a separate testing database file
        cls.orig_db_path = storage.DB_PATH
        storage.DB_PATH = "data/test_profiles.db"
        storage.init_db()

    @classmethod
    def tearDownClass(cls):
        # Restore original database path and cleanup test database
        storage.DB_PATH = cls.orig_db_path
        if os.path.exists("data/test_profiles.db"):
            os.remove("data/test_profiles.db")

    def setUp(self):
        # Clear database before each test
        conn = sqlite3.connect(storage.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM profiles")
        cursor.execute("DELETE FROM embeddings")
        conn.commit()
        conn.close()

    def test_database_registration_and_deletion(self):
        """Test database profile insertion, checking, and cascade deletion."""
        # Register a test student profile
        label_id = 999
        name = "Test Student"
        self.assertTrue(storage.register_user(label_id, name))
        
        # Verify profile is loaded correctly
        profiles = storage.load_profiles()
        self.assertIn(label_id, profiles)
        self.assertEqual(profiles[label_id]["name"], name)
        
        # Delete profile
        self.assertTrue(storage.delete_user(label_id))
        
        # Verify profile is deleted
        profiles_after = storage.load_profiles()
        self.assertNotIn(label_id, profiles_after)

    def test_database_embedding_serialization(self):
        """Test serialization/deserialization of raw NumPy vectors in SQLite."""
        label_id = 888
        name = "Sample Student"
        self.assertTrue(storage.register_user(label_id, name))
        
        # Generate dummy 128-dim float32 face embedding vector
        dummy_vector = np.random.rand(1, 128).astype(np.float32)
        
        # Save to database
        self.assertTrue(storage.save_embedding(label_id, dummy_vector))
        
        # Load and verify
        embeddings_map = storage.load_embeddings()
        self.assertIn(label_id, embeddings_map)
        self.assertEqual(len(embeddings_map[label_id]), 1)
        
        # Check matching dimension and values (almost equal due to serialization float casting)
        loaded_vector = embeddings_map[label_id][0]
        self.assertEqual(loaded_vector.shape, (128,))
        np.testing.assert_array_almost_equal(dummy_vector.flatten(), loaded_vector, decimal=5)

    def test_cosine_similarity_matching_logic(self):
        """Test the SFace Cosine Similarity vector matching mathematics."""
        recognizer = SFRecognizer()
        
        # Query vector: pointing fully along X-axis
        query = np.array([1.0, 0.0, 0.0])
        
        # Database containing:
        # User 1: vector pointing along X-axis (Angle = 0deg, Cosine Similarity = 1.0)
        # User 2: vector pointing along Y-axis (Angle = 90deg, Cosine Similarity = 0.0)
        # User 3: vector pointing along negative X-axis (Angle = 180deg, Cosine Similarity = -1.0)
        db_embeddings = {
            1: [np.array([1.0, 0.0, 0.0])],
            2: [np.array([0.0, 1.0, 0.0])],
            3: [np.array([-1.0, 0.0, 0.0])]
        }
        
        # Test perfect match (User 1 similarity is 1.0, exceeds threshold 0.363)
        matched_id, score = recognizer.match(query, db_embeddings, threshold=0.363)
        self.assertEqual(matched_id, 1)
        self.assertAlmostEqual(score, 1.0)
        
        # Test poor match / threshold limitation (User 2 similarity is 0.0, below threshold 0.363)
        matched_id_poor, score_poor = recognizer.match(query, {2: db_embeddings[2]}, threshold=0.363)
        self.assertEqual(matched_id_poor, -1)
        self.assertAlmostEqual(score_poor, 0.0)

    def test_flask_status_endpoint(self):
        """Test Flask REST status endpoint query response and CORS headers."""
        client = app.test_client()
        # Supplying headers including X-API-Key
        response = client.get("/status", headers={"X-API-Key": "test-key"})
        
        # Verify HTTP status code
        self.assertEqual(response.status_code, 200)
        
        # Verify JSON properties
        json_data = response.get_json()
        self.assertIn("status", json_data)
        self.assertEqual(json_data["status"], "online")
        self.assertIn("registered_users", json_data)
        
        # Verify CORS headers are attached
        self.assertEqual(response.headers.get("Access-Control-Allow-Origin"), "*")
        self.assertIn("GET", response.headers.get("Access-Control-Allow-Methods"))

    def test_flask_status_endpoint_unauthorized(self):
        """Test Flask REST status endpoint without API Key returns 401."""
        client = app.test_client()
        response = client.get("/status")
        self.assertEqual(response.status_code, 401)

    def test_flask_detect_malformed_payload(self):
        """Test that /api/detect with missing parameter returns 400 Bad Request."""
        client = app.test_client()
        response = client.post("/api/detect", json={}, headers={"X-API-Key": "test-key"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_flask_register_malformed_payload(self):
        """Test that /api/register with invalid data types returns 400 Bad Request."""
        client = app.test_client()
        payload = {
            "label_id": "abc",
            "username": "Test User",
            "frame": "dummy"
        }
        response = client.post("/api/register", json=payload, headers={"X-API-Key": "test-key"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_flask_delete_malformed_payload(self):
        """Test that /api/delete with missing parameters returns 400 Bad Request."""
        client = app.test_client()
        response = client.post("/api/delete", json={}, headers={"X-API-Key": "test-key"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    @unittest.mock.patch('app.detector.detect')
    @unittest.mock.patch('app.recognizer.extract_embedding')
    def test_flask_duplicate_registration_prevention(self, mock_extract, mock_detect):
        """Test that registering the same face under a different ID is detected and blocked."""
        # 1. Register User 101 with a specific embedding
        storage.register_user(101, "User A")
        dummy_embedding = np.array([1.0, 0.0, 0.0])
        storage.save_embedding(101, dummy_embedding)
        
        # Sync memory cache
        import app as app_mod
        app_mod.embedding_cache.invalidate()
        
        # 2. Mock detector and recognizer to return a face and the SAME embedding for the new request
        mock_detect.return_value = np.array([[10, 10, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.99]])
        mock_extract.return_value = dummy_embedding
        
        client = app.test_client()
        payload = {
            "label_id": 102, # Different ID
            "username": "User B",
            "frame": "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        }
        
        # 3. Post registration request
        response = client.post("/api/register", json=payload, headers={"X-API-Key": "test-key"})
        self.assertEqual(response.status_code, 200)
        
        json_data = response.get_json()
        self.assertEqual(json_data["status"], "duplicate_detected")
        self.assertEqual(json_data["matched_id"], 101)
        self.assertEqual(json_data["matched_name"], "User A")

    @unittest.mock.patch('app.detector.detect')
    def test_flask_register_poor_quality(self, mock_detect):
        """Test that registering a face with low detection score is rejected as poor quality."""
        # Mock detector to return a face with a LOW confidence score (e.g. 0.65) at index 14
        mock_detect.return_value = np.array([[10, 10, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.65]])
        
        client = app.test_client()
        payload = {
            "label_id": 777,
            "username": "Low Quality User",
            "frame": "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        }
        
        response = client.post("/api/register", json=payload, headers={"X-API-Key": "test-key"})
        self.assertEqual(response.status_code, 200)
        
        json_data = response.get_json()
        self.assertEqual(json_data["status"], "poor_quality")
        self.assertIn("Poor face quality detected", json_data["message"])

    @unittest.mock.patch('app.detector.detect')
    def test_flask_register_pose_mismatch(self, mock_detect):
        """Test that registering a face in the wrong pose is rejected with pose_mismatch."""
        # Mock detector: nose is centered (ratio=1.0, pose=front), score is 0.99
        mock_detect.return_value = np.array([[10, 10, 50, 50, 100.0, 0, 200.0, 0, 150.0, 0, 0, 0, 0, 0, 0.99]])
        
        client = app.test_client()
        # Request target_pose = left, but face is front
        payload = {
            "label_id": 666,
            "username": "Pose User",
            "frame": "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
            "target_pose": "left"
        }
        
        response = client.post("/api/register", json=payload, headers={"X-API-Key": "test-key"})
        self.assertEqual(response.status_code, 200)
        
        json_data = response.get_json()
        self.assertEqual(json_data["status"], "pose_mismatch")
        self.assertIn("Please turn your head slightly to the left", json_data["message"])

if __name__ == "__main__":
    unittest.main()


class TestMultipartUpload(unittest.TestCase):
    """Tests for multipart/form-data image upload support."""

    @classmethod
    def setUpClass(cls):
        app.config["FACE_API_KEY"] = "test-key"
        cls.orig_db_path = storage.DB_PATH
        storage.DB_PATH = "data/test_multipart.db"
        storage.init_db()

    @classmethod
    def tearDownClass(cls):
        storage.DB_PATH = cls.orig_db_path
        for f in ["data/test_multipart.db", "data/test_multipart.db-wal", "data/test_multipart.db-shm"]:
            if os.path.exists(f):
                os.remove(f)

    def test_multipart_detect_endpoint(self):
        """Test that /api/detect accepts multipart/form-data with an image file."""
        client = app.test_client()
        # Create a minimal valid JPEG-like image using OpenCV
        import cv2
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)
        image_bytes = buf.tobytes()

        response = client.post(
            "/api/detect",
            data={"image": (io.BytesIO(image_bytes), "test.jpg")},
            content_type="multipart/form-data",
            headers={"X-API-Key": "test-key"}
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn("faces", json_data)

    def test_multipart_detect_missing_image(self):
        """Test that /api/detect returns 400 when multipart upload has no image field."""
        client = app.test_client()
        response = client.post(
            "/api/detect",
            data={"other_field": "value"},
            content_type="multipart/form-data",
            headers={"X-API-Key": "test-key"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    @unittest.mock.patch('app.detector.detect')
    @unittest.mock.patch('app.recognizer.extract_embedding')
    def test_multipart_register_endpoint(self, mock_extract, mock_detect):
        """Test that /api/register accepts multipart/form-data with image + form fields."""
        import cv2
        # Mock detector and recognizer
        mock_detect.return_value = np.array([[10, 10, 50, 50, 100.0, 0, 200.0, 0, 150.0, 0, 0, 0, 0, 0, 0.99]])
        mock_extract.return_value = np.random.rand(1, 128).astype(np.float32)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)
        image_bytes = buf.tobytes()

        client = app.test_client()
        response = client.post(
            "/api/register",
            data={
                "image": (io.BytesIO(image_bytes), "test.jpg"),
                "label_id": "999",
                "username": "Multipart User",
                "target_pose": "front"
            },
            content_type="multipart/form-data",
            headers={"X-API-Key": "test-key"}
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data["status"], "success")


class TestEmbeddingCache(unittest.TestCase):
    """Tests for the version-based embedding cache invalidation system."""

    @classmethod
    def setUpClass(cls):
        cls.orig_db_path = storage.DB_PATH
        storage.DB_PATH = "data/test_cache.db"
        storage.init_db()

    @classmethod
    def tearDownClass(cls):
        storage.DB_PATH = cls.orig_db_path
        for f in ["data/test_cache.db", "data/test_cache.db-wal", "data/test_cache.db-shm"]:
            if os.path.exists(f):
                os.remove(f)

    def setUp(self):
        conn = sqlite3.connect(storage.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM profiles")
        cursor.execute("DELETE FROM embeddings")
        cursor.execute("UPDATE cache_version SET version = 0 WHERE id = 1")
        conn.commit()
        conn.close()

    def test_cache_version_increments_on_save(self):
        """Test that saving an embedding increments the cache version counter."""
        v_before = storage.get_cache_version()
        storage.register_user(1, "Test")
        storage.save_embedding(1, np.random.rand(1, 128).astype(np.float32))
        v_after = storage.get_cache_version()
        self.assertEqual(v_after, v_before + 1)

    def test_cache_version_increments_on_delete(self):
        """Test that deleting a user increments the cache version counter."""
        storage.register_user(2, "Test2")
        storage.save_embedding(2, np.random.rand(1, 128).astype(np.float32))
        v_before = storage.get_cache_version()
        storage.delete_user(2)
        v_after = storage.get_cache_version()
        self.assertEqual(v_after, v_before + 1)

    def test_cache_auto_reloads_on_version_change(self):
        """Test that EmbeddingCache automatically reloads when version changes."""
        cache = EmbeddingCache()
        
        # Initially empty
        self.assertEqual(len(cache.get_embeddings()), 0)
        
        # Save an embedding (increments version)
        storage.register_user(3, "Test3")
        storage.save_embedding(3, np.random.rand(1, 128).astype(np.float32))
        
        # Cache should auto-detect version change and reload
        embeddings = cache.get_embeddings()
        self.assertIn(3, embeddings)

    def test_cache_no_reload_when_unchanged(self):
        """Test that EmbeddingCache does NOT reload from DB when version hasn't changed."""
        cache = EmbeddingCache()
        
        # First access loads from DB
        result1 = cache.get_embeddings()
        
        # Second access should return same object (no reload) since version unchanged
        result2 = cache.get_embeddings()
        self.assertIs(result1, result2)  # Same object reference = no reload happened
