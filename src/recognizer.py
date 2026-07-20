import cv2
import os
import urllib.request
import numpy as np
from typing import Tuple, Dict, List
from interfaces import BaseFaceRecognizer

class SFRecognizer(BaseFaceRecognizer):
    """Concrete implementation of BaseFaceRecognizer using OpenCV SFace ONNX model."""
    
    MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    
    def __init__(self, model_name: str = "face_recognition_sface_2021dec.onnx", data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.model_path = os.path.join(self.data_dir, model_name)
        
        # Download SFace ONNX model from official OpenCV Zoo repository if not present
        if not os.path.exists(self.model_path):
            print(f"SFace model not found. Downloading from OpenCV Zoo: {self.MODEL_URL}...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
                print("Download complete.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download SFace model from {self.MODEL_URL}: {e}. "
                    f"Please place {model_name} in the '{self.data_dir}/' folder manually."
                )
                
        # Initialize OpenCV FaceRecognizerSF
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.model_path,
            config=""
        )
        self._model_loaded = os.path.exists(self.model_path)

    def extract_embedding(self, frame: np.ndarray, face_detection: np.ndarray) -> np.ndarray:
        if frame is None or face_detection is None:
            return None
            
        try:
            # 1. Align and crop the face using facial landmarks (returns 112x112 aligned BGR image)
            aligned_face = self.recognizer.alignCrop(frame, face_detection)
            
            # 2. Extract the 128-dimensional embedding feature vector
            embedding = self.recognizer.feature(aligned_face)
            return embedding
        except Exception as e:
            print(f"Failed to extract face embedding: {e}")
            return None

    def match(self, query_embedding: np.ndarray, db_embeddings: Dict[int, List[np.ndarray]], threshold: float = 0.363) -> Tuple[int, float]:
        """
        Compare query vector against database of student vectors using Cosine Similarity.
        OpenCV SFace cosine similarity threshold is typically around 0.363.
        
        Args:
            query_embedding: extracted vector of shape (1, 128) or (128,).
            db_embeddings: dictionary mapping label_id (int) -> list of np.ndarray vectors.
            threshold: minimum score to consider a match (0.363 is SFace standard).
        """
        if not db_embeddings or query_embedding is None:
            return -1, 0.0
            
        # Flatten and normalize query vector
        q_vec = query_embedding.flatten()
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return -1, 0.0
        q_vec = q_vec / q_norm
        
        best_label_id = -1
        best_score = -1.0
        
        # Iterative Cosine Similarity search over registered user embeddings
        for label_id, vectors in db_embeddings.items():
            for stored_embedding in vectors:
                s_vec = stored_embedding.flatten()
                s_norm = np.linalg.norm(s_vec)
                if s_norm == 0:
                    continue
                s_vec = s_vec / s_norm
                
                # Cosine similarity is the dot product of the normalized vectors
                similarity = float(np.dot(q_vec, s_vec))
                if similarity > best_score:
                    best_score = similarity
                    best_label_id = label_id
                    
        # Apply strict SFace threshold check
        if best_score >= threshold:
            return best_label_id, best_score
        return -1, best_score

    def is_trained(self) -> bool:
        """Returns True if the SFace model was successfully instantiated."""
        return self._model_loaded
