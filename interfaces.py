from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any

class BaseFaceDetector(ABC):
    """Abstract interface for modern face detection (e.g., YuNet DNN)."""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect faces in a BGR image frame.
        
        Args:
            frame: BGR image numpy array.
            
        Returns:
            A 2D numpy array of shape (N, 15) containing bounding box,
            landmarks, and confidence score for N faces, or None if no faces.
        """
        pass

class BaseFaceRecognizer(ABC):
    """Abstract interface for modern face embedding extraction and vector matching."""
    
    @abstractmethod
    def extract_embedding(self, frame: np.ndarray, face_detection: np.ndarray) -> np.ndarray:
        """
        Aligns the face based on landmarks and extracts a 128-dimensional embedding.
        
        Args:
            frame: Original BGR image frame.
            face_detection: 1D array of 15 elements representing a single face detection.
            
        Returns:
            A 128-dimensional numpy float array, or None if extraction fails.
        """
        pass

    @abstractmethod
    def match(self, query_embedding: np.ndarray, db_embeddings: Dict[int, List[np.ndarray]], threshold: float) -> Tuple[int, float]:
        """
        Compare query face embedding against a database of registered student embeddings.
        
        Args:
            query_embedding: 128-dimensional query vector.
            db_embeddings: Dictionary mapping student label IDs to lists of their saved embedding vectors.
            threshold: Minimum cosine similarity threshold to declare a match.
            
        Returns:
            A tuple of (matched_label_id, similarity_score). Returns (-1, 0.0) if no match.
        """
        pass
