import cv2
import os
import urllib.request
import numpy as np
from interfaces import BaseFaceDetector

class YuNetDetector(BaseFaceDetector):
    """Concrete implementation of BaseFaceDetector using OpenCV's YuNet DNN model."""
    
    MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    
    def __init__(self, model_name: str = "face_detection_yunet_2023mar.onnx", data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.model_path = os.path.join(self.data_dir, model_name)
        
        # Download YuNet ONNX model from official OpenCV Zoo repository if not present
        if not os.path.exists(self.model_path):
            print(f"YuNet model not found. Downloading from OpenCV Zoo: {self.MODEL_URL}...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
                print("Download complete.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download YuNet model from {self.MODEL_URL}: {e}. "
                    f"Please place {model_name} in the '{self.data_dir}/' folder manually."
                )
                
        # Initialize YuNet Face Detector with initial default input size (320x320)
        # Input size is updated dynamically depending on the frames processed.
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.7,  # Balanced threshold for high confidence
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        self._current_input_size = (320, 320)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return None
            
        h, w = frame.shape[:2]
        
        # Dynamically update detector input size if frame shape changes
        if (w, h) != self._current_input_size:
            self.detector.setInputSize((w, h))
            self._current_input_size = (w, h)
            
        retval, faces = self.detector.detect(frame)
        
        # Returns numpy array of shape (N, 15) if faces are found, else None
        if retval and faces is not None:
            return faces
        return None

    def check_pose(self, face_detection: np.ndarray) -> str:
        """Classifies the face pose as 'front', 'left', or 'right' based on YuNet eye/nose landmarks."""
        if face_detection is None or face_detection.size < 15:
            return "front"
            
        x_re = float(face_detection[4])
        x_le = float(face_detection[6])
        x_nt = float(face_detection[8])
        
        dist_re_to_nt = abs(x_nt - x_re)
        dist_le_to_nt = abs(x_nt - x_le)
        
        if dist_re_to_nt == 0: dist_re_to_nt = 1.0
        if dist_le_to_nt == 0: dist_le_to_nt = 1.0
        
        ratio = dist_re_to_nt / dist_le_to_nt
        
        if ratio < 0.6:
            return "left"
        elif ratio > 1.6:
            return "right"
        else:
            return "front"
