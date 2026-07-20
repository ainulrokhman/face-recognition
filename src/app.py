import os
import gc
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import storage
from detector import YuNetDetector
from recognizer import SFRecognizer

# Initialize Flask application
app = Flask(__name__, static_folder="static")

# Initialize SQLite database schema
storage.init_db()

# Cache embeddings in RAM at startup for sub-millisecond matching
cached_embeddings = storage.load_embeddings()

# Dependency Injection of high-accuracy models
detector = YuNetDetector()
recognizer = SFRecognizer()

# Load security configurations into Flask app config
app.config["FACE_API_KEY"] = os.environ.get("FACE_API_KEY")
app.config["ALLOWED_ORIGIN"] = os.environ.get("ALLOWED_ORIGIN", "*")

if not app.config["FACE_API_KEY"]:
    print("WARNING: FACE_API_KEY environment variable is not set. API Key authentication is disabled.")

# CORS filter to allow requests from client-side interfaces (like CodeIgniter 3)
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", app.config["ALLOWED_ORIGIN"])
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization,X-API-Key")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# API Key Validation Hook
@app.before_request
def require_api_key():
    # Bypass verification for preflight options and frontend pages
    if request.method == "OPTIONS" or request.path in ["/", "/docs"]:
        return
        
    api_key = app.config.get("FACE_API_KEY")
    if api_key:
        client_key = request.headers.get("X-API-Key")
        if not client_key or client_key != api_key:
            return jsonify({"error": "Unauthorized: Invalid or missing X-API-Key"}), 401

# Globally handle preflight HTTP requests
@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def handle_options(path):
    return "", 200

@app.route("/")
def index():
    """Serves the test playground HTML file."""
    return send_from_directory(app.static_folder, "index.html")

@app.route("/docs")
def docs():
    """Serves the production API documentation portal."""
    return send_from_directory(app.static_folder, "docs.html")

@app.route("/status", methods=["GET"])
def get_status():
    """Returns the API health status and loaded model details."""
    profiles = storage.load_profiles()
    return jsonify({
        "status": "online",
        "model_trained": recognizer.is_trained() and len(cached_embeddings) > 0,
        "registered_users": profiles
    })

@app.route("/api/detect", methods=["POST"])
def api_detect():
    """Detects faces in a posted base64 frame using CNN YuNet."""
    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "Missing 'frame' parameter"}), 400
        
    img = storage.decode_base64_image(data["frame"])
    if img is None:
        return jsonify({"error": "Failed to decode base64 frame"}), 400
        
    # YuNet operates directly on BGR frames
    faces = detector.detect(img)
    
    result = []
    if faces is not None:
        for face in faces:
            # First 4 parameters are x, y, w, h
            result.append({
                "x": int(face[0]),
                "y": int(face[1]),
                "w": int(face[2]),
                "h": int(face[3]),
                "confidence": round(float(face[14]), 2)
            })
            
    return jsonify({"faces": result})


@app.route("/api/register", methods=["POST"])
def api_register():
    """Registers a face vector under a specific label ID, verifying pose and quality."""
    global cached_embeddings
    
    data = request.get_json()
    if not data or "frame" not in data or "label_id" not in data or "username" not in data:
        return jsonify({"error": "Missing frame, label_id, or username parameters"}), 400
        
    try:
        label_id = int(data["label_id"])
    except ValueError:
        return jsonify({"error": "label_id must be an integer"}), 400
        
    username = data["username"].strip()
    if not username:
        return jsonify({"error": "username cannot be empty"}), 400
        
    target_pose = data.get("target_pose", "front")
    
    img = storage.decode_base64_image(data["frame"])
    if img is None:
        return jsonify({"error": "Failed to decode base64 frame"}), 400
        
    # Run face detection
    faces = detector.detect(img)
    
    if faces is None or len(faces) == 0:
        return jsonify({"status": "no_face_detected", "message": "No face detected in this frame."}), 200
        
    if len(faces) > 1:
        return jsonify({
            "status": "multiple_faces",
            "message": "Multiple faces detected! Ensure only one person is in front of the camera."
        }), 200
        
    # Crop and align the first detected face, then extract the 128-dimensional embedding
    face_detection = faces[0]
    
    # 1. Biometric Quality Filter: Ensure face detection confidence is >= 80% (0.80)
    detection_score = float(face_detection[14])
    if detection_score < 0.80:
        return jsonify({
            "status": "poor_quality",
            "message": f"Poor face quality detected ({round(detection_score * 100, 1)}%). Please adjust lighting or position."
        }), 200
        
    # 2. Biometric Pose Verification: Ensure client face pose matches instructions
    detected_pose = detector.check_pose(face_detection)
    if target_pose != "any" and detected_pose != target_pose:
        pose_instructions = {
            "front": "Please look straight ahead.",
            "left": "Please turn your head slightly to the left.",
            "right": "Please turn your head slightly to the right."
        }
        return jsonify({
            "status": "pose_mismatch",
            "message": pose_instructions.get(target_pose, "Please adjust your head position.")
        }), 200
        
    # Extract the 128-dimensional embedding vector
    embedding = recognizer.extract_embedding(img, face_detection)
    if embedding is None:
        return jsonify({"status": "error", "message": "Failed to extract face vector."}), 500
        
    # Duplicate face prevention check
    if len(cached_embeddings) > 0:
        # Match using a strict threshold (0.45) to ensure it's a confident match
        matched_id, score = recognizer.match(embedding, cached_embeddings, threshold=0.45)
        # Block only if the face matches a different label_id
        if matched_id != -1 and matched_id != label_id:
            profiles = storage.load_profiles()
            matched_name = profiles.get(matched_id, {}).get("name", "Unknown")
            return jsonify({
                "status": "duplicate_detected",
                "message": f"Face is already registered under name '{matched_name}' (ID: {matched_id})",
                "matched_id": matched_id,
                "matched_name": matched_name
            }), 200
        
    # Save embedding and profile metadata to SQLite database
    storage.save_embedding(label_id, embedding)
    storage.register_user(label_id, username)
    
    # Reload embeddings cache in memory
    cached_embeddings = storage.load_embeddings()
    
    return jsonify({
        "status": "success",
        "message": f"Saved face vector for '{username}'",
        "box": {
            "x": int(face_detection[0]),
            "y": int(face_detection[1]),
            "w": int(face_detection[2]),
            "h": int(face_detection[3])
        }
    })

@app.route("/api/train", methods=["POST"])
def api_train():
    """Keeps backward compatibility with the frontend. Simply synchronizes database and runs GC."""
    global cached_embeddings
    cached_embeddings = storage.load_embeddings()
    gc.collect()
    return jsonify({"status": "success", "message": "Database synchronized and memory optimized."})

@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """Runs high-accuracy SFace recognition on a posted BGR base64 frame."""
    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "Missing 'frame' parameter"}), 400
        
    img = storage.decode_base64_image(data["frame"])
    if img is None:
        return jsonify({"error": "Failed to decode base64 frame"}), 400
        
    # Detect faces via YuNet CNN
    faces = detector.detect(img)
    
    profiles = storage.load_profiles()
    results = []
    
    if faces is not None:
        for face_detection in faces:
            x, y, w, h = int(face_detection[0]), int(face_detection[1]), int(face_detection[2]), int(face_detection[3])
            
            label_id = -1
            confidence = 0.0
            name = "Unknown"
            
            # Extract face vector embedding
            embedding = recognizer.extract_embedding(img, face_detection)
            if embedding is not None and len(cached_embeddings) > 0:
                # Query nearest-neighbor matching via Cosine Similarity
                # OpenCV SFace Cosine match threshold is ~0.363
                matched_id, score = recognizer.match(embedding, cached_embeddings, threshold=0.363)
                if matched_id != -1:
                    label_id = matched_id
                    confidence = score
                    if label_id in profiles:
                        name = profiles[label_id]["name"]
                else:
                    confidence = score # Record best score even if below threshold
                    
            results.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "label_id": int(label_id),
                "name": name,
                "confidence": round(float(confidence), 3) # Float representing cosine similarity (-1 to 1)
            })
            
    return jsonify({"faces": results})

@app.route("/api/delete", methods=["POST"])
def api_delete():
    """Deletes a user profile and updates cache."""
    global cached_embeddings
    
    data = request.get_json()
    if not data or "label_id" not in data:
        return jsonify({"error": "Missing label_id parameter"}), 400
        
    try:
        label_id = int(data["label_id"])
    except ValueError:
        return jsonify({"error": "label_id must be an integer"}), 400
        
    success = storage.delete_user(label_id)
    if success:
        cached_embeddings = storage.load_embeddings()
        return jsonify({"status": "success", "message": f"Deleted user ID {label_id}"})
    return jsonify({"status": "error", "message": "Failed to delete user."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
