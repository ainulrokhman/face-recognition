# Face Recognition API (Production Ready)

A high-performance, containerized, and secure **Face Recognition REST API** built with Flask, OpenCV's DNN module, SQLite, and Nginx. 

This service implements a state-of-the-art biometric pipeline utilizing:
- **YuNet CNN**: Real-time face detection, landmark localization, and head pose estimation.
- **SFace**: 128-dimensional biometric embedding extraction.
- **SQLite3 (WAL Mode)**: Relational storage of student profiles and serialized vectors.
- **In-Memory Cache (Version-controlled)**: Super-fast vector similarity matching (<0.5ms) across multiple WSGI workers.

---

## 🏗️ Production Architecture

```
                       [ HTTPS (8443) ]          [ HTTP (8080) ]
                              │                        │
                              ▼                        ▼
                      ┌───────────────┐        ┌───────────────┐
                      │  Nginx Proxy  │◄───────│  Redirect 301 │
                      └───────┬───────┘        └───────────────┘
                              │ (Internal HTTP)
                              ▼
                      ┌───────────────┐
                      │ Gunicorn WSGI │ (4 workers, direct access on 5000)
                      └───────┬───────┘
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
     ┌───────────────┐┌───────────────┐┌───────────────┐
     │  YuNet (CNN)  ││ SFace (ONNX)  ││ SQLite (WAL)  │
     │  Face Detect  ││ Feature Extr  ││  Vector DB   │
     └───────────────┘└───────────────┘└───────────────┘
```

### Key Production Enhancements
1. **High Concurrency**: Powered by **Gunicorn** WSGI server with 4 worker threads for reliable concurrent processing.
2. **Reverse Proxy & SSL**: Protected by an **Nginx Alpine** container providing SSL/TLS termination, HTTP→HTTPS redirects, and strict security headers (`HSTS`, `X-Frame-Options`, `X-Content-Type-Options`).
3. **Smart Cache Synchronization**: Uses a custom **SQLite WAL + Version-based counter** for cross-worker memory invalidation. Each worker checks a single database integer (~5μs overhead) before reusing its local memory cache, ensuring data consistency without the overhead of a Redis container.
4. **Dual Input Methods**: Endpoints automatically parse both `multipart/form-data` (binary file upload, ~33% bandwidth saving) and `application/json` (Base64 string).
5. **Baked Weights**: YuNet and SFace model weights are baked into the Docker image, preventing internet-dependent startup failures in isolated networks (VPC).

---

## 📁 Directory Structure

```
├── data/                  # SQLite DB and ONNX models (persistent host bind-mount)
├── dataset/               # Persistent image uploads (optional host bind-mount)
├── nginx/
│   ├── nginx.conf         # SSL & proxy routing configuration
│   └── generate-cert.sh   # Utility script to generate self-signed dev certificates
├── src/
│   ├── app.py             # Main Flask application and security hooks
│   ├── cache.py           # In-memory version-based embedding cache
│   ├── detector.py        # YuNet face detector and landmark analyzer
│   ├── recognizer.py      # SFace embedding extractor and cosine similarity matcher
│   ├── storage.py         # SQLite connection manager and CRUD operations
│   └── image_utils.py     # Dual-input image parser (multipart & base64 JSON)
├── tests/
│   └── test_backend.py    # Suite of 18 integration and unit tests
├── Dockerfile             # Production multi-stage Flask & Gunicorn image
└── docker-compose.yml     # Multi-container stack (Flask app + Nginx proxy)
```

---

## 🚀 Quick Start

### 1. Environment Configurations
Copy the template configuration file:
```bash
cp .env.example .env
```
Open `.env` and configure:
- `FACE_API_KEY`: Strong security key for API access.
- `ALLOWED_ORIGIN`: Allowed domain origins for CORS restrictions (e.g. `*` or your web portal domain).

### 2. Generate Development SSL Certificates
Before launching the stack, generate a self-signed SSL certificate for development:
```bash
bash nginx/generate-cert.sh
```
> **Note**: For production environments, replace the files in `nginx/ssl/` with real certificates issued by a trusted CA (e.g., Let's Encrypt, Cloudflare).

### 3. Start the Stack
Build and launch the containers in detached mode:
```bash
docker compose up -d --build
```

### 4. Running Tests
Run the 18 automated backend test cases within the isolated Docker environment:
```bash
docker compose run --entrypoint "python -m unittest tests/test_backend.py" face-recognition-api
```

---

## 🌐 Endpoint Access Points

| URL | Protocol | Access | Environment |
|---|---|---|---|
| `https://localhost:8443/` | HTTPS | Webcam Playground | Production/Staging |
| `https://localhost:8443/docs` | HTTPS | Interactive API Docs | Production/Staging |
| `http://localhost:8080/` | HTTP | Redirects to HTTPS `8443` | Production/Staging |
| `http://localhost:5000/` | HTTP | Direct Gunicorn Access | Development/Debugging |

---

## ✉️ API Specifications

All requests (except `OPTIONS` and UI pages) must include the `X-API-Key` header if `FACE_API_KEY` is set in the environment.

### 1. Face Detection (`POST /api/detect`)
Detects faces in an image and returns bounding box coordinates and confidences.

*   **Option A: Multipart Upload (Recommended)**
    ```bash
    curl -X POST https://localhost:8443/api/detect \
      -H "X-API-Key: YOUR_API_KEY" \
      -F "image=@frame.jpg" -k
    ```
*   **Option B: Base64 JSON (Legacy)**
    ```bash
    curl -X POST https://localhost:8443/api/detect \
      -H "Content-Type: application/json" \
      -H "X-API-Key: YOUR_API_KEY" \
      -d '{"frame": "data:image/jpeg;base64,/9j/4..."}' -k
    ```
*   **Success Response (200 OK)**:
    ```json
    {
      "faces": [
        { "x": 120, "y": 80, "w": 150, "h": 150, "confidence": 0.99 }
      ]
    }
    ```

### 2. Register Face (`POST /api/register`)
Registers a face embedding under a unique label ID and name. Filters for biometric quality ($\ge 80\%$) and head pose accuracy.

*   **Option A: Multipart Upload (Recommended)**
    ```bash
    curl -X POST https://localhost:8443/api/register \
      -H "X-API-Key: YOUR_API_KEY" \
      -F "image=@frame.jpg" \
      -F "label_id=101" \
      -F "username=John Doe" \
      -F "target_pose=front" -k
    ```
*   **Option B: Base64 JSON (Legacy)**
    ```bash
    curl -X POST https://localhost:8443/api/register \
      -H "Content-Type: application/json" \
      -H "X-API-Key: YOUR_API_KEY" \
      -d '{
        "frame": "data:image/jpeg;base64,...",
        "label_id": 101,
        "username": "John Doe",
        "target_pose": "front"
      }' -k
    ```

### 3. Recognize Face (`POST /api/recognize`)
Extracts a biometric vector from the image and compares it against all registered vectors using Cosine Similarity.

*   **Option A: Multipart Upload (Recommended)**
    ```bash
    curl -X POST https://localhost:8443/api/recognize \
      -H "X-API-Key: YOUR_API_KEY" \
      -F "image=@frame.jpg" -k
    ```
*   **Option B: Base64 JSON (Legacy)**
    ```bash
    curl -X POST https://localhost:8443/api/recognize \
      -H "Content-Type: application/json" \
      -H "X-API-Key: YOUR_API_KEY" \
      -d '{"frame": "data:image/jpeg;base64,..."}' -k
    ```
*   **Success Response (200 OK)**:
    ```json
    {
      "faces": [
        {
          "x": 120, "y": 80, "w": 150, "h": 150,
          "label_id": 101,
          "name": "John Doe",
          "confidence": 0.892
        }
      ]
    }
    ```

---

## 💻 Hardware & Sizing Guide

Because OpenCV DNN runs inferences on CPU threads, speed is dependent on processor speed and modern instruction sets.

*   **AVX2/AVX-512**: Highly recommended. Accelerates YuNet and SFace calculations by 4x on CPU.
*   **RAM**: Footprint is extremely low (~100MB for engine, ~12MB for 5,000 cached vectors).

| Scale (Students) | CPU (Cores) | RAM (vCPU) | Storage (SSD) | Suggested VPS Instance |
|---|---|---|---|---|
| **Small (<1,000)** | 1 Core (AVX2) | 1 GB | 10 GB | AWS `t3.micro` / DO $6 Droplet |
| **Medium (1,000 - 5,000)** | 2 Cores (AVX2) | 2 GB | 20 GB | AWS `t3.medium` / DO $12 Droplet |
| **Large (>5,000)** | 4 Cores (AVX2) | 4 GB | 50 GB | AWS `c5.large` / Compute Optimized |
