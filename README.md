# Face Recognition Backend API (OpenCV DNN + SFace)

A high-performance, containerized, and memory-optimized **Face Recognition REST API** built with Flask, OpenCV's deep learning module, and SQLite.

Designed for production environments to handle student attendance systems with thousands of records, using state-of-the-art CNN models:
- **YuNet CNN**: Dynamic input face detection and landmark extraction.
- **SFace**: 128-dimensional biometric embedding extraction.
- **SQLite3**: Relational persistence of student profiles and serialized vectors.
- **NumPy**: Vectorized Cosine Similarity matching (<0.5ms search time).

---

## Architectural & Production Features

### 1. Biometric Pipeline
- **Quality Filter**: Rejects blurry, poorly lit, or out-of-focus frames during registration (YuNet confidence threshold $\ge 80\%$).
- **Pose Verification**: Real-time left, right, and front head pose validation using landmarks ratios to build a complete 3D profile.
- **Deduplication**: Biometric check preventing registering the same face under multiple student IDs.

### 2. Deployment Security
- **Container Hardening**: Execution under a non-root system user (`appuser` with UID/GID 1000) inside the Docker container.
- **HTTPS/SSL**: Nginx reverse proxy with SSL termination, HSTS, and security headers (`X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`).
- **Access Authentication**: Bearer-token authentication using the `X-API-Key` request header.
- **CORS Restriction**: Configurable allowed origin header mapping.

### 3. Scalability
- **Smart Cache Sync**: In-memory embedding cache with SQLite WAL + version-based invalidation. Each request checks a single integer (~5μs overhead) — auto-reloads only when data changes. Safe for multi-worker deployment (Gunicorn/uWSGI).
- **Dual Input Support**: API endpoints accept both `multipart/form-data` (binary file upload, ~33% less bandwidth) and `application/json` (Base64 string) — auto-detected from `Content-Type`.
- **Zero Retraining**: Adding a new user is instant and requires no model training (unlike LBPH/Eigenfaces).

---

## Project Structure

- `src/`: Core Python backend source code.
  - `app.py`: Main Flask REST API server and security hooks.
  - `detector.py`: YuNet CNN detector wrapper, landmark locator, and pose classifier.
  - `recognizer.py`: SFace ONNX feature extractor and Cosine similarity math.
  - `storage.py`: SQLite Database CRUD actions, WAL mode, and cache version management.
  - `cache.py`: Version-based embedding cache with auto-invalidation for multi-worker sync.
  - `image_utils.py`: Dual-input image parser (multipart file upload + JSON Base64).
  - `interfaces.py`: Clean interfaces separating responsibilities (SOLID).
- `static/`: Static assets served by Flask.
  - `index.html`: Client-side webcam interactive sandbox.
  - `docs.html`: Production-grade API documentation portal (Swagger-style).
- `tests/`: Automated unit tests.
  - `test_backend.py`: Automated test cases (security, boundaries, math, CRUD, multipart, cache).
- `nginx/`: Nginx reverse proxy configuration.
  - `nginx.conf`: SSL termination, security headers, and proxy settings.
  - `generate-cert.sh`: Self-signed certificate generator for development.
- `docker-compose.yml` & `Dockerfile`: Package orchestration for deployments.

---

## Quick Start

### 1. Configure Environments
Copy the environment template file:
```bash
cp .env.example .env
```
Open `.env` and fill in your configurations:
- `FACE_API_KEY`: API authentication key.
- `ALLOWED_ORIGIN`: Allowed domain origins for CORS (e.g. `*` for dev, or `https://absensi.sekolah.sch.id` for prod).

### 2. Generate SSL Certificate
Generate a self-signed certificate for HTTPS (development):
```bash
bash nginx/generate-cert.sh
```
> For production, replace the generated files in `nginx/ssl/` with certificates from a trusted CA (Let's Encrypt, Cloudflare, etc.).

### 3. Launch the Application
Build and run the API containers in detached mode:
```bash
docker compose up -d --build
```

Access points:
| URL | Description |
|-----|-------------|
| `http://localhost:5000/` | Webcam playground (direct Flask, development) |
| `http://localhost:5000/docs` | API documentation (direct Flask, development) |
| `https://localhost:8443/` | Webcam playground (via Nginx HTTPS) |
| `https://localhost:8443/docs` | API documentation (via Nginx HTTPS) |
| `http://localhost:8080/` | Auto-redirects to HTTPS |

### 4. Run Test Suite
Execute the automated tests in the isolated Docker container environment:
```bash
docker compose run --entrypoint "python -m unittest tests/test_backend.py" face-recognition-api
```

---

## API Input Formats

All image-accepting endpoints (`/api/detect`, `/api/register`, `/api/recognize`) support two input formats:

### JSON Base64 (Backward Compatible)
```bash
curl -X POST http://localhost:5000/api/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"frame": "data:image/jpeg;base64,/9j/4..."}'
```

### Multipart File Upload (Saves ~33% Bandwidth)
```bash
curl -X POST http://localhost:5000/api/detect \
  -H "X-API-Key: YOUR_KEY" \
  -F "image=@photo.jpg"
```

For `/api/register`, include additional form fields:
```bash
curl -X POST http://localhost:5000/api/register \
  -H "X-API-Key: YOUR_KEY" \
  -F "image=@photo.jpg" \
  -F "label_id=101" \
  -F "username=Budi" \
  -F "target_pose=front"
```

---

## Server Specifications Recommendation

Since OpenCV DNN performs neural network inferences on CPU, performance depends heavily on the processor's clock speed and modern instruction sets.

### 1. Key Hardware Requirements
* **CPU**: Modern processor (Intel Xeon, AMD EPYC, Intel Core, AMD Ryzen) supporting **AVX2 or AVX-512** instructions. *AVX2 acceleration is critical to speed up YuNet and SFace calculations by 4x-5x on CPU.*
* **RAM**: Very low memory footprint (~100MB for model runtime, ~12MB for 5,000 cached student vectors). 
* **Storage**: Fast SSD is highly recommended to prevent write latency during SQLite inserts.

### 2. Sizing Guidelines

| Scale (Students) | CPU (Cores) | RAM (vCPU) | Disk (SSD) | Typical VPS Instance |
|------------------|-------------|------------|------------|----------------------|
| **Small (<1,000)** | 1 Core (AVX2) | 1 GB | 10 GB | AWS `t3.micro` / DO $6 Droplet |
| **Medium (1,000 - 5,000)** | 2 Cores (AVX2) | 2 GB | 20 GB | AWS `t3.medium` / DO $12 Droplet |
| **Large (>5,000)** | 4 Cores (AVX2) | 4 GB | 50 GB | AWS `c5.large` / Compute Optimized |
