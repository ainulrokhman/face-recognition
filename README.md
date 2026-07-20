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
- **Access Authentication**: Bearer-token authentication using the `X-API-Key` request header.
- **CORS Restriction**: Configurable allowed origin header mapping.

### 3. Scalability
- **RAM Caching**: Embeddings are loaded from SQLite into RAM at startup, allowing ultra-fast vector math comparisons.
- **Zero Retraining**: Adding a new user is instant and requires no model training (unlike LBPH/Eigenfaces).

---

## Project Structure

- `app.py`: Main Flask REST API server and security hooks.
- `detector.py`: YuNet CNN detector wrapper and landmark locator.
- `recognizer.py`: SFace ONNX feature extractor and Cosine similarity math.
- `storage.py`: SQLite Database CRUD actions and image Base64 decoder.
- `test_backend.py`: Automated test cases (security, boundaries, math, CRUD).
- `docs.html`: Production-grade API documentation portal (Swagger-style).
- `index.html`: Client-side webcam interactive sandbox.
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

### 2. Launch the Application
Build and run the API container in detached mode:
```bash
docker compose up -d --build
```
* Access the webcam playground: `http://localhost:5000/`
* Access the interactive API documentation: `http://localhost:5000/docs`

### 3. Run Test Suite
Execute the automated tests in the isolated Docker container environment:
```bash
docker compose run --entrypoint "python -m unittest test_backend.py" face-recognition-api
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
