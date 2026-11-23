# VehicleIQ: Multimodal Semantic Vehicle Retrieval Engine

VehicleIQ is a multimodal retrieval system designed for urban traffic monitoring and surveillance analytics. It enables real-time vehicle identification across large camera networks through both natural language queries (such as color, vehicle class, motion, and spatial attributes) and visual image queries (image-to-vehicle re-identification).

The system integrates dense representation learning using Google SigLIP2, sparse lexical retrieval via BM25Okapi, reciprocal rank fusion (RRF) for hybrid scoring, and decoupled S3-compatible cloud object storage for low-latency media delivery.

---

## Key Capabilities

- Dual-Encoder Hybrid Search: Combines dense vision-language embeddings with sparse keyword matching to eliminate vocabulary mismatch and capture fine-grained vehicle descriptions.
- Visual Re-Identification: Allows operators to upload a query image to retrieve matching vehicles across multi-camera galleries using cosine similarity over normalized vision features.
- Decoupled Cloud Architecture: Separates dense vector indexing and tabular metadata from raw image storage, serving high-resolution evidence crops directly via an S3-compatible CDN.
- Sub-15ms Query Latency: Optimized retrieval pipeline querying 2,320 indexed vehicles and over 17,000 crops with sub-linear response time.

---

## System Architecture

```
                                      +--------------------------+
                                      |   Client / Streamlit UI  |
                                      +-------------+------------+
                                                    |
                                    HTTP Requests (Text / Image)
                                                    |
                                                    v
                                      +-------------+------------+
                                      |     FastAPI Service      |
                                      +-------------+------------+
                                                    |
                    +-------------------------------+-------------------------------+
                    |                                                               |
             [ Text Query ]                                                  [ Image Query ]
                    |                                                               |
                    v                                                               v
         SigLIP2 Text Encoder                                            SigLIP2 Vision Encoder
          (768-dim L2 vector)                                             (768-dim L2 vector)
                    |                                                               |
         +----------+----------+                                                    |
         |                     |                                                    |
         v                     v                                                    v
    Dense Search         Sparse Search                                         Dense Search
  (Cosine Similarity)    (BM25Okapi Index)                                   (Cosine Similarity)
         |                     |                                                    |
         +----------+----------+                                                    |
                    |                                                               |
                    v                                                               |
          Reciprocal Rank Fusion                                                    |
                   (RRF)                                                            |
                    |                                                               |
                    +-------------------------------+-------------------------------+
                                                    |
                                                    v
                                      +-------------+------------+
                                      |  Metadata Enrichment     |
                                      |  (SQLite Database)       |
                                      +-------------+------------+
                                                    |
                                                    v
                                      +-------------+------------+
                                      |  Cloud Media Resolution  |
                                      |  (S3 / Supabase CDN)     |
                                      +--------------------------+
```

---

## Technical Details

### 1. Dense Semantic Retrieval (SigLIP2)
Natural language queries and image inputs are mapped to a joint 768-dimensional visual-semantic embedding space using `google/siglip2-base-patch16-224`. Unlike standard contrastive models trained with softmax loss, SigLIP utilizes a sigmoid loss function that improves zero-shot classification calibration and multi-label scene understanding. All embeddings are unit L2-normalized:

    similarity(q, d) = dot_product(q, d)

### 2. Sparse Lexical Retrieval (BM25Okapi)
To ensure exact attribute matching (such as specific vehicle body types, license numbers, or distinctive camera annotations), an inverted index is constructed across all ground-truth human descriptions using BM25Okapi:

    score(D, Q) = sum( IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * (|D| / avgdl))) )

Parameters configured: `k1 = 1.5`, `b = 0.75`.

### 3. Reciprocal Rank Fusion (RRF)
To merge dense and sparse rankings without requiring score normalization calibration across disparate scales, reciprocal rank fusion is applied:

    RRF_Score(d) = alpha * (1 / (k + rank_dense(d))) + (1 - alpha) * (1 / (k + rank_sparse(d)))

The smoothing constant `k` is set to 60. The parameter `alpha` balances semantic relevance versus keyword precision.

### 4. Decoupled Cloud Storage
Image crops and video representations are hosted on AWS S3-compatible cloud object storage (Supabase Storage). Local storage footprint is limited to lightweight metadata (1.9 MB SQLite) and vector index files (7.1 MB NumPy), enabling instant application deployment without transferring gigabytes of raw media files.

---

## Dataset & Benchmarks

The gallery is indexed using the CityFlow-NL benchmark dataset:
- Total Indexed Vehicles: 2,320 unique vehicle trajectories
- Representative Image Crops: 17,025 curated high-resolution crops
- Text Annotations: Multi-view natural language descriptions per vehicle
- Camera Coverage: Multi-camera surveillance network across multiple intersections

---

## Repository Structure

```
.
|-- .gitignore                  # Git ignore rules for media and environment files
|-- README.md                   # Technical documentation
|-- track1/                     # CityFlow-NL benchmark embeddings and metadata
|   |-- embeddings/             # Precomputed SigLIP2 vehicle embeddings
|   `-- metadata/               # Vehicle trajectory annotations and frame indices
|-- track2/                     # Multi-camera temporal tracking gallery and crops
|   `-- temporal_crops/         # Representative multi-angle crops per vehicle
|-- storage/
|   |-- db/                     # SQLite database storing vehicle attributes
|   `-- faiss/                  # Vector embeddings and serialized BM25 index
|-- vehicleiq/
|   |-- app/
|   |   |-- main.py             # FastAPI entrypoint and lifespan management
|   |   |-- config.py           # Centralized environment configuration
|   |   |-- routers/
|   |   |   |-- search.py       # Text and image search endpoints
|   |   |   `-- vehicles.py     # Vehicle metadata lookup routes
|   |   `-- services/
|   |       |-- embedding.py    # SigLIP2 inference service (text and image)
|   |       |-- retrieval.py    # Hybrid dual-encoder retrieval engine
|   |       |-- sparse.py       # Pure-Python BM25Okapi implementation
|   |       |-- database.py     # SQLite persistence and migration manager
|   |       `-- storage.py      # AWS S3 / Supabase CDN client
|   |-- frontend/
|   |   `-- streamlit_app.py    # Interactive search dashboard
|   `-- tests/
|       `-- test_components.py  # System verification test suite
`-- requirements.txt            # Python dependencies
```

---

## API Reference

### 1. Hybrid Text Search
- Endpoint: `POST /search`
- Request Payload:
```json
{
  "query": "red sedan turning left",
  "top_k": 10,
  "alpha": 0.5
}
```
- Response: Returns top matching vehicles with individual dense scores, sparse scores, fused RRF score, detected attributes, and public CDN thumbnail URLs.

### 2. Visual Re-Identification Search
- Endpoint: `POST /search/image`
- Request Type: `multipart/form-data`
- Parameters:
  - `file`: Vehicle image file (`JPEG`, `PNG`, `WebP`)
  - `top_k`: Number of matches to return (default: 10)
- Response: Returns top visually similar vehicles ranked by cosine similarity against the gallery.

### 3. Vehicle Detail Lookup
- Endpoint: `GET /vehicles/{vehicle_id}`
- Response: Detailed attributes, trajectory timeline, camera metadata, and full evidence crops.

### 4. Health Check
- Endpoint: `GET /health`
- Response: Service readiness, index size, and cloud storage connection status.

---

## Setup and Installation

### Prerequisites
- Python 3.10+
- PyTorch (CPU or CUDA)
- AWS / Supabase S3 credentials (for cloud media delivery)

### 1. Clone and Install Dependencies
```bash
git clone https://github.com/yaswantsaiadapa/vehicleiq-project.git
cd vehicleiq-project
pip install -r vehicleiq/requirements.txt
```

### 2. Configure Environment
Create a `.env` file inside the `vehicleiq/` directory:
```env
S3_ENDPOINT_URL=https://<project-ref>.storage.supabase.co/storage/v1/s3
AWS_ACCESS_KEY_ID=<your-access-key-id>
AWS_SECRET_ACCESS_KEY=<your-secret-access-key>
S3_BUCKET_NAME=vehicleiq-crops
S3_REGION_NAME=us-east-1
S3_PUBLIC_DOMAIN=https://<project-ref>.supabase.co/storage/v1/object/public/vehicleiq-crops
```

### 3. Run Verification Tests
```bash
python vehicleiq/tests/test_components.py
```

### 4. Start the Application
Terminal 1 - FastAPI Backend:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir vehicleiq
```

Terminal 2 - Streamlit Dashboard:
```bash
streamlit run vehicleiq/frontend/streamlit_app.py
```
Access the web console at `http://localhost:8501`.
