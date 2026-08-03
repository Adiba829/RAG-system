# VectorDB Engine

> A vector database built from scratch in Python — no FAISS, no Pinecone, no shortcuts.
>---

## What this is

Every production vector database — Pinecone, Weaviate, Qdrant — uses approximate nearest neighbour search under the hood. This project implements that search layer from scratch, along with a full RAG pipeline, to understand how it actually works rather than just calling an API.

Three search algorithms are implemented and kept in sync on every insert, so you can benchmark them against each other on identical data and measure recall accuracy in real time.

---

## Features

- **Three ANN algorithms** — HNSW, KD-Tree, and Brute Force, all hand-implemented in pure Python
- **Four distance metrics** — Cosine, Euclidean, Dot Product, Manhattan (L1)
- **Live algorithm benchmarking** — compare all three on the same query with microsecond timing
- **Recall@K evaluation** — measure HNSW and KD-Tree accuracy against brute-force ground truth
- **Full RAG pipeline** — upload documents → chunk → embed → HNSW retrieval → LLM generation
- **Dual embedding backend** — Ollama (nomic-embed-text, 768-dim) with automatic offline fallback to SentenceTransformers (MiniLM, 384-dim)
- **HNSW graph visualizer** — live canvas rendering of the multi-layer graph, filterable by layer
- **REST API** — 14 endpoints via FastAPI, fully documented at `/docs`
- **Lazy deletion** — tombstone-based deletion that preserves graph integrity (the production-correct approach)
- **Metadata filtering** — filter search results by category and tags
- **Thread-safe** — `threading.Lock` on all index mutations

---

## Screenshots

| Overview | Vector Search | HNSW Graph |
|---|---|---|
| Live stats and algorithm complexity table | Query with algorithm + metric + filter | Multi-layer graph rendered on Canvas |

| Benchmark | RAG Pipeline | Embed Text |
|---|---|---|
| µs comparison across all three algorithms | Upload docs, ask questions, see sources | Dimension heatmap of raw embeddings |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI REST API                      │
│           14 endpoints · Pydantic validation             │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
   ┌──────▼──────┐          ┌───────▼──────┐
   │   VectorDB  │          │  DocumentDB  │
   │  (16-D demo)│          │ (real embeds)│
   └──────┬──────┘          └───────┬──────┘
          │                         │
   ┌──────▼──────────────┐  ┌───────▼──────────────┐
   │ BruteForce          │  │ HNSW index           │
   │ KD-Tree             │  │ + BruteForce fallback│
   │ HNSW                │  └───────┬──────────────┘
   └─────────────────────┘          │
                               ┌────▼────────────────┐
                               │  RAG Pipeline        │
                               │  chunk → embed       │
                               │  retrieve → generate │
                               └─────────────────────┘
```

---

## Algorithm Comparison

| Algorithm | Build | Query | Memory | Accuracy | Best for |
|---|---|---|---|---|---|
| **Brute Force** | O(1) | O(n·d) | O(n·d) | Exact | Ground truth, < 5K vectors |
| **KD-Tree** | O(n log n) | O(log n)* | O(n·d) | Exact | Low dims ≤ 50 |
| **HNSW** | O(n log n) | O(log n) | O(n·M) | ~95–99% | Any dims, millions of vectors |

*KD-Tree degrades to O(n) above ~50 dimensions (curse of dimensionality).

### How HNSW works

HNSW builds a multi-layer graph where each node is a vector. Upper layers have few nodes with long-range edges for fast coarse navigation. Layer 0 has all nodes with dense local edges for precise retrieval. Search enters at the top layer, greedily moves toward the query, drops to the next layer, and repeats. This gives approximately O(log n) search regardless of vector dimension — which is why every production vector database uses it.

---

## RAG Pipeline

```
User uploads document
        ↓
Text chunked at 250 words, 30-word overlap
        ↓
Each chunk embedded → high-dimensional vector
(Ollama nomic-embed-text OR local SentenceTransformers)
        ↓
Vectors stored in HNSW DocumentDB
        ↓
User asks a question
        ↓
Question embedded with same model
        ↓
HNSW retrieves top-K most similar chunks (cosine)
        ↓
Chunks assembled into context prompt
        ↓
Ollama LLM generates grounded answer
        ↓
Response includes answer + source chunks + distances
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Git

### Install and run

```bash
# Clone
git clone https://github.com/Adiba829/vectorDB-engine-Python-.git
cd vectorDB-engine-Python-

# Virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn requests python-multipart pydantic --only-binary=:all:
pip install numpy --only-binary=:all:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers --only-binary=:all:

# Run
python main.py
```

Open **http://localhost:8080**

> First run downloads the MiniLM embedding model (~90MB). One-time only. The app works fully offline after that.

### Enable full RAG (optional)

Install [Ollama](https://ollama.com), then:

```bash
ollama pull nomic-embed-text   # 768-dim embedding model
ollama pull llama3.2           # generation model
ollama serve
```

Restart the server. The Ollama badge in the UI turns green and the Ask AI tab becomes fully functional.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/search` | KNN search — algorithm, metric, K, category filter |
| `POST` | `/api/insert` | Insert vector with metadata and tags |
| `PUT` | `/api/update/{id}` | Edit metadata without re-indexing |
| `DELETE` | `/api/delete/{id}` | Lazy-delete from all three indices |
| `GET` | `/api/items` | Paginated browse with category filter |
| `GET` | `/api/benchmark` | Run all 3 algorithms, return µs timings |
| `GET` | `/api/recall` | Recall@K vs brute-force ground truth |
| `GET` | `/api/hnsw-graph` | Export HNSW graph (nodes, edges, layers) |
| `POST` | `/api/doc/insert` | Upload, chunk, and embed a document |
| `POST` | `/api/doc/search` | Semantic search over document chunks |
| `POST` | `/api/doc/ask` | Full RAG: embed → retrieve → generate |
| `GET` | `/api/doc/list` | List all stored document chunks |
| `POST` | `/api/embed` | Embed any text, return raw vector |
| `GET` | `/api/status` | Ollama status, model names, doc counts |

Full interactive docs available at `/docs` (auto-generated by FastAPI).

### Example requests

```bash
# Search
curl "http://localhost:8080/api/search?v=0.9,0.85,0.72,0.68,0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05&algo=hnsw&metric=cosine&k=5"

# Insert
curl -X POST http://localhost:8080/api/insert \
  -H "Content-Type: application/json" \
  -d '{"metadata":"Transformer self-attention","category":"cs","embedding":[0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],"tags":["nlp","transformers"]}'

# Ask a question (RAG)
curl -X POST http://localhost:8080/api/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How does HNSW achieve O(log n) search?","k":3}'
```

---

## Project Structure

```
vectordb/
├── main.py                  ← entry point
├── requirements.txt         ← dependencies
├── Dockerfile               ← container deploy
├── Procfile                 ← Railway / Heroku
├── render.yaml              ← Render deploy
├── README.md                ← this file
├── README.html              ← interactive portfolio README
│
├── api/
│   └── app.py               ← FastAPI routes (14 endpoints)
│
├── core/
│   ├── indices.py           ← HNSW · KD-Tree · BruteForce from scratch
│   ├── database.py          ← VectorDB + DocumentDB classes
│   ├── embeddings.py        ← Ollama client + local fallback + RAG
│   └── demo_data.py         ← 20 hand-crafted 16-D demo vectors
│
└── static/
    └── index.html           ← full SPA web UI (no framework)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| Search indices | Pure Python — HNSW, KD-Tree, Brute Force |
| Embeddings | Ollama (nomic-embed-text) / SentenceTransformers (MiniLM) |
| LLM generation | Ollama (llama3.2) |
| Validation | Pydantic v2 |
| Frontend | Vanilla JS, CSS Grid, Syne + IBM Plex Mono |
| Deployment | Docker, Railway, Render |

---

## Deploy

### Railway (recommended — 2 minutes)

1. Fork this repo
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Add environment variable: `PORT = 8080`
4. Set start command: `uvicorn api.app:app --host 0.0.0.0 --port 8080`
5. Deploy

### Render

1. Fork this repo
2. Go to [render.com](https://render.com) → New Web Service → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn api.app:app --host 0.0.0.0 --port $PORT`
5. Deploy

### Docker

```bash
docker build -t vectordb-engine .
docker run -p 8080:8080 vectordb-engine
```

---

## Key Design Decisions

**Why implement HNSW from scratch instead of using hnswlib?**
To understand what's actually happening. Using hnswlib is one line. Building HNSW taught me why the layer structure matters, what ef_search controls, what happens to recall after deletion, and why KD-Trees fail at high dimensions. The production-correct version would use hnswlib or FAISS — both are available as drop-in replacements once you understand the algorithm.

**Why lazy deletion instead of graph repair?**
Proper node deletion in HNSW requires reconnecting the graph to maintain search quality — this is complex and expensive. Lazy deletion tombstones the node and skips it during traversal. This is what Qdrant and other production systems do because it keeps deletion O(1) while maintaining graph integrity.

**Why two embedding backends?**
Ollama provides higher-quality embeddings (768-dim nomic-embed-text) and LLM generation, but requires a local installation. SentenceTransformers works fully offline with no setup. The fallback means the system is always functional — useful for demos and deployments where Ollama can't run.

**Why FastAPI over Flask?**
Native async support, automatic OpenAPI documentation, and Pydantic validation with zero boilerplate. For a pure REST API serving ML workloads, FastAPI is the correct choice.

---

## What I'd add next

- **Payload filtering at index level** — filter by category inside HNSW rather than post-retrieval (Qdrant-style filtered vector search)
- **Hybrid search** — combine HNSW dense retrieval with BM25 sparse keyword search using Reciprocal Rank Fusion
- **Reranking** — cross-encoder reranking of top-20 candidates to improve final top-5 precision
- **LangChain tool wrapper** — expose RAG pipeline as a LangChain tool so an agent can call it
- **RAGAS evaluation** — automated RAG evaluation (faithfulness, context precision, answer relevancy)
- **Persistence** — binary serialization of HNSW index to disk on every insert

---

## License

MIT — use it, modify it, learn from it.

---

## Author

Built by **Adiba** as a deep-dive into how vector search actually works under the hood.
