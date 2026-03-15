# DiaMate-RAG

A **Retrieval-Augmented Generation (RAG)** system built for the DiaMate project. It ingests documents, generates embeddings, stores them in a vector database, and uses retrieved context to ground LLM responses — designed for diabetes-related health information.

## Features

- **Document Ingestion** — Upload TXT and PDF files, automatically chunk them with configurable size and overlap
- **Multi-Provider Support** — Swap between Google Gemini, OpenAI, OpenRouter, and Ollama for both embeddings and LLM via `.env`
- **Vector Search** — Store and retrieve document embeddings using Qdrant with cosine similarity
- **Conversation Memory** — Redis-backed chat history using native Lists with per-session isolation and TTL-based expiry
- **Model Switch Safety** — Detects embedding model changes and blocks search until data is re-indexed
- **Strict Grounding** — LLM answers only from retrieved context to prevent hallucination of medical information
- **CORS Support** — Configurable cross-origin resource sharing for frontend integration

## Architecture

```
Request → FastAPI Routes → Controllers → Providers / Stores
                                            │           │
                                     LLM & Embedding   Qdrant & Redis
```

| Layer | Purpose |
|-------|---------|
| **Routes** | REST endpoints for data, process, and chat |
| **Controllers** | Business logic — file handling, embedding orchestration, RAG pipeline |
| **Providers** | Unified async interface for LLM and embedding providers |
| **Stores** | Qdrant (vector storage) and Redis (conversation memory) |

## Project Structure

```
DiaMate-RAG/
├── docker/
│   ├── docker-compose.yml       # Qdrant + Redis services
│   └── env/
│       ├── qdrant.env.example
│       └── redis.env.example
├── src/
│   ├── main.py                  # FastAPI app with lifespan + CORS
│   ├── .env                     # Configuration (not committed)
│   ├── requirements.txt
│   ├── assets/files/            # Uploaded documents
│   ├── controllers/
│   │   ├── base_controller.py   # Shared settings, file path helpers
│   │   ├── data_controller.py   # File upload, validation, deletion
│   │   ├── process_controller.py# Document loading, chunking, embedding
│   │   └── chat_controller.py   # RAG pipeline: retrieve → prompt → generate
│   ├── providers/
│   │   ├── base_provider.py     # EmbeddingProvider & LLMProvider ABCs
│   │   ├── factory.py           # ProviderFactory from config
│   │   ├── google_provider.py   # Google Gemini
│   │   ├── openai_provider.py   # OpenAI
│   │   ├── openrouter_provider.py # OpenRouter (LLM only)
│   │   └── ollama_provider.py   # Ollama (local)
│   ├── stores/
│   │   ├── qdrant_store.py      # Qdrant vector DB wrapper
│   │   └── redis_store.py       # Redis conversation memory (List-based)
│   ├── routes/
│   │   ├── base.py              # Health check
│   │   ├── data.py              # File upload, list, delete
│   │   ├── process.py           # Embed files, status, reindex
│   │   ├── chat.py              # RAG chat, sessions, history
│   │   └── schemes/             # Pydantic request/response models
│   ├── models/enums/            # Response signals and processing enums
│   └── utils/config.py          # Pydantic Settings from .env
└── README.md
```

## Requirements

- Python 3.11+
- Docker & Docker Compose (for Qdrant and Redis)
- An API key for at least one provider (Google, OpenAI, or OpenRouter) — or Ollama running locally

## Setup

### 1. Clone and create environment

```bash
git clone <repo-url>
cd DiaMate-RAG
```

Using Miniconda (recommended):
```bash
conda create -n diamate-rag python=3.11
conda activate diamate-rag
```

Or using venv:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install dependencies

```bash
cd src
pip install -r requirements.txt
```

### 3. Start infrastructure

```bash
cd docker
docker compose up -d
```

This starts:
- **Qdrant** on `localhost:6333` (dashboard at http://localhost:6333/dashboard)
- **Redis** on `localhost:6379`

### 4. Configure environment

```bash
cd src
cp .env.example .env
```

Edit `.env` and set your provider and API key:

```env
# Choose your providers: google | openai | openrouter | ollama
EMBEDDING_PROVIDER="google"
LLM_PROVIDER="google"

# Set the API key for your chosen provider
GOOGLE_API_KEY="your-key-here"
```

### 5. Run the server

```bash
cd src
uvicorn main:app --reload
```

The API will be available at http://localhost:8000. Interactive docs at http://localhost:8000/docs.

## API Endpoints

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/` | Health check — returns app name, version, status |

### Data

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/data/upload` | Upload a TXT or PDF file |
| GET | `/api/v1/data/files` | List all uploaded file IDs |
| DELETE | `/api/v1/data/{file_id}` | Delete an uploaded file |

### Process

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/process/status` | Show embedding status — total, embedded, pending files |
| POST | `/api/v1/process/push` | Embed a single file into Qdrant |
| POST | `/api/v1/process/push_all` | Embed all un-embedded files into Qdrant |
| POST | `/api/v1/process/reindex` | Wipe collection and re-embed all files |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat/` | Send a question — returns grounded answer with sources |
| GET | `/api/v1/chat/sessions` | List all active chat sessions |
| GET | `/api/v1/chat/{session_id}` | Get conversation history for a session |
| DELETE | `/api/v1/chat/{session_id}` | Clear conversation history for a session |

## Usage Example

```bash
# 1. Upload a document
curl -X POST http://localhost:8000/api/v1/data/upload \
  -F "file=@diabetes_info.pdf"

# 2. Embed it into the vector store
curl -X POST http://localhost:8000/api/v1/process/push \
  -H "Content-Type: application/json" \
  -d '{"file_id": "returned-file-id-from-upload"}'

# 3. Ask a question
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-123", "question": "What are the symptoms of type 2 diabetes?"}'

# 4. View embedding status
curl http://localhost:8000/api/v1/process/status

# 5. View chat history
curl http://localhost:8000/api/v1/chat/user-123
```

## Supported Providers

| Provider | Embedding | LLM | Default Model |
|----------|-----------|-----|---------------|
| Google Gemini | Yes | Yes | `text-embedding-004` / `gemini-2.0-flash` |
| OpenAI | Yes | Yes | `text-embedding-3-small` / `gpt-4o-mini` |
| OpenRouter | No | Yes | `meta-llama/llama-3-8b-instruct` |
| Ollama (local) | Yes | Yes | `nomic-embed-text` / `llama3` |

> **Note:** OpenRouter does not support embeddings. If using OpenRouter as your LLM provider, set a different `EMBEDDING_PROVIDER` (e.g., `google` or `ollama`).

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `google` | Embedding provider: `google`, `openai`, `ollama` |
| `LLM_PROVIDER` | `google` | LLM provider: `google`, `openai`, `openrouter`, `ollama` |
| `GOOGLE_API_KEY` | — | Google Gemini API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | Provider default | Override embedding model name |
| `LLM_MODEL` | Provider default | Override LLM model name |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION` | `diamate` | Qdrant collection name |
| `REDIS_HOST` | `localhost` | Redis server host |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_CHAT_TTL` | `86400` | Chat session TTL in seconds (24h) |
| `MAX_FILE_SIZE` | `50` | Max upload file size in MB |
| `ALLOWED_FILE_TYPES` | `["text/plain", "application/pdf"]` | Allowed MIME types |

## License

See [LICENSE](LICENSE) for details.