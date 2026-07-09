# MailKB — AI-Powered Email Archival & Analysis System

MailKB (Mail Knowledge Base) ingests corporate email archives (MBOX/PST), stores them in ClickHouse, uses LLMs to clean and parse email bodies, indexes them into a Qdrant vector store, and provides LangGraph-powered agents for batch and global project analysis.

The system processes raw email dumps into a structured, searchable, and analyzable knowledge base — enabling semantic search, thread analysis, and automated project reporting over thousands of emails.

---

## Architecture

```
User ─► FastAPI (port 8010) / CLI
            │
       Pipeline (6 steps):
          1. Init DB (DDL)
          2. Import MBOX → ClickHouse
          3. Deduplicate
          4. Clean bodies (LLM)
          5. Parse emails (LLM structured output)
          6. Index messages → Qdrant
            │
       Agent Analysis:
          Batch Agent → per-batch summaries
          Global Agent → consolidated report
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| OLAP DB | ClickHouse (clickhouse-connect) |
| Vector DB | Qdrant (langchain-qdrant) |
| Embeddings | Qwen3-Embedding-0.6B via vLLM |
| LLMs | DeepSeek / OpenAI / Anthropic |
| Agents | LangChain, LangGraph |
| Frontend | Node.js Express + vanilla HTML/CSS/JS |
| Infra | Docker Compose (6 services) |

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with drivers (for local embeddings via vLLM) — *optional, you can use a remote embeddings API*
- At least 16 GB RAM recommended

### 1. Clone and configure

```bash
git clone https://github.com/PeterOstr/shagal.git
cd shagal
```

Copy the example environment file and adjust settings:

```bash
cp .env.example .env
```

Minimum required settings:

```env
# Your MBOX file directory (required for import)
MBOX_DIR=/path/to/mbox/files

# LLM API keys
DEEPSEEK_API_KEY=sk-your-deepseek-key
# or
OPENAI_API_KEY=sk-your-openai-key

# HuggingFace token for downloading embedding models
HF_TOKEN=hf_your_token
```

### 2. Start all services

```bash
docker compose up -d
```

This starts:

| Container | Port | Purpose |
|-----------|------|---------|
| `mailkb-clickhouse` | 8123 | OLAP database |
| `mailkb-qdrant` | 6333 | Vector database |
| `mailkb-vllm` | 8000 | Embeddings server |
| `mailkb-api` | 8010 | FastAPI backend |
| `mailkb-ui` | 8501 | Web frontend |

### 3. Run the pipeline

**Option A — via Web UI:** Open `http://localhost:8501`, click each step or **Run Full Pipeline**.

**Option B — via API:** Send POST requests to `http://localhost:8010`:
```bash
curl -X POST http://localhost:8010/pipeline/init-db
curl -X POST http://localhost:8010/pipeline/import-mbox
curl -X POST http://localhost:8010/pipeline/dedup
curl -X POST http://localhost:8010/pipeline/clean-bodies
curl -X POST http://localhost:8010/pipeline/parse
curl -X POST http://localhost:8010/pipeline/index-messages
```

**Option C — via CLI** (runs directly on the host, not in Docker):
```bash
cd project
python cli.py import-mbox --max-emails 1000
python cli.py dedup
python cli.py clean-bodies --fetch-batch 30 --llm-batch 5
python cli.py parse --limit 50 --batch-size 3 --max-workers 6
python cli.py index-messages --batch-size 1000
```

### 4. Run analysis

```bash
# Per-batch summarization
python cli.py batch-analysis "project-name" --max-batches 5

# Global consolidated report (must run batch-analysis first)
python cli.py global-analysis "project-name"
```

---

## Configuration

All settings are controlled via environment variables (loaded from `.env` by `config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `CH_HOST` | `84.201.160.255` | ClickHouse server host |
| `CH_PORT` | `8123` | ClickHouse HTTP port |
| `CLICKHOUSE_USER` | `peter` | ClickHouse username |
| `CLICKHOUSE_PASSWORD` | `1234` | ClickHouse password |
| `CLICKHOUSE_DATABASE` | `mailkb` | ClickHouse database name |
| `MBOX_DIR` | `E:\outlook\mbox` | Directory containing MBOX files |
| `ATTACH_DIR` | `E:\outlook\attachments` | Attachment output directory |
| `SAVE_ATTACHMENTS` | `true` | Save attachments to disk |
| `LLM_MODEL` | `deepseek-chat` | LLM model for analysis agents |
| `EMBEDDINGS_BASE_URL` | `http://localhost:8000/v1` | Embeddings API endpoint |
| `EMBEDDINGS_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Embeddings model name |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `MESSAGES_COLLECTION` | `mailkb_messages` | Qdrant collection name |
| `SUMMARY_DIR` | `summaries` | Directory for analysis summaries |

---

## Pipeline Steps (Detailed)

### 1. `init-db` — Database Initialization

Creates the `mailkb` database in ClickHouse and runs all SQL DDL files from `project/sql/` in order (01–08). The SQL files create tables for raw emails, attachments, deduplicated views, parsed data cache, and threads.

### 2. `import-mbox` — Import Emails from MBOX

Reads all `.mbox` files from the configured `MBOX_DIR`, parses each message (headers, body, attachments), and batch-inserts into ClickHouse tables `emails` and `attachments`.

- Handles large mbox files (tested with 8.5 GB) using a custom `_iter_mbox()` generator to avoid Python's `mailbox.mbox()` hanging.
- Extracts both plain text and HTML bodies (HTML is stripped via BeautifulSoup).
- Decodes MIME-encoded headers (RFC 2047).
- Deduplicates by `message_id` within a batch.

### 3. `dedup` — De-duplication

Groups emails by `thread_key` (derived from subject normalization) and removes textual duplicates — keeps the earliest email when one body is a substring of another.

### 4. `clean-bodies` — Clean Email Bodies (LLM)

For each unique email, sends the body to an LLM that strips:
- Quoted/replied text (lines starting with `>`, `On ... wrote:`, etc.)
- Email signatures
- Forwarded message headers
- Excess whitespace

Results are cached in the `llm_body_clean_cache` table (keyed by MD5 hash of the original body) to avoid redundant LLM calls.

### 5. `parse` — Structured Email Parsing (LLM)

Extracts structured fields from each cleaned email body via a structured LLM call:

| Field | Description |
|-------|-------------|
| `from` | Sender name/email |
| `to` | Recipients |
| `cc` | CC recipients |
| `subject` | Email subject |
| `date` | Date string |
| `body` | Core message content |

Results are stored in the `mail_parsed` table as a JSON blob in the `parsed_json` column.

### 6. `index-messages` — Vector Indexing into Qdrant

JOINs `emails_unique` with `mail_parsed`, builds LangChain `Document` objects (one per thread entry), generates deterministic UUID5 IDs, and uploads to the `mailkb_messages` Qdrant collection.

Each document stores:
- **Content:** Parsed email body
- **Metadata:** `email_id`, `thread_key`, `subject`, `topic`, `date`, `participants`, `folder`, `from`, `to`

---

## Analysis Agents

### Batch Analysis Agent

Processes the project corpus in batches:

1. **`get_project_corpus_batch`** — Semantic search via Qdrant with `project_hint`, fetches full messages from ClickHouse, deduplicates and sorts by date.
2. **`save_summary`** — Saves each batch summary as a text file.
3. **Agent** — Iterates until `has_more=False`, extracting:
   - Main topics
   - Decisions and agreements
   - Problems, risks, open questions
   - Explicit and probable tasks
   - Responsible persons

### Global Analysis Agent

Consolidates all batch summaries into a single report with:
1. Project summary
2. Key discussion topics
3. Decisions made
4. Explicit tasks and owners
5. Probable tasks (reconstructed from context)
6. Risks and open questions
7. Recurring issues and bottlenecks
8. Final recommendations

---

## API Reference

The FastAPI server runs on port 8010. Swagger docs at `http://localhost:8010/docs`.

### Pipeline

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `GET` | `/health` | — | Health check |
| `POST` | `/pipeline/init-db` | — | Create tables |
| `POST` | `/pipeline/import-mbox` | `{max_emails?: int}` | Import MBOX |
| `POST` | `/pipeline/dedup` | — | Deduplicate |
| `POST` | `/pipeline/clean-bodies` | `{fetch_batch?: int, llm_batch?: int}` | Clean bodies |
| `POST` | `/pipeline/parse` | `{limit?: int, batch_size?: int, max_workers?: int}` | Parse emails |
| `POST` | `/pipeline/index-messages` | `{batch_size?: int, recreate?: bool}` | Index to Qdrant |

### Search & Analysis

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/search/threads` | `{project_hint: str, limit?: int}` | Semantic thread search |
| `POST` | `/search/corpus-batch` | `{project_hint: str, offset?: int, batch_size?: int}` | Paginated corpus |
| `POST` | `/analysis/batch` | `{project_hint: str, max_batches?: int}` | Batch analysis |
| `POST` | `/analysis/global` | `{project_hint: str}` | Global report |
| `POST` | `/summaries/clear` | — | Delete all summaries |

---

## CLI Reference

Run commands from the `project/` directory:

| Command | Arguments | Description |
|---------|-----------|-------------|
| `python cli.py import-mbox` | `--max-emails N` | Import MBOX → ClickHouse |
| `python cli.py dedup` | — | Deduplicate emails |
| `python cli.py clean-bodies` | `--fetch-batch N --llm-batch N` | Clean bodies via LLM |
| `python cli.py parse` | `--limit N --batch-size N --max-workers N` | Parse emails |
| `python cli.py index-messages` | `--batch-size N --recreate` | Index to Qdrant |
| `python cli.py batch-analysis` | `project_hint [--max-batches N]` | Batch analysis |
| `python cli.py global-analysis` | `project_hint` | Global report |
| `python cli.py clear-summaries` | — | Delete summaries |

---

## Frontend UI

The web UI runs at `http://localhost:8501` and provides:

- **Pipeline tab** — 6-step pipeline with per-step status, parameter configuration, and "Run All" button.
- **Search tab** — Project search with semantic thread lookup, corpus browsing, batch/global analysis triggers, and summary clearing.
- **Markdown rendering** — Analysis results rendered with full markdown support (tables, headings, lists, code blocks).
- **Language switcher** — Toggle between Russian and English. Preference is saved to `localStorage`.

---

## Development

### Running locally (without Docker)

```bash
# Backend
cd project
pip install -r requirements-dev.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8010

# Frontend
cd project/ui
npm install
node server.js  # serves on port 3000 by default
```

### Running tests

```bash
# Python tests
cd project
pip install -r requirements-dev.txt
pytest

# UI E2E tests (with Docker)
docker compose run --rm ui-test

# UI E2E tests (local, requires Node.js)
cd project/ui
npm test              # headless
HEADED=1 npm test    # headed (see the browser)
```

### Docker rebuild

After code changes, rebuild individual services:

```bash
docker compose build api
docker compose up -d --force-recreate api
```

---

## Project Structure

```
├── docker-compose.yaml        # 6-service Docker stack
├── AGENTS.md                  # Agent development guide
├── project/                   # ★ Active backend (MVP)
│   ├── app.py                 # FastAPI routes (11 endpoints)
│   ├── cli.py                 # CLI entry point (8 commands)
│   ├── config.py              # Env-based configuration
│   ├── infra.py               # Client factories (CH, Qdrant, LLM, embeddings)
│   ├── pipeline.py            # 6-step ETL pipeline (~1000 lines)
│   ├── retrieval.py           # LangGraph agents, tools, analysis
│   ├── Dockerfile             # API container build
│   ├── requirements.txt       # Python dependencies
│   ├── sql/                   # ClickHouse DDL/DML (8 files)
│   ├── tests/                 # Python unit tests
│   ├── ui/                    # Express.js frontend
│   │   ├── server.js          # Static file server + config endpoint
│   │   ├── package.json       # Node dependencies
│   │   ├── Dockerfile.ui      # UI container build
│   │   ├── public/            # SPA assets
│   │   │   ├── index.html     # Main HTML
│   │   │   ├── app.js         # UI logic (i18n, markdown, API calls)
│   │   │   ├── style.css      # Dark theme styles
│   │   │   └── marked.min.js  # Markdown renderer
│   │   └── tests/
│   │       └── ui.spec.js     # Playwright E2E tests
├── archive/                   # Legacy Jupyter notebooks
├── old_version/               # Earlier architecture attempts
└── work/                      # Active experimentation scripts
```

---

## FAQ

**Q: What format should my email archive be in?**
A: MBOX format. You can export from Outlook using tools like `readpst`, or directly export from Thunderbird, Apple Mail, etc.

**Q: Can I use a different LLM?**
A: Yes. Set `LLM_MODEL` and the corresponding API key. DeepSeek (`deepseek-chat`), OpenAI (`gpt-4o`, `gpt-4o-mini`), and any OpenAI-compatible API are supported.

**Q: Do I need a GPU?**
A: Not strictly — the embeddings server (vLLM) requires GPU for reasonable performance, but you can point `EMBEDDINGS_BASE_URL` to a remote embeddings API instead.

**Q: How long does the pipeline take for 10,000 emails?**
A: The LLM steps (clean-bodies and parse) are the bottleneck — roughly 3–10 seconds per email depending on the LLM provider. With parallel processing (`max_workers=6`) and batching, expect ~1–3 hours for 10,000 emails. Import and indexing are fast.

**Q: Can I search without running the full pipeline?**
A: Pipeline steps 1–6 must complete at least once to populate ClickHouse and Qdrant. After that, search and analysis work independently.
