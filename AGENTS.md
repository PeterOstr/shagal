# Shagal / MailKB — AI Email Archival & Analysis System

## Project Overview

MailKB (Mail Knowledge Base) is an AI-powered system for ingesting, cleaning, deduplicating, indexing, and analyzing corporate email correspondence. It ingests Outlook PST/MBOX files, stores them in ClickHouse, uses LLMs to clean and parse email bodies, indexes them into Qdrant vector store, and provides a LangGraph agent pipeline for batch and global project analysis.

### Key Technologies

| Layer | Tech |
|---|---|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| OLAP DB | ClickHouse (clickhouse-connect) |
| Vector DB | Qdrant (langchain-qdrant) |
| Embeddings | Qwen3-Embedding-0.6B via vLLM |
| LLMs | DeepSeek, OpenAI, Anthropic |
| Agents | LangChain, LangGraph |
| Frontend | Node.js Express + vanilla HTML/CSS/JS |
| Infra | Docker Compose (6 services) |

### Architecture Overview

```
User → FastAPI (port 8010) / CLI
         ↓
    Pipeline (6 steps):
      1. Init DB (DDL)
      2. Import MBOX → ClickHouse
      3. Deduplicate
      4. Clean bodies (LLM)
      5. Parse emails (LLM structured output)
      6. Index messages → Qdrant
         ↓
    Agent Analysis:
      Batch Agent → batch summaries
      Global Agent → consolidated report
```

## Project Structure

```
project/              # ★ Active backend (MVP)
  app.py              # FastAPI endpoints
  cli.py              # CLI entry point
  config.py           # Env-based config loader
  infra.py            # ClickHouse, Qdrant, embedding clients
  pipeline.py         # 6-step ETL pipeline
  retrieval.py        # LangChain agents, tools, search/analysis
  sql/*.sql           # ClickHouse DDL/DML
  ui/                 # Express.js frontend
    server.js
    public/index.html, app.js, style.css
    tests/ui.spec.js  # Playwright E2E
archive/              # Legacy notebooks (Jupyter)
old_version/          # Earlier architecture attempts
work/                 # Active experimentation notebooks/scripts
```

## Core Pipeline (project/pipeline.py)

1. **import_mbox_to_clickhouse()** — Parse MBOX, extract emails + attachments, batch-insert to `emails` + `attachments` tables
2. **deduplicate_emails()** — Remove duplicates within threads (keeps earliest email)
3. **clean_email_bodies_from_db()** — LLM agent strips quoted history, signatures, headers; results cached in `llm_body_clean_cache`
4. **parse_emails_from_db()** — LLM extracts structured fields (topic, body, date, order number, keywords); stored in `mail_parsed`
5. **index_messages()** — Join `emails_unique` + `mail_parsed`, build LangChain Documents, upload to Qdrant `mailkb_messages`

## Agent Analysis (project/retrieval.py)

- **search_project_threads** — Tool: search Qdrant by project query
- **get_project_corpus_batch** — Tool: fetch paginated batches from Qdrant
- **save_summary** — Tool: save batch summary to file
- **load_all_summaries** — Tool: load all batch summaries for global analysis
- **build_batch_agent()** — Per-batch summarization agent with PII/summarization middleware
- **build_global_agent()** — Consolidates batch summaries into one global report
- Middleware: PII redaction, context window management

## Commands

### API
```bash
cd project && uvicorn app:app --reload --host 0.0.0.0 --port 8010
```

### CLI (pipeline steps)
```bash
cd project
python cli.py import-mbox
python cli.py dedup
python cli.py clean-bodies --fetch-batch 30 --llm-batch 5
python cli.py parse --limit 50 --batch-size 3 --max-workers 6
python cli.py index-messages --batch-size 1000 --recreate
python cli.py batch-analysis "project-name"
python cli.py global-analysis "project-name"
python cli.py clear-summaries
```

### Docker (full stack)
```bash
docker-compose up -d
# Services: clickhouse, qdrant, vllm, api (8010), ui (8501), ui-test
```

### UI Tests
```bash
cd project/ui && npm install && npm test
```

## Agent Instructions for Development

When working on this project, follow these rules:

### 1. Understand before changing
- Read the relevant source file(s) completely before making any edit
- Check `config.py` for env vars, `infra.py` for client construction
- Check the ClickHouse SQL schema in `project/sql/` for table structures

### 2. Code style
- Module-level functions (not classes) for business logic; Pydantic models for I/O
- Type hints with Python 3.11+ syntax (`list[X]` not `List[X]`, `X | None` not `Optional[X]`)
- Snake_case for functions/variables, PascalCase for models
- Comments in Russian are acceptable (team convention)
- Batch processing with configurable sizes + tqdm progress bars
- CLI and API must call the same business functions (thin wrappers)

### 3. Pipeline changes
- Each pipeline step is a standalone function in `pipeline.py`
- Must be callable identically from both `cli.py` and `app.py`
- Never break the 6-step sequential model unless explicitly asked

### 4. Agent changes (retrieval.py)
- Tools use `@tool` decorator from `langchain.tools`
- Agents built with `create_agent()` or LangGraph's `StateGraph`
- Structured LLM output via `with_structured_output()` + Pydantic schema
- Wrap agent runs with PIIMiddleware + SummarizationMiddleware
- Use `call_with_retry()` for LLM resilience

### 5. Database
- ClickHouse: raw SQL via clickhouse_connect, batch inserts, `query_df()` for DataFrames
- Qdrant: `QdrantVectorStore` from `langchain_qdrant`, `add_texts()`, `similarity_search()`
- No ORM — direct SQL and vector store APIs only

### 6. Frontend changes
- Express serves static SPA from `project/ui/public/`
- Dark theme (see `style.css`)
- Two tabs: Pipeline (6-step) and Search (project analysis)
- JS modules in `app.js`, endpoints call FastAPI at `http://localhost:8010`
- E2E tests in `project/ui/tests/` via Playwright

### 7. Before committing
- `pip install -r requirements-dev.txt && pytest` for Python tests
- `cd project/ui && npm test` for frontend tests
- Review `git diff` for secrets exposure (API keys, passwords)
