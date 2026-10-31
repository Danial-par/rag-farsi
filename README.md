# Persian RAG Assistant

A local, production-oriented RAG chat application for Persian PDF documents. It extracts and indexes documents in PostgreSQL with pgvector, answers through an OpenAI-compatible API, and returns citations with the physical page and paragraph number. The repository includes a FastAPI backend, ingestion worker, authentication, evaluation tools, and a minimal browser interface.

## Answer orchestration

Each chat request runs through a stateless LangGraph workflow. It first checks conversation access and loads up to 12 prior completed messages for conversational context, then creates or reuses the idempotent message execution. The current question is retrieved on its own from documents authorized for that conversation. If retrieval returns no chunks, the workflow abstains; otherwise it asks the structured answer model to answer only from the retrieved chunks. The answer is checked so its citations point only to retrieved chunks, then the completed answer and citation snapshot are persisted. Any failure after execution creation is recorded as a failed assistant message.

Langfuse records a readable input and output for each graph node, along with a short node description and duration. Payloads include the current workflow state, retrieved evidence, generated candidate, validation result, and persisted result so a trace can be followed from question to answer.

## Local setup

### 1. Install the required tools

Install these once before continuing:

- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://docs.docker.com/desktop/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

Install `uv` from PowerShell on Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

On macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the terminal after installing the tools. Start Docker Desktop and wait until it reports that Docker is running.

### 2. Clone and install the project

Open PowerShell or a terminal and run:

```powershell
git clone https://github.com/Danial-par/rag-farsi.git
cd rag-farsi
uv python install 3.12
uv sync --locked
```

### 3. Create the local configuration

The following command copies the configuration template and generates a random local JWT secret:

```powershell
uv run python -c "from pathlib import Path; import secrets; source = Path('.env.example').read_text(encoding='utf-8'); Path('.env').write_text(source.replace('replace-with-at-least-32-random-bytes', secrets.token_urlsafe(48)), encoding='utf-8')"
```

Open the new `.env` file:

```powershell
notepad .env
```

On macOS, use `open -e .env`. On Linux, use `nano .env`.

Set these values and save the file:

```dotenv
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=your-api-key
RERANKER_BASE_URL=http://127.0.0.1:8081
```

`OPENROUTER_BASE_URL` may instead be the full `/api/v1` URL of another OpenRouter-compatible service. Keep these defaults unless the service uses different model names:

```dotenv
CHAT_MODEL=openai/gpt-6-luna
CHAT_REASONING_EFFORT=none
EMBEDDING_MODEL=openai/text-embedding-3-large
EMBEDDING_DIMENSIONS=1024
```

The `.env` file contains secrets and is ignored by Git. Do not commit or share it. Langfuse settings are optional and may remain empty.

The local reranker runs in Hugging Face Text Embeddings Inference (TEI) and downloads the model weights the first time it starts. `RERANKER_BASE_URL` must point to that service; the default Compose mapping is `http://127.0.0.1:8081`. The cross-encoder model is configurable with `RERANKER_MODEL`.

When an OpenRouter key is configured, ingestion uses the chat model to identify heading levels and save parent/child section paths in chunk metadata. If annotation is disabled or unavailable, deterministic PDF heading labels remain as a fallback. Retrieval is a set of visible LangGraph stages: authorized vector search gathers up to 30 candidates, a multilingual `BAAI/bge-reranker-v2-m3` cross-encoder selects the final top six, and a compact chat-model assessment either uses those passages or asks for a bounded parent-section or sibling-section fetch. Expansion is reauthorized in the database and capped by `RETRIEVAL_MAX_EXPANDED_SECTIONS` and `RETRIEVAL_SECTION_MAX_CHARACTERS`.

### 4. Start PostgreSQL and prepare the database

```powershell
docker compose up -d database reranker
uv run alembic upgrade head
```

The reranker may take a few minutes to download and load its model on first startup. Wait for `http://127.0.0.1:8081/health` to return successfully before sending chat requests.

Verify that the configured chat and embedding models are reachable:

```powershell
uv run rag-provider-check
```

The result should contain `"success": true` and report `1024` embedding dimensions.

### 5. Index the included sample PDF

Create the shared sample document and process it through the real ingestion pipeline:

```powershell
uv run rag-seed-sample
uv run rag-worker --once
```

The sample is now stored, chunked, embedded, and ready for retrieval. Both commands are safe to run again.

### 6. Run the application

Keep two terminals open in the `rag-farsi` folder.

In the first terminal, start the ingestion worker. Leave it running so newly uploaded PDFs are processed:

```powershell
uv run rag-worker
```

In the second terminal, start the API and web interface:

```powershell
uv run rag-api --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser.

### 7. Start chatting

1. Register a local account in the browser.
2. Create a conversation.
3. The indexed sample document is selected automatically.
4. Ask a question about the document.
5. Expand the returned citations to inspect their document, page, paragraph, and original text.

You can also upload another born-digital PDF from the document panel. Keep the worker terminal running while its status changes from pending/processing to ready.

To run the broad-section retrieval evaluation against the seeded sample, use:

```powershell
uv run rag-eval-live --case-id fa-rag-components-overview
```

## Stop the application

Press `Ctrl+C` in the API and worker terminals, then stop PostgreSQL:

```powershell
docker compose down
```

The database and uploaded local documents remain available for the next run.

## Useful checks

Check API and database readiness:

```text
http://127.0.0.1:8000/health/ready
```

Run the automated test suite:

```powershell
uv run pytest
```

For an interactive view of the current PDF chunks, use
[`tests/notebooks/chunking_inspection.ipynb`](tests/notebooks/chunking_inspection.ipynb).
Notebook setup and execution commands are in
[`tests/notebooks/README.md`](tests/notebooks/README.md).

If setup fails, first confirm that Docker Desktop is running, the API URL ends in `/api/v1`, and the API key supports both the configured chat and embedding models.
