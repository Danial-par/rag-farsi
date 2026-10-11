# Project Contract

## Purpose and stack

- Build a small, production-quality RAG chat backend for Persian documents.
- Use FastAPI and Pydantic for the HTTP API and schemas.
- Use async SQLAlchemy with PostgreSQL, UUID primary keys, JSONB metadata, and pgvector.
- Authenticate with JWTs carried only in HTTP-only cookies.
- Implement the workflow with LangGraph and an explicit `TypedDict` state.
- Use LangChain `ChatOpenAI` with structured output and trace every graph node with Langfuse.
- Ingest PDFs with stable, human-verifiable page and paragraph citations.

## Architectural invariants

- Generate answers only from retrieved document evidence. Do not fill evidence gaps with model knowledge.
- Every returned citation must resolve to persisted source text.
- Citation identifiers are ingestion metadata; the LLM may select supplied identifiers but must never create or alter them.
- Enforce document authorization inside the retrieval query before vector ranking. Never retrieve broadly and filter afterward.
- Fail closed when authorization scope or document ownership cannot be established.
- Preserve extracted original source text separately from normalized retrieval text.
- Keep retrieval chunks separate from citation units and retain the source-unit identifiers represented by every chunk.
- Prefer direct, typed components and explicit data flow over generic frameworks or speculative abstractions.

## Citation contract

- A page number is the 1-based physical page index in the PDF, independent of any printed page label.
- Paragraph numbering restarts at 1 on every physical page.
- A citation paragraph is one page-bounded logical source block.
- Prose paragraphs, headings, individual list items, and captions are separate citation paragraphs.
- Each logical table row is a citation paragraph. Include its applicable table-header context in normalized retrieval text while preserving the original row text separately.
- Split content that crosses a physical page boundary into independently numbered page-local citation paragraphs.
- Citation identity is based on the document, physical page, and page-local paragraph number; chunk boundaries must not change it.

## Evidence behavior

- Distinguish fully answered, partially answered, and insufficient-evidence results in the structured response.
- For a partially supported question, answer only the supported portion with citations and explicitly identify the unsupported portion.
- When no material answer is supported, explicitly abstain and return no citations.
- Citations may reference only evidence supplied by retrieval and validated against stored source units.

## Documents and models

- The approved sample PDF is a non-sensitive, redistributable test fixture and belongs under `tests/fixtures/documents/`.
- Store production source PDFs as immutable objects in private S3-compatible object storage.
- Store each source object's owner, object key, content hash, ingestion status, and extraction metadata in PostgreSQL.
- Access models through OpenRouter's OpenAI-compatible API using `OPENROUTER_API_KEY`.
- Default to `openai/gpt-6-luna` for chat and `openai/text-embedding-3-large` with 1024 dimensions for embeddings.
- Keep gateway URL, model identifiers, and embedding dimensions configurable.
- Use OpenRouter's default automatic upstream routing.
- Persist the embedding model and dimensions with indexed data, and never compare vectors from incompatible embedding configurations.

## Persian text rules

- Treat UTF-8 Persian and right-to-left content as first-class throughout extraction, storage, retrieval, APIs, logs, and tests.
- Preserve original extracted Unicode without retrieval normalization for audit and citation display.
- Normalize only the retrieval representation, including consistent Persian/Arabic character forms, digits, whitespace, and half-spaces.
- Normalization must be deterministic and must not destroy the mapping from retrieval text to its original citation unit.

## Repository development rules

- Keep application boundaries small and cohesive; use async I/O end to end for database and external-service calls.
- Use typed configuration sourced from environment variables. Never commit secrets or credentials.
- Make schema changes with reviewed migrations; do not rely on runtime table creation in production.
- Add focused tests for behavior, authorization boundaries, Persian normalization, citation resolution, abstention, and migrations as those features are introduced.
- Use deterministic fixtures and mock external model, object-storage, and observability calls in automated tests.
- Trace every LangGraph node with Langfuse without recording secrets or authentication tokens.
- Preserve this file for durable, project-wide facts only. Do not add roadmap status, implementation history, rationale, or temporary task notes.
