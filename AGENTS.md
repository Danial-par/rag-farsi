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
- Cite supported answer claims individually; the generator assesses evidence adequacy and may abstain even when retrieval is nonempty.
- Use temporary evidence aliases for model-selected citations, resolve them server-side, and allow one bounded retry for malformed structured output.
- Persist only allowlisted model and token-usage metadata; never persist raw provider payloads.

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
- Normalize retrieval text with NFKC, Persian kaf and yeh forms, ASCII digits, collapsed whitespace, and valid single ZWNJs; remove Arabic vowel marks, tatweel, soft hyphens, ZWJ, and bidirectional controls.

## PDF extraction rules

- Use pypdf as the authoritative PDF text parser and pdfplumber only for geometry and explicit ruled-table detection.
- The initial extraction profile supports born-digital, predominantly single-column PDFs and unambiguous ruled tables.
- Fail ingestion for encrypted, malformed, empty, OCR-required, materially image-only, or ambiguous unsupported layouts rather than persisting incomplete evidence.
- Add table-header context only to normalized table-row text; add section context later during chunk construction, not to source paragraphs.

## Indexing rules

- Count chunk tokens with the embedding model's tiktoken encoding, falling back to cl100k_base.
- Target 400-token chunks with a 600-token hard limit and one complete-paragraph overlap.
- Repeat the nearest preceding heading as provenance-backed retrieval context.
- Oversized-paragraph offsets are half-open character offsets into normalized source text.
- Successful indexing activates automatically in the final publication transaction.
- Protect retriable ingestion attempts with attempt identifiers and renewable leases.
- Load immutable sources through async S3-compatible storage in production and a confined local adapter for fixtures.
- Generate OpenRouter embeddings through direct async HTTP with bounded configurable batching, concurrency, timeouts, and retries.

## Persistence rules

- Generate UUIDs and timezone-aware UTC timestamps in the application.
- Soft-delete logical documents. Versions that have ever been active remain immutable and addressable until an explicit aggregate purge; never-activated versions may be hard-deleted.
- A document points to its one active, successfully ingested immutable version. Version numbers are sequential within a document, while content hashes are indexed but not unique.
- Normalized chunk-to-paragraph rows are authoritative provenance. Chunks also retain a JSONB provenance snapshot, and completed assistant messages retain immutable validated citation snapshots.
- Store execution status separately from answer outcome on assistant messages.
- Run the RAG graph statelessly per request without checkpointing or graph-level retries. After conversation authorization, atomically create and commit the user message plus pending assistant message; terminal completion or failure is persisted separately.
- Require an `Idempotency-Key` for chat message creation, scoped to the conversation and authenticated owner; retries return the original assistant result, and key reuse with a different question conflicts.
- Conversation selection updates replace the explicit selection while retaining active shared default documents automatically. Conversation and message pagination uses descending creation or update timestamps with UUID tie-breakers and opaque cursors.
- The chat API waits for the complete non-streaming workflow. Failed assistant executions return HTTP 503 with the persisted assistant-message identifier; execution status and answer outcome remain separate response fields.
- API citations represent the complete immutable source paragraph and expose null offsets, including for paragraphs split across retrieval chunks.
- Use at most the previous 12 completed messages as generation-only conversation context. Retrieve using the current question alone; an empty retrieval result must persist a citation-free abstention.
- Authentication sessions are server-side revocation anchors referenced by JWT session claims; never store raw JWTs.
- Public self-registration is enabled. Authentication uses HS256 with a dedicated environment secret, 15-minute access tokens, and rotating 30-day absolute refresh sessions.
- Users may keep multiple independent device sessions. Refresh reuse outside a 10-second concurrency grace window revokes only the affected session family.
- Authentication cookies are host-only and `SameSite=Lax`; they are secure in production, with double-submit CSRF protection for unsafe cookie-authenticated requests.
- Production login and registration rate limiting is enforced at the deployment edge.
- Conversations select logical documents. Active shared default documents are automatically selected for every user conversation.

## Repository development rules

- Support Python 3.12 and manage dependencies with uv using a committed lockfile.
- Use `rag_assistant` as the application package and psycopg 3 as the PostgreSQL driver.
- Run PostgreSQL with pgvector in Docker Compose for local development and run the API on the host.
- Construct FastAPI through an application factory while exposing a module-level ASGI application.
- Emit structured JSON logs and propagate safe, bounded `X-Request-ID` values using request-local context.
- Expose unversioned liveness and database-backed readiness endpoints.
- Keep application boundaries small and cohesive; use async I/O end to end for database and external-service calls.
- Use typed configuration sourced from environment variables. Never commit secrets or credentials.
- Make schema changes with reviewed migrations; do not rely on runtime table creation in production.
- Add focused tests for behavior, authorization boundaries, Persian normalization, citation resolution, abstention, and migrations as those features are introduced.
- Use deterministic fixtures and mock external model, object-storage, and observability calls in automated tests.
- Run PostgreSQL integration and migration tests against a disposable pgvector database provided by Testcontainers.
- Trace every LangGraph node with Langfuse without recording secrets or authentication tokens.
- Preserve this file for durable, project-wide facts only. Do not add roadmap status, implementation history, rationale, or temporary task notes.
