# Module inspection notebooks

Keep interactive module experiments here; automated regression tests remain in
`tests/test_*.py`. Notebooks import application code instead of copying algorithms.

## Chunking inspection

`chunking_inspection.ipynb` reads the sample PDF, runs the current extractor,
structure annotator, and chunker in memory, and displays Persian RTL chunk cards
with their section paths. When an OpenRouter key is configured, it calls the
configured chat model for heading labels; otherwise it clearly reports that it is
using the deterministic fallback. It does not connect to the database or create
embeddings. Change `PAGE_LIMIT` to inspect more physical pages (the fixture contains
13).

From the repository root, launch Jupyter with temporary notebook dependencies:

```powershell
uv run --no-sync --with jupyterlab --with ipykernel jupyter lab tests/notebooks
```

After changing application code, **restart the kernel and run all cells** so imports
reflect your edits. The notebook does not access the database or persist ingestion
data. Jupyter saves the notebook's displayed outputs; tiktoken may download and cache
its encoding asset on first use. With OpenRouter configured, heading annotation sends
the selected sample paragraphs to the configured chat model; no embedding API call is
made.

To execute and refresh the saved outputs without opening Jupyter:

```powershell
uv run --no-sync --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute --inplace tests/notebooks/chunking_inspection.ipynb --ExecutePreprocessor.timeout=180
```

## Chunking rules

- Each citation paragraph retains its own identity, even when grouped for retrieval.
- A prose block immediately followed by list items is treated as their introduction.
  Consecutive list items form a group, including across physical pages.
- A group stays together up to the hard token limit, even above the soft target.
- Long lists split at item boundaries and repeat the introduction when it leaves
  room for item text. Oversized introductions fall back to bounded paragraph splits.
- Ordinary prose can overlap by one complete paragraph. Isolated bullets do not
  overlap into a different group.
- Consecutive headings stay with the following content. Heading levels are not
  inferred from PDF numbering; an ending heading is preserved even without content.
- Oversized source units retain exact half-open offsets into normalized source text.
