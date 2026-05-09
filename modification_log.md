# Game Assistant Modification Log

## Working Rules

- Modify one blocking point at a time.
- Run one small, targeted test after each step.
- Do not move to the next step until the current step passes.
- Record changed files, intent, and validation result here.

## Step Plan

| Step | Goal | Main Files | Validation | Status |
| --- | --- | --- | --- | --- |
| 1 | Make Router fall back to heuristic routing so Orchestrator can start | src/core/router.py, tests/test_workflows.py | pytest tests/test_workflows.py -q | Done |
| 2 | Implement real LLM Router with robust parsing and fallback | src/core/router.py, src/llm/client.py, tests/test_workflows.py | pytest tests/test_workflows.py -q | Done |
| 3 | Disable BM25 runtime failure so retrieval can degrade to dense-only mode | src/retrieval/hybrid_retriever.py, scripts/build_indexes.py, scripts/chunk_and_clean.py, tests/test_retrieval.py, tests/test_data_pipeline.py | targeted retriever unit test + sample index rebuild | Done |
| 4 | Bypass unfinished ReAct loop in WikiAgent and CommunityAgent with deterministic retrieval | src/agents/wiki_agent.py, src/agents/community_agent.py, tests/test_agents.py | targeted agent tests with mocked search | Planned |
| 5 | Replace synthesis TODO output with real LLM call plus safe fallback | src/agents/synthesis_agent.py, tests/test_agents.py | targeted synthesis tests | Planned |
| 6 | Add one MVP smoke test for the minimal pipeline | tests/ | patched end-to-end smoke test | Planned |
| 7 | Revisit ReAct loop, BM25 ranking quality, and query rewrite after MVP pipeline is stable | multiple files | phased tests | Planned |

## Change Entries

### 2026-05-09 Step 1

- Intent: remove the first hard blocker in the runtime path.
- Changes:
  - Updated Router to try LLM routing only when available and implemented; otherwise fall back to heuristic routing.
  - Added a test that covers the case where Router receives an injected llm_client but still needs to fall back safely.
- Files:
  - src/core/router.py
  - tests/test_workflows.py
- Validation:
  - pytest tests/test_workflows.py -q
  - exit code: 0
- Result:
  - Orchestrator will no longer fail immediately at Router because of NotImplementedError.

### 2026-05-09 Step 2

- Intent: make Router capable of using a real LLM classification path without giving up the safe fallback added in Step 1.
- Changes:
  - Implemented Router LLM routing through self.llm.complete with a constrained system instruction.
  - Added response parsing for plain category output, JSON output, and category mentions embedded in text.
  - Kept heuristic fallback in route() for API failures and malformed model responses.
  - Updated LLMClient to automatically load environment variables from .env when python-dotenv is available.
  - Added tests for valid direct LLM output, JSON-shaped LLM output, and LLM failure fallback.
- Files:
  - src/core/router.py
  - src/llm/client.py
  - tests/test_workflows.py
- Validation:
  - pytest tests/test_workflows.py -q
  - 9 passed in 0.71s
  - exit code: 0
- Result:
  - Router can now use a real LLM classification path when credentials are available.
  - If the API call fails or the model returns an invalid label, Router still falls back to heuristic routing.
  - .env values are now loaded automatically by LLMClient when python-dotenv is installed.

### 2026-05-10 Step 2 Extension

- Intent: validate the Router against a free provider without rewriting the router logic again.
- Changes:
  - Added Groq text-completion support to `LLMClient` using the OpenAI-compatible chat completions API.
  - Added provider/model resolution through `LLM_PROVIDER`, `LLM_MODEL`, and `GROQ_MODEL`.
  - Updated `.env.example` to document Groq-related environment variables.
  - Added targeted tests for Groq provider selection and request payload shape.
- Files:
  - src/llm/client.py
  - tests/test_llm_client.py
  - .env.example
- Validation:
  - pytest tests/test_workflows.py tests/test_llm_client.py -q
  - 11 passed in 1.78s
  - live Groq smoke test result:
    - PROVIDER=groq
    - MODEL=llama-3.1-8b-instant
    - RAW=boss_strategy
    - ROUTE=boss_strategy
- Result:
  - Router now works with a real Groq API key for text intent classification.
  - Live verification confirmed that the route result came from an actual model response, not only the heuristic fallback.

### 2026-05-10 Step 3 Notes

- While rebuilding sample indexes, the raw-to-chunks pipeline exposed a bug in `scripts/chunk_and_clean.py`.
- Short texts could enter a non-progressing overlap loop and eventually raise `MemoryError`.
- Fixed the chunker to stop at the end of text and always advance the sliding window.
- Added targeted chunking tests before retrying the sample index rebuild.

### 2026-05-10 Step 3

- Intent: remove the next runtime blocker in retrieval without fabricating sparse results.
- Changes:
  - Implemented safe BM25 loading that supports the new `{bm25, documents}` index format.
  - Disabled sparse retrieval automatically for legacy BM25 indexes that do not contain document mappings.
  - Implemented real sparse document reconstruction and filtering when the new BM25 bundle is available.
  - Updated Chroma filter building to support multiple conditions via `$and`.
  - Updated index building to store BM25 together with original chunk documents.
  - Fixed Chroma index build compatibility by flattening nested metadata and restoring it on retrieval.
  - Fixed the chunking loop in `scripts/chunk_and_clean.py` so short texts no longer trigger a non-progressing overlap loop.
- Files:
  - src/retrieval/hybrid_retriever.py
  - scripts/build_indexes.py
  - scripts/chunk_and_clean.py
  - tests/test_retrieval.py
  - tests/test_data_pipeline.py
- Validation:
  - pytest tests/test_retrieval.py tests/test_data_pipeline.py -q
  - 7 passed in 0.68s
  - sample chunk rebuild succeeded: `data/processed/chunks.jsonl`
  - sample index rebuild succeeded: `data/indexes/chroma_db`, `data/indexes/bm25_index.pkl`
  - sample retrieval smoke test succeeded for NGA sample data without chapter filter
- Result:
  - Retrieval no longer crashes on the unfinished BM25 branch.
  - New BM25 indexes can participate in real sparse retrieval.
  - Existing or incomplete BM25 indexes degrade safely to dense-only retrieval.
  - The sample dataset is now indexed and queryable, with the current limitation that the sample wiki file does not yet produce wiki chunks because it lacks a `raw_text` field.
