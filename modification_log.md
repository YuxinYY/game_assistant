# Game Assistant Modification Log

## Working Rules

- Modify one blocking point at a time.
- Run one small, targeted test after each step.
- Do not move to the next step until the current step passes.
- Record changed files, intent, and validation result here.

## Current Decision

- `BaseAgent._decide()` is intentionally deferred for now.
- Reason: implementing `_decide()` alone would not make the pipeline usable yet; it also requires provider-compatible tool calling, structured tool output handling, and state write-back from ReAct observations.
- Near-term priority is to replace ReAct-dependent execution in `WikiAgent` and `CommunityAgent` with deterministic retrieval that writes real results into `AgentState`.
- ReAct and `_decide()` will be revisited only after the MVP pipeline can run end-to-end on a normal text query.

## Step Plan

| Step | Goal | Main Files | Validation | Status |
| --- | --- | --- | --- | --- |
| 1 | Make Router fall back to heuristic routing so Orchestrator can start | src/core/router.py, tests/test_workflows.py | pytest tests/test_workflows.py -q | Done |
| 2 | Implement real LLM Router with robust parsing and fallback | src/core/router.py, src/llm/client.py, tests/test_workflows.py | pytest tests/test_workflows.py -q | Done |
| 3 | Disable BM25 runtime failure so retrieval can degrade to dense-only mode | src/retrieval/hybrid_retriever.py, scripts/build_indexes.py, scripts/chunk_and_clean.py, tests/test_retrieval.py, tests/test_data_pipeline.py | targeted retriever unit test + sample index rebuild | Done |
| 4 | Replace ReAct-dependent WikiAgent and CommunityAgent execution with deterministic retrieval that updates AgentState | src/agents/wiki_agent.py, src/agents/community_agent.py, scripts/chunk_and_clean.py, tests/test_agents.py, tests/test_data_pipeline.py | targeted agent tests + sample rebuild + orchestrator smoke test | Done |
| 5 | Make SynthesisAgent produce a real answer with safe fallback and visible citations | src/agents/synthesis_agent.py, tests/test_agents.py | targeted synthesis tests + orchestrator smoke test | Done |
| 6 | Add one MVP smoke test for the minimal text-query pipeline | tests/test_mvp_smoke.py | patched end-to-end smoke test | Done |
| 7 | Revisit ReAct `_decide()`, tool-use compatibility, BM25 quality, and query rewrite after MVP stability | multiple files | phased tests | Planned |

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

### 2026-05-10 Step 4

- Intent: bypass the unfinished ReAct agent loop in runtime-critical agents while keeping real retrieval behavior.
- Changes:
  - Reworked `WikiAgent.execute()` to perform deterministic wiki retrieval and write real documents and identified entities into `AgentState`.
  - Reworked `CommunityAgent.execute()` to use deterministic query rewriting plus NGA retrieval, with a real fallback from chapter-filtered retrieval to unfiltered retrieval when metadata is incomplete.
  - Explicitly kept `BaseAgent._decide()` deferred; the MVP path no longer depends on ReAct to progress.
  - Updated the wiki chunking pipeline so sample wiki files without `raw_text` can synthesize text from `moves` and `tips` and enter the index.
  - Rebuilt sample chunks and indexes after the wiki pipeline update.
- Files:
  - src/agents/wiki_agent.py
  - src/agents/community_agent.py
  - scripts/chunk_and_clean.py
  - tests/test_agents.py
  - tests/test_data_pipeline.py
- Validation:
  - pytest tests/test_agents.py tests/test_retrieval.py tests/test_data_pipeline.py -q
  - 16 passed in 1.37s
  - sample rebuild succeeded: `data/processed/chunks.jsonl`, `data/indexes/chroma_db`, `data/indexes/bm25_index.pkl`
  - retrieval smoke test succeeded for both wiki and NGA sample data
  - orchestrator smoke test now completes and returns a populated `AgentState` with `final_answer=[TODO: synthesis LLM call not yet implemented]`
- Result:
  - A normal text query now passes Router, ProfileAgent, WikiAgent, CommunityAgent, and AnalysisAgent without hitting `BaseAgent._decide()`.
  - The current end-to-end blocker is no longer ReAct; it is only the placeholder synthesis output.

### 2026-05-10 Step 5

- Intent: replace the synthesis placeholder with real answer generation while keeping an evidence-based fallback.
- Changes:
  - Connected `SynthesisAgent.execute()` to `self.llm.complete(...)` using the existing synthesis prompt.
  - Added a no-results branch so synthesis does not fabricate answers when retrieval returns nothing.
  - Added an extractive fallback summary that is built only from retrieved documents and consensus analysis when LLM generation fails.
  - Appended a visible citation block to the final answer so sources remain inspectable even when the model summary is brief.
  - Added targeted synthesis tests for successful generation, fallback generation, and empty-result handling.
- Files:
  - src/agents/synthesis_agent.py
  - tests/test_agents.py
- Validation:
  - pytest tests/test_agents.py tests/test_retrieval.py tests/test_data_pipeline.py -q
  - 19 passed in 1.37s
  - orchestrator smoke test now completes with a non-placeholder `final_answer`
- Result:
  - The text-query pipeline now reaches a real synthesis output instead of returning a TODO marker.
  - If the LLM provider fails, the system still returns an evidence-based extractive summary rather than an empty or fabricated answer.

### 2026-05-10 Step 6

- Intent: lock the current MVP path with one automated text-query smoke test.
- Changes:
  - Added `tests/test_mvp_smoke.py` to validate that a normal text question can move through the MVP pipeline and produce a non-empty final answer plus citations.
  - Stubbed retrieval and LLM calls inside the smoke test so it remains deterministic and does not depend on live external APIs.
- Files:
  - tests/test_mvp_smoke.py
- Validation:
  - pytest tests/test_agents.py tests/test_retrieval.py tests/test_data_pipeline.py tests/test_mvp_smoke.py -q
  - 20 passed in 1.40s
- Result:
  - The current text-query MVP path is now covered by an automated smoke test.
  - Future changes can be checked against a concrete regression target instead of relying only on manual runs.
