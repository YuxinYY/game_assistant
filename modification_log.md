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
| 3 | Disable BM25 runtime failure so retrieval can degrade to dense-only mode | src/retrieval/hybrid_retriever.py | targeted retriever unit test | Planned |
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
