# Game Assistant Modification Log

## Working Rules

- Modify one blocking point at a time.
- Run one small, targeted test after each step.
- Do not move to the next step until the current step passes.
- Record changed files, intent, and validation result here.

## Current Decision

- `BaseAgent._decide()` is now implemented as a provider-agnostic JSON action planner on top of normal chat completion.
- `WikiAgent`, `CommunityAgent`, and `AnalysisAgent` now write tool results back into `AgentState` through the shared ReAct loop instead of bypassing it with deterministic-only execution.
- Query rewriting and reranking are now active runtime features instead of placeholders.
- Remaining work should focus on expanding community sources and stabilizing screenshot/VLM behavior rather than rebuilding the core agent loop again.

## Temporary Load-Shedding Overrides

| Date | Override | Scope | Revert Path | Status |
| --- | --- | --- | --- | --- |
| 2026-05-12 | Set `retrieval.reranker_mode=lexical` to stop per-document LLM reranking during query-time retrieval | Runtime load shedding for Groq rate-limit pressure; retrieval quality may degrade slightly on ambiguous queries | Change `retrieval.reranker_mode` back to `llm` in `config.yaml` after stable API credits/rate limits are available | Active |

## Step Plan

| Step | Goal | Main Files | Validation | Status |
| --- | --- | --- | --- | --- |
| 1 | Make Router fall back to heuristic routing so Orchestrator can start | src/core/router.py, tests/test_workflows.py | pytest tests/test_workflows.py -q | Done |
| 2 | Implement real LLM Router with robust parsing and fallback | src/core/router.py, src/llm/client.py, tests/test_workflows.py | pytest tests/test_workflows.py -q | Done |
| 3 | Disable BM25 runtime failure so retrieval can degrade to dense-only mode | src/retrieval/hybrid_retriever.py, scripts/build_indexes.py, scripts/chunk_and_clean.py, tests/test_retrieval.py, tests/test_data_pipeline.py | targeted retriever unit test + sample index rebuild | Done |
| 4 | Replace ReAct-dependent WikiAgent and CommunityAgent execution with deterministic retrieval that updates AgentState | src/agents/wiki_agent.py, src/agents/community_agent.py, scripts/chunk_and_clean.py, tests/test_agents.py, tests/test_data_pipeline.py | targeted agent tests + sample rebuild + orchestrator smoke test | Done |
| 5 | Make SynthesisAgent produce a real answer with safe fallback and visible citations | src/agents/synthesis_agent.py, tests/test_agents.py | targeted synthesis tests + orchestrator smoke test | Done |
| 6 | Add one MVP smoke test for the minimal text-query pipeline | tests/test_mvp_smoke.py | patched end-to-end smoke test | Done |
| 7 | Restore ReAct `_decide()`, reconnect tool-driven agent execution, and activate query rewrite + rerank | src/agents/base_agent.py, src/agents/wiki_agent.py, src/agents/community_agent.py, src/agents/analysis_agent.py, src/retrieval/query_rewriter.py, src/retrieval/reranker.py, src/retrieval/hybrid_retriever.py, tests/test_agents.py, tests/test_retrieval.py, tests/test_mvp_smoke.py | pytest tests -q | Done |

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

### 2026-05-10 Wiki Crawl Refresh

- Intent: make wiki data collection actually runnable again and remove the dependency on the currently unavailable Fextralife origin.
- Changes:
  - Replaced the fixed-page Fextralife crawler in `scripts/crawl_bwiki.py` with a BWIKI-based crawler.
  - Added automatic page discovery from the BWIKI `妖王` and `头目` index pages instead of hard-coding 7 bosses.
  - Switched detail-page fetching to the MediaWiki parse API for more stable structured extraction.
  - Added request retry handling for transient wiki-side 5xx/567 failures during page fetches.
  - Extracted infobox fields, gameplay-relevant sections, move descriptions, and a retrieval-friendly `raw_text` payload for each discovered boss page.
  - Updated `scripts/chunk_and_clean.py` so sample wiki files are skipped automatically when a real page for the same boss already exists.
  - Added parser-focused unit tests for index discovery, infobox parsing, move parsing, raw-text synthesis, and sample-file de-duplication.
- Files:
  - scripts/crawl_bwiki.py
  - scripts/chunk_and_clean.py
  - tests/test_crawl_bwiki.py
  - tests/test_data_pipeline.py
  - modification_log.md
- Validation:
  - pytest tests/test_crawl_bwiki.py tests/test_data_pipeline.py -q
  - 8 passed in 2.37s
  - live BWIKI crawl discovered 81 boss pages from index pages (`妖王` + `头目`)
  - real JSON files written for the full discovered boss set; `data/raw/wiki` now contains 81 real boss pages plus the original sample file
  - processed chunks rebuild succeeded: `data/processed/chunks.jsonl` with 163 chunks
  - index rebuild succeeded: `data/indexes/chroma_db`, `data/indexes/bm25_index.pkl`
  - retrieval smoke test succeeded with entity-filtered wiki hits for `石先锋`, `广智`, and `虎先锋`
- Result:
  - The project is no longer blocked by the Fextralife 502 origin failure for wiki collection.
  - Wiki scraping now aligns with the existing `bwiki` naming and sample-data conventions used elsewhere in the repo.
  - Newly scraped wiki pages are already integrated into the retrieval layer instead of only sitting in raw-data storage.

### 2026-05-10 English Wiki Extension

- Intent: add an English wiki source that can be retrieved directly by English questions without requiring cross-lingual matching.
- Changes:
  - Added `scripts/crawl_ign_wiki.py` to auto-discover boss guide pages from IGN's `Boss_List_and_Guides` index and scrape English boss guide content into the existing wiki raw-data schema.
  - Updated the wiki chunking pipeline to carry `source_site`, `source_language`, and page title into chunk metadata.
  - Extended the chunker to prefer English sentence breaks as well as Chinese sentence breaks.
  - Updated wiki search to auto-apply a language filter for pure English and pure Chinese queries.
  - Extended retrieval filter handling so metadata fields like `language` work in both Chroma and BM25 paths.
- Files:
  - scripts/crawl_ign_wiki.py
  - scripts/chunk_and_clean.py
  - src/tools/search.py
  - src/retrieval/hybrid_retriever.py
  - tests/test_crawl_ign_wiki.py
  - tests/test_data_pipeline.py
  - tests/test_retrieval.py
  - tests/test_search_tools.py
  - modification_log.md
- Validation:
  - pytest tests/test_crawl_ign_wiki.py tests/test_data_pipeline.py tests/test_retrieval.py tests/test_search_tools.py -q
  - 18 passed in 1.14s
  - live IGN discovery found 93 boss-guide pages
  - full IGN crawl completed successfully into `data/raw/wiki`
  - processed chunk rebuild succeeded: `data/processed/chunks.jsonl` with 1723 chunks
  - index rebuild succeeded: `data/indexes/chroma_db`, `data/indexes/bm25_index.pkl`
  - English retrieval smoke test succeeded for:
    - `How do I beat Tiger Vanguard?`
    - `Tiger Vanguard spinning kick`
    - `Where do I find Black Bear Guai?`
  - retrieved English docs carried `metadata.language=en` and `metadata.author=ign`
- Result:
  - The wiki corpus now contains both Chinese BWIKI pages and English IGN boss guides in a shared schema.
  - Pure English wiki queries are now restricted to English wiki chunks, so English questions no longer depend on cross-lingual matching.
  - Existing Chinese wiki data remains available under the same `source="wiki"` path, with language metadata attached during chunking.

### 2026-05-10 Step 7

- Intent: restore the README-level agent autonomy and make retrieval quality less dependent on raw BM25 / dense fusion order.
- Changes:
  - Implemented a provider-agnostic JSON action planner in `BaseAgent._decide()` using normal chat completion instead of provider-specific tool-calling.
  - Extended the shared ReAct loop so tool outputs can be written back into `AgentState` through subclass hooks.
  - Reconnected `WikiAgent`, `CommunityAgent`, and `AnalysisAgent` to the shared ReAct loop with action fallbacks that keep the pipeline usable even when the model output is malformed or unavailable.

  ### 2026-05-12 Temporary Load-Shedding 1

  - Intent: reduce Groq request volume without rewriting the workflow or removing the final-answer fallback behavior.
  - Changes:
    - Added a configurable reranker mode so retrieval can explicitly run in `llm` or `lexical` mode.
    - Updated `HybridRetriever` to avoid constructing an LLM client when lexical reranking is selected.
    - Switched the project config to `retrieval.reranker_mode=lexical` as a temporary load-shedding measure.
    - Added tests to lock both lexical-only reranking and the no-LLM-client construction path.
  - Files:
    - src/retrieval/reranker.py
    - src/retrieval/hybrid_retriever.py
    - config.yaml
    - tests/test_retrieval.py
    - modification_log.md
  - Validation:
    - pytest tests/test_retrieval.py -q
    - 18 passed in 2.03s
    - pytest -q
    - 105 passed in 4.21s
    - runtime config check: `retrieval.reranker_mode=lexical`
  - Result:
    - Query-time retrieval no longer spends one LLM request per rerank candidate, which substantially reduces Groq pressure during a full workflow run.
    - The change is intentionally reversible: switching `retrieval.reranker_mode` back to `llm` restores the previous reranking behavior once API credits and rate limits are no longer the bottleneck.
    - Expected tradeoff: retrieval ordering may be slightly weaker on ambiguous queries because reranking now relies on lexical overlap instead of LLM judgment.
  - Implemented LLM-backed query rewriting with robust parsing plus stronger rule-based fallback rewrites.
  - Implemented reranker scoring with LLM scoring first and lexical fallback second, then integrated reranking into `HybridRetriever.search()`.
  - Updated focused tests and MVP smoke coverage to reflect the restored ReAct path and the new retrieval ranking stage.
- Files:
  - src/agents/base_agent.py
  - src/agents/wiki_agent.py
  - src/agents/community_agent.py
  - src/agents/analysis_agent.py
  - src/retrieval/query_rewriter.py
  - src/retrieval/reranker.py
  - src/retrieval/hybrid_retriever.py
  - tests/test_agents.py
  - tests/test_retrieval.py
  - tests/test_mvp_smoke.py
  - modification_log.md
- Validation:
  - pytest tests/test_agents.py tests/test_retrieval.py tests/test_mvp_smoke.py -q
  - 20 passed in 0.95s
  - pytest tests -q
  - 53 passed in 2.05s
- Result:
  - The runtime path is no longer dependent on deterministic-only retrieval for `WikiAgent` and `CommunityAgent`.
  - Query rewrite and reranking are now part of the live retrieval path instead of documented but unfinished modules.
  - The repo is back in sync with the README claim that the system uses a real multi-agent tool loop for core retrieval and analysis stages.

### 2026-05-10 Step 8

- Intent: make the screenshot recognition path usable in the course-demo web app even when the main text model does not support vision.
- Changes:
  - Split text-provider and vision-provider resolution inside `LLMClient`, adding optional `VLM_PROVIDER` and `VLM_MODEL` support plus media-type inference for PNG/JPEG/WEBP.
  - Made screenshot parser stack fail closed: classifier/parsers now return safe defaults on missing vision support or parser exceptions instead of crashing.
  - Updated `ProfileAgent` to detect unavailable vision capability, trace `vision_unavailable`, and keep state stable when screenshots cannot be parsed.
  - Updated `Orchestrator` to preprocess screenshots before routing and support the screenshot-only path by returning `workflow="profile_update"` when the user uploads screenshots without a text query.
  - Updated Streamlit UI to support multi-image upload, a screenshot-only recognition button, and profile-update summaries in the right-side source panel.
  - Added/updated tests for independent vision-provider resolution, no-vision graceful degradation, and screenshot-only orchestrator smoke coverage.
- Files:
  - src/llm/client.py
  - src/tools/parsers/__init__.py
  - src/tools/parsers/classifier.py
  - src/agents/profile_agent.py
  - src/core/orchestrator.py
  - app/components/chat_ui.py
  - app/components/source_panel.py
  - .env.example
  - tests/test_llm_client.py
  - tests/test_agents.py
  - tests/test_mvp_smoke.py
  - modification_log.md
- Validation:
  - pytest tests/test_llm_client.py tests/test_agents.py tests/test_mvp_smoke.py -q
  - 18 passed in 0.14s
  - pytest -q (inside the configured project Conda environment)
  - 57 passed in 1.85s
- Result:
  - The app now supports screenshot-only profile updates end to end instead of requiring a text query as a trigger.
  - Text generation provider choice no longer blocks screenshot parsing, because vision can be configured separately.
  - Missing vision capability now degrades into a visible, non-crashing user path that preserves the rest of the pipeline.

### 2026-05-10 Step 9

- Intent: eliminate the remaining wiki no-result path for exact boss-name questions like `虎先锋有几个大招` when dense retrieval is down and BM25 tokenization is too narrow.
- Changes:
  - Added direct entity inference for wiki queries so `WikiAgent` can seed `identified_entities` from the raw user question before any document is retrieved.
  - Made `WikiAgent` deterministically do one entity-filtered `wiki_search`, then fall back to `entity_lookup` when that narrow search still returns no wiki docs.
  - Updated wiki tool lookup so `entity_lookup` no longer depends on BM25 full-text matching; it now reads exact-match wiki chunks straight from indexed chunk metadata first.
  - Extended Chinese query normalization for count-style phrasing such as `有几个大招`, while preserving the combat keyword `大招` as a usable sparse token.
  - Added regression coverage for entity-query fallback, direct entity lookup, and count-question tokenization.
- Files:
  - src/agents/wiki_agent.py
  - src/tools/search.py
  - src/retrieval/hybrid_retriever.py
  - tests/test_agents.py
  - tests/test_retrieval.py
  - tests/test_search_tools.py
  - modification_log.md
- Validation:
  - pytest tests/test_agents.py tests/test_retrieval.py tests/test_search_tools.py -q
  - 30 passed in 1.09s
  - live backend replay for `虎先锋有几个大招`:
    - with `PlayerProfile(chapter=1)`: `doc_count=1`, `citation_count=1`, answer now returns spoiler-gated feedback instead of `未找到足够资料`
    - with `PlayerProfile(chapter=2)`: `doc_count=1`, `citation_count=1`, answer returns a cited虎先锋 wiki response
- Result:
  - The failure mode has been shifted from `wiki检索为空` to normal answer generation with citations.
  - Exact entity lookup is now resilient even when the dense Chroma index is still broken and BM25 cannot match the entity page by raw token overlap.
  - The remaining limitation on the demo path is now mostly spoiler/profile behavior, not missing retrieval evidence.

### 2026-05-10 Step 10

- Intent: stop applying chapter/build/profile filters when the user has not explicitly filled any profile fields, and remove the legacy placeholder profile that was causing default Chapter 1 gating in the web app.
- Changes:
  - Changed `PlayerProfile` defaults so `chapter`, `build`, and `staff_level` start as unset instead of `1/dodge/1`.
  - Updated `ProfileAgent` and `apply_spoiler_filter()` to skip profile filtering and spoiler gating entirely when chapter is unset.
  - Updated the Streamlit profile sidebar to support `未设置` values for chapter/build/staff level and added a `清空筛选` action.
  - Added session-side migration for legacy placeholder profiles so old browser tabs no longer keep the implicit `第1章 + dodge + Lv.1` filters.
  - Added regression tests for unset-profile behavior in both spoiler filtering and profile-based doc filtering.
- Files:
  - src/core/state.py
  - src/agents/profile_agent.py
  - src/tools/spoiler_filter.py
  - app/components/profile_panel.py
  - app/session.py
  - tests/test_tools.py
  - tests/test_agents.py
  - modification_log.md
- Validation:
  - pytest tests/test_tools.py tests/test_agents.py tests/test_mvp_smoke.py -q
  - 24 passed in 1.21s
  - pytest -q
  - 66 passed in 1.74s
  - live backend replay for `虎先锋有几个大招`:
    - with unset profile: returns a cited wiki answer instead of treating the user as Chapter 1 by default
    - with explicit `chapter=1`: still behaves as chapter-gated content, as intended
- Result:
  - Unfilled profile fields no longer act as hidden filters.
  - Existing web sessions with the old placeholder profile are migrated away from implicit Chapter 1 gating.
  - The remaining runtime issues are unrelated to hidden filters: Chroma dense retrieval still logs `Error loading hnsw index`, and Groq can occasionally return `429` rate-limit errors during synthesis.

### 2026-05-10 Step 11

- Intent: stop different Tiger Vanguard questions from collapsing onto the same first wiki chunk and producing the same `乌鸦坐飞机`-biased answer.
- Changes:
  - Extended `entity_lookup()` to accept the original user query and rank exact-match wiki chunks inside the same entity by query relevance instead of always returning the first chunk.
  - Added entity-summary synthesis inside direct lookup so count-style questions can see move counts and move-name lists across multiple Tiger Vanguard chunks.
  - Updated `WikiAgent` to pass the original question into `entity_lookup`.
  - Added regression tests covering query-sensitive chunk selection and move-summary generation.
- Files:
  - src/tools/search.py
  - src/agents/wiki_agent.py
  - tests/test_search_tools.py
  - tests/test_agents.py
  - modification_log.md
- Validation:
  - pytest tests/test_search_tools.py tests/test_agents.py -q
  - 20 passed in 1.41s
  - pytest -q
  - 68 passed in 1.97s
  - live backend replay:
    - `虎先锋有几个大招` now retrieves a move-summary document with `可识别招式数量：15`
    - `虎先锋那个蓄力的大招怎么躲` now no longer falls back to the fixed first wiki chunk; it uses query-ranked exact entity lookup
- Result:
  - The old behavior where both questions were effectively answered from the first Tiger Vanguard wiki chunk has been removed.
  - Remaining quality issues for the second question are now mostly in generation fallback when Groq returns `429`, not in exact wiki evidence selection.

### 2026-05-10 Step 12

- Intent: stop `fact_lookup` and `navigation` questions from inheriting the boss-strategy answer template when they already route to the correct workflow.
- Changes:
  - Updated `SynthesisAgent` to choose a workflow-specific prompt file instead of always loading the single shared synthesis prompt.
  - Added dedicated prompt templates for `fact_lookup` and `navigation` so factual and location questions no longer ask for `build` advice or `招式识别` headings.
  - Added workflow-specific fallback and no-results formatting so Groq/API failures also preserve the correct answer shape.
  - Added regression tests to verify that `fact_lookup` uses fact-oriented sections in both normal and fallback paths.
- Files:
  - src/agents/synthesis_agent.py
  - src/llm/prompts/synthesis_fact_lookup.txt
  - src/llm/prompts/synthesis_navigation.txt
  - tests/test_agents.py
  - modification_log.md
- Validation:
  - pytest tests/test_agents.py -q
  - 16 passed in 1.98s
  - pytest -q
  - 70 passed in 2.36s
- Result:
  - Questions like `虎先锋有几个大招` no longer need to render under the strategy-only heading set.
  - Even when synthesis falls back after provider failure, `fact_lookup` now stays in a fact-summary format instead of reverting to `build/共识度` sections.

### 2026-05-11 Step 13

- Intent: stop wiki count-style answers from presenting community meme move names as if they were clean factual labels, and make source links usable in the web UI.
- Changes:
  - Updated `entity_lookup()` summary building so count-style queries now return `页面条目数 + 命名风险 + 统计口径说明` instead of a raw `招式清单`.
  - Added suspicious move-name detection for entries that look like community戏称/玩梗命名 or whose descriptions contain obvious meme phrases.
  - Narrowed count-query detection so questions like `那个蓄力的大招怎么躲` are no longer misclassified as `count` queries just because they contain `大招`.
  - Updated the Streamlit source panel to show clickable `打开原文` links and a clear fallback message when a citation has no URL.
- Files:
  - src/tools/search.py
  - app/components/source_panel.py
  - tests/test_search_tools.py
  - modification_log.md
- Validation:
  - pytest tests/test_search_tools.py tests/test_agents.py -q
  - 22 passed in 1.24s
  - pytest -q
  - 70 passed in 1.81s
  - live entity lookup check for `虎先锋有几个大招`:
    - URL remains `https://wiki.biligame.com/wukong/%E8%99%8E%E5%85%88%E9%94%8B`
    - summary now warns that the page contains community-style move naming and that the count is a page-entry count, not a strict official `大招数`
- Result:
  - The system no longer endorses strings like `后撤步7777` as if they were clean official move names in count-style fact answers.
  - Reviewers can now click through to the original source directly from the web UI instead of only seeing a raw URL string.

### 2026-05-11 Step 14

- Intent: stop move-listing answers like `虎先锋的大招都叫什么` from feeding raw meme-heavy wiki body text into synthesis, and remove the `（URL）` link formatting that caused Streamlit auto-linking to swallow parentheses.
- Changes:
  - Added a dedicated move-listing summary path in `entity_lookup()` for questions such as `都叫什么` / `有哪些大招` / `招式名字`, so direct lookup now returns structured move-name summaries instead of the full raw chunk body.
  - Kept suspicious move entries in a separate downgraded bucket, explicitly warning that quoted lines and community梗 should not be treated as formal move names.
  - Updated synthesis fallback and citation rendering to use explicit markdown links like `[原文](...)` instead of wrapping bare URLs in Chinese parentheses.
  - Added regression tests for move-listing summaries and fallback link formatting.
- Files:
  - src/tools/search.py
  - src/agents/synthesis_agent.py
  - tests/test_search_tools.py
  - tests/test_agents.py
  - modification_log.md
- Validation:
  - pytest tests/test_search_tools.py tests/test_agents.py -q
  - 23 passed in 1.04s
  - pytest -q
  - 71 passed in 1.87s
  - live direct lookup check:
    - `虎先锋的大招都叫什么` no longer returns raw chunk body that contains quoted meme lines like `老子起兵能源城之日`
    - fallback/citation links now render as explicit markdown links instead of `（URL）`
- Result:
  - The remaining issue is not an inherent model limitation; it was mainly caused by feeding noisy source text and ambiguous link formatting into the final answer layer.
  - Move-listing answers are now structurally constrained before synthesis, so the model has much less opportunity to mistake description quotes for move names.

### 2026-05-11 Step 15

- Intent: stop fact-enumeration questions like `虎先锋的大招都叫什么` from being routed into the `boss_strategy` workflow, which was reintroducing strategy-style synthesis and community-analysis stages even after the wiki summary was cleaned up.
- Changes:
  - Added a router-level priority rule for fact-shaped listing/count questions such as `几个` / `有哪些` / `都叫什么` / `招式名字`.
  - Kept action-oriented phrases like `怎么躲` / `怎么打` / `打法` on the `boss_strategy` path so move-counter questions still use the strategy workflow.
  - Added routing regression tests for count queries, listing queries, dodge queries, and an override case where the LLM returns `boss_strategy` but the query shape should still force `fact_lookup`.
- Files:
  - src/core/router.py
  - tests/test_workflows.py
  - modification_log.md
- Validation:
  - pytest tests/test_workflows.py -q
  - 13 passed in 0.86s
  - direct orchestrator replay for `虎先锋的大招都叫什么`:
    - workflow changed from `boss_strategy` to `fact_lookup`
    - trace now ends at `WikiAgent -> SynthesisAgent` instead of going through `CommunityAgent` / `AnalysisAgent`
  - pytest -q
  - 75 passed in 1.63s
- Result:
  - The remaining `还是一样` symptom was caused by routing, not by the previous search/synthesis fixes failing to apply.
  - Fact-enumeration questions now bypass the strategy pipeline and keep the cleaned-up wiki summary shape end to end.

### 2026-05-11 Step 16

- Intent: turn the demo into an English-first deliverable so the final web app, question flow, and synthesis layer all work naturally for English reviewers, while keeping the existing Chinese regression path intact.
- Changes:
  - Switched the Streamlit page title, chat panel copy, sidebar labels, source-panel labels, screenshot-update messages, and empty-state messages to English.
  - Added English-aware profile formatting so player state can be rendered cleanly in the UI and in synthesis prompts.
  - Added English prompt files for general synthesis, fact lookup, and navigation; English questions now receive English no-result messages, English fallback formatting, and English citation labels.
  - Strengthened routing heuristics for English queries such as `How do I beat ...`, `Which build is better ...`, `Where is ...`, and `How many ...`.
  - Improved query rewriting instructions so English questions keep English retrieval phrasing instead of drifting back into Chinese-first wording.
  - Added English regression coverage for routing, synthesis formatting, and an end-to-end English smoke path.
- Files:
  - app/streamlit_app.py
  - app/components/chat_ui.py
  - app/components/profile_panel.py
  - app/components/source_panel.py
  - src/core/state.py
  - src/core/router.py
  - src/retrieval/query_rewriter.py
  - src/agents/synthesis_agent.py
  - src/utils/language.py
  - src/llm/prompts/synthesis_agent_en.txt
  - src/llm/prompts/synthesis_fact_lookup_en.txt
  - src/llm/prompts/synthesis_navigation_en.txt
  - tests/test_workflows.py
  - tests/test_agents.py
  - tests/test_mvp_smoke.py
  - modification_log.md
- Validation:
  - pytest tests/test_workflows.py tests/test_agents.py tests/test_mvp_smoke.py -q
  - 40 passed in 1.55s
  - pytest -q
  - 84 passed in 1.97s
  - streamlit startup smoke check:
    - `python -m streamlit run app/streamlit_app.py --server.headless true --server.port 8502`
    - app booted successfully and exposed `http://localhost:8502`
- Result:
  - The demo is now English-first at the UI and final-answer layers instead of only having English retrieval hidden underneath a Chinese product shell.
  - English questions now have explicit regression coverage for routing, synthesis formatting, and end-to-end orchestration, while the existing Chinese path still passes the full suite.

### 2026-05-11 Step 17

- Intent: make sure the already-ingested English wiki corpus is always usable even when community corpora are sparse, and extend the local community pipeline so English questions can retrieve at least some indexed community evidence instead of relying on wiki-only answers forever.
- Changes:
  - Added indexed-source availability checks so CommunityAgent now plans searches by query language instead of always trying `NGA -> Bilibili -> Reddit` in that order.
  - English queries now prefer indexed English Reddit data first; if no compatible community source is indexed, CommunityAgent exits cleanly and leaves the answer path on wiki-only evidence instead of probing empty sources.
  - Added explicit trace events for `community_sources_unavailable` and `community_docs_not_found` so the no-community fallback is visible in debugging.
  - Extended `scripts/chunk_and_clean.py` to process Reddit and Bilibili JSONL files, and added an English Reddit sample corpus under `data/raw/reddit`.
  - Added English keyword support to consensus/conflict detection so downstream analysis can use English community evidence instead of treating it as opaque text.
- Files:
  - src/tools/search.py
  - src/agents/community_agent.py
  - scripts/chunk_and_clean.py
  - src/tools/consensus.py
  - data/raw/reddit/black_myth_wukong_guides_sample.jsonl
  - tests/test_agents.py
  - tests/test_data_pipeline.py
  - tests/test_tools.py
  - modification_log.md
- Validation:
  - pytest tests/test_agents.py tests/test_data_pipeline.py tests/test_tools.py tests/test_mvp_smoke.py -q
  - 42 passed in 1.53s
  - python scripts/chunk_and_clean.py
  - regenerated `data/processed/chunks.jsonl` with 1726 chunks
  - python scripts/build_indexes.py
  - rebuilt Chroma and BM25 indexes with 1726 docs
  - runtime retrieval check:
    - `wiki_search("How do I beat Tiger Vanguard?")` returned English IGN docs for `Tiger Vanguard`
    - `reddit_search("How do I beat Tiger Vanguard?")` returned English Reddit docs for `Tiger Vanguard`
  - pytest -q
  - 88 passed in 1.95s
  - streamlit startup smoke check:
    - `python -m streamlit run app/streamlit_app.py --server.headless true --server.port 8503`
    - app booted successfully and exposed `http://localhost:8503`
- Result:
  - Existing English wiki content is now guaranteed to remain usable even when community corpora are missing, because the community stage exits cleanly instead of probing empty sources and leaving the pipeline in a brittle state.
  - The local community pipeline now has an indexed English Reddit path, so English boss questions can retrieve both official IGN wiki evidence and English community strategy snippets in the same corpus.
  - Dense Chroma loading is still noisy at runtime on this machine, but the system now demonstrably falls back to sparse retrieval and continues returning English wiki and Reddit results instead of failing hard.

### 2026-05-11 Step 18

- Intent: stop recurring Chroma HNSW load failures from staying sticky across rebuilds, and make dense retrieval self-heal once before falling back for the rest of the process.
- Changes:
  - Extracted shared index-building helpers into `src/retrieval/index_builder.py` so runtime recovery and the one-off rebuild script use the same Chroma rebuild path.
  - Changed Chroma rebuild behavior to remove and recreate the entire persist directory instead of only deleting the collection, which avoids stale local HNSW segment state surviving across rebuilds on this machine.
  - Added HNSW-error detection plus one-time auto-rebuild-and-retry logic to `HybridRetriever._dense_search()`.
  - Added a process-local dense disable path so if recovery still fails, later queries stop retrying the broken dense index and continue through sparse retrieval without repeated HNSW log spam.
  - Added regression tests covering Chroma directory reset, successful dense self-heal, failed dense recovery disablement, and compactor/backfill HNSW error detection.
- Files:
  - src/retrieval/index_builder.py
  - src/retrieval/hybrid_retriever.py
  - scripts/build_indexes.py
  - tests/test_retrieval.py
  - tests/test_data_pipeline.py
  - modification_log.md
- Validation:
  - pytest tests/test_retrieval.py tests/test_data_pipeline.py -q
  - 26 passed in 2.95s
  - python scripts/build_indexes.py
  - rebuilt 1726 Chroma docs successfully after relocating Chroma persistence from the OneDrive-backed project folder to `C:\Users\22629\AppData\Local\game_assistant\chroma_a3eb888a77`
  - runtime dense retrieval check:
    - `HybridRetriever._dense_search("How do I beat Tiger Vanguard?", filters={"source": "wiki", "language": "en"})` returned 3 docs
    - first result URL: `https://www.ign.com/wikis/black-myth-wukong/Tiger_Vanguard`
    - dense retrieval remained enabled after the query
  - pytest -q
  - 93 passed in 3.93s
- Result:
  - The recurring Windows HNSW issue was traced to storing Chroma inside the OneDrive-backed workspace, where the persist directory becomes a reparse-point path and is prone to both index-load failures and directory-reset permission errors.
  - Dense retrieval now uses a local non-OneDrive cache path automatically on Windows, so normal indexing and querying no longer depend on the unstable synced folder.
  - If a dense HNSW load failure still occurs, the retriever now attempts one controlled rebuild and retry before disabling dense retrieval for the rest of the process and continuing on sparse fallback.

### 2026-05-12 Step 19

- Intent: increase agentic behavior without sacrificing the submission-safe reliability of the current web app.
- Changes:
  - Added a bounded execution planner in `src/core/planner.py` that attaches explicit goals, evidence gaps, stop conditions, and per-step status tracking to the selected workflow instead of opening the system to unrestricted planning.
  - Extended `AgentState` with structured planning/runtime fields including `execution_plan`, `evidence_gaps`, `completed_steps`, `skipped_steps`, `answer_confidence`, and `stop_reason`.
  - Updated the orchestrator to build a runtime plan, trace the plan, and skip low-value agents when their preconditions are not met, such as `ProfileAgent` without new profile evidence or `AnalysisAgent` without multi-source evidence.
  - Exposed `has_profile_signal_in_text()` from `ProfileAgent` so orchestration can make safe precondition checks without executing the whole profile-update agent.
  - Upgraded `CommunityAgent` from one-shot source probing to bounded goal-driven retrieval: it now derives search goals from the execution plan, generates targeted query variants, and can retry a source with a more specific query before moving on.
  - Added answer confidence tracking plus evidence-gap-aware next-step hints in `SynthesisAgent` so low-evidence branches stay grounded and actionable instead of silently collapsing into generic no-results text.
  - Added regression coverage for planning, conditional skipping, goal-driven community retrieval, and the end-to-end smoke path.
- Files:
  - src/core/state.py
  - src/core/planner.py
  - src/core/orchestrator.py
  - src/agents/profile_agent.py
  - src/agents/community_agent.py
  - src/agents/synthesis_agent.py
  - tests/test_agents.py
  - tests/test_planner.py
  - tests/test_mvp_smoke.py
  - modification_log.md
- Validation:
  - pytest tests/test_agents.py tests/test_planner.py tests/test_mvp_smoke.py -q
  - 31 passed in 1.75s
  - pytest -q
  - 99 passed in 5.11s
- Result:
  - The system is still workflow-bounded and submission-safe, but it is no longer just a fixed RAG chain: runtime planning now makes conditional decisions about which agents should execute and what evidence each stage should gather.
  - `CommunityAgent` now behaves more like a bounded evidence-seeking agent than a single-pass retriever, because it can refine the search objective and retry a source with a more targeted query before giving up.
  - Failure branches remain robust: when evidence is thin, the final answer now surfaces targeted next-step hints and an explicit confidence signal instead of simply returning a generic empty response.

### 2026-05-12 Step 20

- Intent: make the final generation step more resilient to Groq rate limits and expose enough execution state in the web UI to distinguish “pipeline failed early” from “pipeline completed but synthesis degraded”.
- Changes:
  - Added finite retry/backoff handling for Groq text completion in `src/llm/client.py`, retrying `429` and transient `5xx` responses before surfacing an error.
  - Added configurable `llm.retry_attempts` and `llm.retry_base_delay_seconds` to `config.yaml`.
  - Updated `SynthesisAgent` to classify fallback stop reasons more precisely, including `generation_rate_limited_fallback` for final-step rate limits.
  - Extended the Streamlit source/debug panel to show execution goals, evidence gaps, completed/skipped steps, stop reason, and answer confidence, so runtime debugging no longer depends only on the free-text fallback answer.
  - Added regression coverage for Groq 429 retry behavior and synthesis fallback stop-reason classification.
- Files:
  - src/llm/client.py
  - src/agents/synthesis_agent.py
  - app/components/source_panel.py
  - config.yaml
  - tests/test_llm_client.py
  - tests/test_agents.py
  - modification_log.md
- Validation:
  - pytest tests/test_llm_client.py tests/test_agents.py -q
  - 30 passed in 1.89s
  - pytest -q
  - 101 passed in 3.88s
- Result:
  - The web app can now survive short Groq rate-limit spikes more gracefully because the final synthesis call no longer fails fast on the first `429`.
  - When synthesis still has to fall back, the UI now makes it clear that retrieval/planning completed and only the final generation step degraded, which addresses the earlier ambiguity in the runtime output.

### 2026-05-12 Step 21

- Intent: add a second text-generation provider fallback for the final answer step only, so synthesis can still complete when the primary Groq text call fails.
- Changes:
  - Extended `LLMClient` with explicit provider/model overrides and availability checks so a second text-only client can be created safely without changing the rest of the pipeline.
  - Added synthesis-only fallback configuration in `config.yaml`: the primary text path can remain Groq, while final-answer fallback can explicitly target Anthropic.
  - Updated `SynthesisAgent` so if the primary provider fails during final answer generation, it attempts one secondary-provider completion before falling back to extractive formatting.
  - Added a dedicated stop reason `answered_via_secondary_provider` and a new trace action `synthesis_secondary_llm` so the UI/debug panel can show when the answer was rescued by the second provider rather than generated by the primary provider.
  - Added regression coverage for provider overrides and for the synthesis path where a Groq failure is recovered by a secondary Anthropic-like provider.
- Files:
  - src/llm/client.py
  - src/agents/synthesis_agent.py
  - config.yaml
  - tests/test_llm_client.py
  - tests/test_agents.py
  - modification_log.md
- Validation:
  - pytest tests/test_llm_client.py tests/test_agents.py -q
  - 32 passed in 1.71s
  - pytest -q
  - 103 passed in 3.93s
- Result:
  - The final answer step is now more robust: when Groq generation fails, the system can still produce a normal LLM-generated answer through the secondary provider instead of dropping straight to extractive fallback.
  - This fallback is tightly scoped to `SynthesisAgent`, so it improves answer completion reliability without changing provider behavior for routing, query rewriting, or other bounded-agent steps.

### 2026-05-12 Step 22

- Intent: stop English questions from producing mixed-language trace and debug-panel text, without adding a translation step that would increase model calls and failure surface.
- Changes:
  - Added query-language-aware localization to `BaseAgent`, so shared ReAct system prompts, state snapshots, tool descriptions, and fallback observations now switch between Chinese and English based on the user's question language.
  - Localized hard-coded fallback and trace text in `WikiAgent`, `CommunityAgent`, `AnalysisAgent`, and `ProfileAgent` so English queries no longer emit Chinese observations like `已有足够 wiki 证据`.
  - Added English ReAct prompt files for wiki/community/analysis agents so model-generated reasoning for English queries is less likely to drift back into Chinese.
  - Localized the Streamlit source/debug panel headings, labels, goal/gap captions, and stop-reason display so the right-side panel follows the same language as the question.
  - Added regression coverage to verify English trace observations stay free of Chinese fallback text in both unit-agent paths and the English MVP smoke path.
- Files:
  - src/utils/language.py
  - src/agents/base_agent.py
  - src/agents/wiki_agent.py
  - src/agents/community_agent.py
  - src/agents/analysis_agent.py
  - src/agents/profile_agent.py
  - app/components/source_panel.py
  - src/llm/prompts/wiki_agent_en.txt
  - src/llm/prompts/community_agent_en.txt
  - src/llm/prompts/analysis_agent_en.txt
  - tests/test_agents.py
  - tests/test_mvp_smoke.py
  - modification_log.md
- Validation:
  - pytest tests/test_agents.py tests/test_mvp_smoke.py -q
  - 31 passed in 1.99s
  - pytest -q
  - 107 passed in 3.88s
- Result:
  - English questions now keep the trace/debug surface in English rather than mixing in Chinese fallback text.
  - This solution is intentionally lighter than adding runtime translation: it removes the mixed-language UX problem without increasing LLM traffic, cost, or rate-limit pressure.

### 2026-05-12 Step 23

- Intent: fix two recurring runtime failures in the English boss-strategy path at the root cause level rather than patching a single question.
- Changes:
  - Added state-aware tool-argument binding in `BaseAgent` so required tool inputs can be resolved from `AgentState` when the model omits them or refers to them symbolically.
  - The new binding layer now fills common required arguments such as `query`, `topic`, `docs`, `entities`, `profile`, and a single inferred `entity` from the current shared state.
  - Added coercion for symbolic references like `retrieved_docs` and `state.user_query`, which prevents errors where the model sends a string placeholder instead of the real runtime object.
  - Updated synthesis source numbering so the `[Source N]` indices shown to the model now follow the same deduplicated citation numbering used in the final citation block.
  - Tightened citation de-duplication to use `(source, url)` keys and added regression coverage for both tool-arg binding and citation numbering consistency.
- Files:
  - src/agents/base_agent.py
  - src/agents/synthesis_agent.py
  - tests/test_agents.py
  - modification_log.md
- Validation:
  - pytest tests/test_agents.py -q
  - 31 passed in 1.44s
  - pytest -q
  - 110 passed in 3.89s
- Result:
  - `CommunityAgent` no longer fails when the model leaves out the required `query` argument for `query_rewrite`; the runtime now binds it from `state.user_query`.
  - `AnalysisAgent` no longer breaks when the model passes `retrieved_docs` as a literal string; the runtime now resolves that placeholder to the actual document list.
  - The final answer no longer risks citing a `Source N` that has no matching link in the appended citation block, because synthesis context numbering and citation numbering now use the same source keying rules.
