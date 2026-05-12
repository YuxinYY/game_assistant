# Black Myth: Wukong — Multi-Agent Gameplay Assistant

A **multi-agent** system that answers boss-strategy questions with verifiable, source-attributed,
player-state-aware advice. Built for the Text as Data final project.

<!-- ## Why Not Just Use an LLM?

| Dimension | Vanilla LLM | This System |
|-----------|-------------|-------------|
| Move name accuracy | ❌ Often fabricated | ✅ Grounded in wiki |
| Specific values | ❌ Made up | ✅ From player reports |
| Personalization | ❌ Generic advice | ✅ Filtered by your build |
| Verifiability | ❌ No sources | ✅ Citations + links |
| Uncertainty | ❌ Always confident | ✅ Flags consensus/dispute | -->

## Architecture

```
Query + PlayerProfile
        │
        ▼
   Orchestrator
   ├── Router (intent → workflow)
   └── Workflow execution
        │
        ├── WikiAgent       → identifies move names from bwiki
        ├── CommunityAgent  → retrieves NGA/Bilibili/Reddit tactics
        ├── AnalysisAgent   → consensus count + conflict detection
        ├── ProfileAgent    → filters by player state + spoiler guard
        └── SynthesisAgent  → writes cited, honest final answer
              │
              ▼
        AgentState (shared whiteboard across all agents)
```

Each agent runs a **ReAct loop** (Think → Tool → Observe → repeat, max 3 iterations).

## Project Structure

```
.
├── config.yaml                # all magic numbers — no constants in code
├── .env.example               # API key template
│
├── data/
│   ├── raw/{wiki,nga,bilibili,reddit}/
│   ├── processed/chunks.jsonl
│   └── indexes/{chroma_db/,bm25_index.pkl}
│
├── scripts/                   # run once, not part of runtime
│   ├── crawl_bwiki.py
│   ├── crawl_nga.py
│   ├── chunk_and_clean.py
│   ├── build_indexes.py
│   └── build_eval_set.py
│
├── src/
│   ├── core/
│   │   ├── state.py           # AgentState — shared whiteboard
│   │   ├── orchestrator.py    # main controller
│   │   ├── router.py          # intent → workflow name
│   │   └── workflows.py       # workflow definitions as data, not if-else
│   ├── agents/
│   │   ├── base_agent.py      # ReAct loop, one implementation
│   │   ├── wiki_agent.py
│   │   ├── community_agent.py
│   │   ├── profile_agent.py
│   │   ├── analysis_agent.py
│   │   └── synthesis_agent.py
│   ├── tools/                 # pure functions, stateless
│   │   ├── search.py
│   │   ├── consensus.py
│   │   ├── spoiler_filter.py
│   │   └── screenshot_parser.py
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # BM25 + ChromaDB, RRF fusion
│   │   ├── reranker.py
│   │   └── query_rewriter.py   # vague desc → wiki terms
│   ├── llm/
│   │   ├── client.py
│   │   └── prompts/            # all prompts as .txt files, not code strings
│   └── utils/
│       ├── tracing.py
│       ├── cache.py
│       └── logging.py
│
├── eval/
│   ├── eval_set.jsonl          # 5 seed QA pairs (expand to 30+)
│   ├── run_eval.py
│   └── metrics.py              # citation_rate, spoiler_rate, keyword_coverage, workflow_accuracy
│
├── app/
│   ├── streamlit_app.py        # 3-column layout
│   ├── session.py
│   └── components/
│       ├── chat_ui.py
│       ├── profile_panel.py    # sidebar: chapter, build, unlocked items
│       └── source_panel.py     # citations + consensus + agent trace (demo killer)
│
└── tests/
    ├── test_agents.py
    ├── test_tools.py
    └── test_workflows.py
```

## Setup

```bash
cp .env.example .env
# fill in one text provider key: ANTHROPIC_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY

pip install -r requirements.txt
```

## Data Pipeline (run once)

```bash
python scripts/crawl_bwiki.py
python scripts/crawl_nga.py
python scripts/chunk_and_clean.py
python scripts/build_indexes.py
python scripts/build_eval_set.py
```

## Run the App

```bash
streamlit run app/streamlit_app.py
```

## Deploy on Streamlit Community Cloud

This repository is small enough to deploy directly on Streamlit Community Cloud, including the checked-in retrieval data.

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create a new app from that GitHub repository.
3. Set the main file path to `app/streamlit_app.py`.
4. Keep the Python version on 3.11. This repo includes `runtime.txt` for that.
5. Add your secrets in the app settings before the first run.

The app now supports Streamlit secrets in either of these shapes:

```toml
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "replace_me"
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_API_KEY = "optional_but_recommended_for_vision_and_fallback"
GROQ_API_KEY = ""
GROQ_MODEL = "llama-3.1-8b-instant"
```

or:

```toml
[llm]
provider = "openai"

[openai]
api_key = "replace_me"
model = "gpt-4o-mini"

[anthropic]
api_key = "optional_but_recommended_for_vision_and_fallback"

[groq]
api_key = ""
model = "llama-3.1-8b-instant"
```

You can copy from `.streamlit/secrets.toml.example` when filling the Streamlit Cloud secrets editor.

Notes:

- Do not upload your local `.env` file to GitHub.
- Rotate any API keys that have already been exposed outside your machine.
- Screenshot parsing still benefits from an Anthropic key, even if text generation uses OpenAI.

## Run Evaluation

```bash
python eval/run_eval.py
```

## Run Tests

```bash
pytest tests/
```

## Key Design Decisions

- **State-driven, not param-passing**: all agents read/write `AgentState`; no spaghetti param chains
- **Workflows as data**: `workflows.py` maps name → `[AgentClass, ...]`; adding a workflow = one list entry
- **Prompts as files**: all `.txt` in `src/llm/prompts/`; tweak prompt without touching code
- **Tools are pure functions**: stateless, independently testable, reusable across agents
- **Player-state filter at retrieval**: chapter-gated content never reaches the LLM
- **Honest uncertainty**: `SynthesisAgent` prompt forbids unsupported claims; conflicts flagged with ⚠️
