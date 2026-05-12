# Black Myth: Wukong Multi-Agent Gameplay Assistant

A Streamlit-based multi-agent assistant that answers Black Myth: Wukong questions with source-grounded, player-state-aware advice.

## Quick Start

### Option 1: Use the hosted app

This is the recommended path for instructors and reviewers.

- Open the public Streamlit Community Cloud deployment URL: https://gameassistant-yuxin.streamlit.app/.
- No local Python environment, dependency installation, or data pipeline setup is required.
- The hosted app can run directly from the checked-in data and retrieval indexes in this repository.

### Option 2: Run locally

Use this path if you want to develop or debug the project locally.

Prerequisites:

- Python 3.11
- At least one text-model API key: OpenAI, Anthropic, or Groq
- Anthropic is still recommended if you want screenshot parsing and stronger synthesis fallback behavior

1. Create a Python 3.11 environment.

```bash
python -m venv .venv
```

If you prefer conda, any Python 3.11 environment works as well.

2. Activate the environment.

```powershell
.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

3. Copy `.env.example` to `.env`, then fill in your provider keys.

```powershell
Copy-Item .env.example .env
```

```bash
cp .env.example .env
```

4. Install dependencies from the repository root.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

5. Start the app from the repository root.

```bash
streamlit run app/streamlit_app.py
```

## Deploy on Streamlit Community Cloud

This repository is small enough to deploy directly on Streamlit Community Cloud because the processed data and retrieval indexes are already committed.

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Set the main file path to `app/streamlit_app.py`.
4. Keep Python on 3.11. This repo includes `runtime.txt` for that.
5. Add your secrets before the first run.

The app supports Streamlit secrets in either of these shapes:

```toml
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "replace_me"
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_API_KEY = "optional_but_recommended_for_vision_and_fallback"
GROQ_API_KEY = ""
GROQ_MODEL = "llama-3.1-8b-instant"
VLM_PROVIDER = "anthropic"
VLM_MODEL = "claude-sonnet-4-7"
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

- Do not commit `.env` or `.streamlit/secrets.toml`.
- One text provider key is enough for normal text QA.
- Screenshot parsing still benefits from an Anthropic key even if text generation uses OpenAI or Groq.

## What The App Does

- Answers boss-strategy, navigation, fact-lookup, and build-comparison questions.
- Grounds answers in local wiki and community evidence instead of relying on unsupported generation.
- Uses player context such as chapter, build, unlocked abilities, and screenshots to personalize advice.
- Shows citations, consensus notes, and execution traces in the UI.

## Architecture

```text
Query + PlayerProfile
      |
      v
   Orchestrator
   |- Router (intent -> workflow)
   `- Workflow execution
      |
      |- ProfileAgent   -> update player state and spoiler constraints
      |- WikiAgent      -> identify entities and gather wiki evidence
      |- CommunityAgent -> retrieve community tactics
      |- AnalysisAgent  -> summarize agreement and conflict
      `- SynthesisAgent -> write the final cited answer
          |
          v
      AgentState (shared whiteboard across all agents)
```

The system is intentionally bounded: agents operate within fixed workflows, shared state, and a small toolset instead of unconstrained autonomous planning.

## Repository Layout

```text
.
|- app/          Streamlit UI, session state, and UI components
|- data/         Checked-in raw data, processed chunks, and retrieval indexes
|- eval/         Evaluation runner, metrics, and manual test sets
|- scripts/      One-time crawling and index-building utilities
|- src/          Core orchestration, agents, tools, retrieval, and prompts
`- tests/        Automated regression tests
```

## Developer Commands

Run the automated tests:

```bash
pytest -q
```

Run the evaluation script:

```bash
python eval/run_eval.py
```

## Optional: Rebuild Data And Indexes

You do not need this section to use the hosted app or to run the checked-in demo locally. It is only for maintainers who want to refresh the corpus.

```bash
python scripts/crawl_bwiki.py
python scripts/crawl_ign_wiki.py
python scripts/chunk_and_clean.py
python scripts/build_indexes.py
python scripts/build_eval_set.py
```

`scripts/crawl_nga.py` is still partial and should not be treated as a required setup step.

## Design Choices

- State-driven instead of passing long parameter chains between agents.
- Workflows are defined as data, which keeps routing and execution easier to test.
- Prompts are stored as files, so prompt iteration does not require editing Python logic.
- Tools stay mostly pure and stateless, which makes regression testing easier.
- Spoiler filtering and build filtering happen before final synthesis whenever profile context is available.
- The final answer is required to stay honest about missing evidence and source conflicts.
