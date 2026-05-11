# English Manual Test Set

This file is a manual evaluation set for the English demo path of the Black Myth: Wukong multi-agent assistant.

It is designed to test the abilities that matter most for the final project demo:

- English-only question handling
- correct workflow routing
- grounded wiki retrieval
- English community evidence usage
- bounded multi-agent execution
- player-state-aware personalization
- consensus/conflict handling
- honest uncertainty instead of fabrication
- English trace and English debug-panel output

## How To Use This Test Set

Use each question as a normal English user query in the Streamlit app.

For each question, record these items from the UI:

- final answer quality
- whether the answer stays in English
- whether the right-side trace/debug panel stays in English
- workflow name
- citations shown in the source panel
- completed steps and skipped steps
- stop reason and answer confidence

Recommended setup:

- Start from a clean session unless the question itself includes profile context.
- If a question includes chapter/build information in the text, do not override it manually unless you are intentionally testing a mismatch.
- For this round, the pass condition is language separation, not runtime translation. If the system cites cross-language evidence, the answer and trace should still remain English.

## What Agent Behavior Demonstrates The Design

These are the concrete behaviors that show the system is using the intended design rather than acting like a single free-form chatbot.

1. Router behavior

The workflow should match the question type.

- boss strategy questions should route to boss_strategy
- count/name/property questions should route to fact_lookup
- location questions should route to navigation
- build comparison questions should route to decision_making

This demonstrates that the system uses explicit workflow control instead of one undifferentiated prompt.

2. WikiAgent behavior

The trace should show wiki-first evidence collection for boss names, move names, and factual queries.

- good sign: entity grounding before synthesis
- good sign: finish after enough wiki evidence is collected
- bad sign: unsupported move names that are not grounded in sources

This demonstrates grounded retrieval from the local wiki index rather than pure generation.

3. CommunityAgent behavior

For strategy or decision questions, the trace should show community retrieval when it adds value.

- good sign: English queries prioritize English-friendly sources such as Reddit when indexed
- good sign: if community evidence is unavailable, the agent skips cleanly with an explicit reason instead of failing

This demonstrates bounded agentic retrieval instead of open-ended web browsing.

4. AnalysisAgent behavior

The agent should run only when there is enough multi-source evidence.

- good sign: consensus/conflict summary appears when sources disagree
- good sign: the step is skipped when there are not enough sources, with a visible reason

This demonstrates plan-aware execution instead of blindly running every agent every time.

5. ProfileAgent behavior

The system should only personalize when there is real profile evidence.

- good sign: text like "I'm in Chapter 2 and using a spell build" changes the answer scope
- good sign: locked or spoiler-heavy recommendations are avoided when chapter context is limited

This demonstrates player-state-aware filtering and spoiler control.

6. SynthesisAgent behavior

The final answer should be source-grounded and honest about evidence quality.

- good sign: citations are present
- good sign: uncertainty is admitted when evidence is thin or conflicting
- good sign: the answer shape matches the workflow, for example a direct fact answer for fact_lookup

This demonstrates grounded answer generation rather than generic advice dumping.

7. Source panel and trace behavior

The right-side panel should help explain why the answer looks the way it does.

- good sign: execution summary is readable and matches the workflow
- good sign: stop reason is informative
- good sign: English questions produce English trace text and English panel labels

This demonstrates observability and makes the multi-agent design legible to reviewers.

## Suggested Pass Checklist For Each Question

- workflow is correct
- answer is mainly English
- trace/debug panel is English
- at least one relevant citation is shown
- no obvious unsupported fabrication
- skipped steps, if any, are reasonable and explained

## Test Questions

| ID | English question | Main capability under test | Expected workflow | What to look for |
| --- | --- | --- | --- | --- |
| E1 | How do I dodge Tiger Vanguard's delayed slam? | Boss strategy, move grounding, English retrieval | boss_strategy | The answer should focus on dodge timing rather than generic build talk. The trace should show wiki retrieval first, then community retrieval if available, and the final answer should include citations. |
| E2 | How many major attacks does Tiger Vanguard have? | Fact counting, workflow override, fact-format synthesis | fact_lookup | The answer should look like a direct factual answer, not a boss-strategy template. Community and analysis steps may be skipped, and that is acceptable. |
| E3 | What are Tiger Vanguard's major attack names? | Entity summary, factual listing, anti-fabrication | fact_lookup | The answer should prefer grounded names from evidence. If the source naming is noisy, the system should admit uncertainty instead of inventing a clean list. |
| E4 | Where do I find Xu Dog? | Navigation workflow, minimal execution | navigation | The answer should be a location/path answer. The trace should not waste time on community strategy analysis. Skipping extra agents is a good sign here. |
| E5 | Which build is better for Yellow Wind Sage: dodge or spell? | Decision-making workflow, tradeoff comparison | decision_making | The answer should compare options rather than only recommending one build with no tradeoff. The evidence should be cited and framed as a comparison. |
| E6 | I'm in Chapter 2 and running a spell build. How should I approach Tiger Vanguard? | Text-based profile update, personalization, spoiler guard | boss_strategy | The answer should reflect Chapter 2 and a spell-oriented setup. It should avoid recommending clearly later-game locked content. This is a good question to confirm ProfileAgent is actually influencing downstream behavior. |
| E7 | Can I use staff parry against Tiger Vanguard's charge slash, or is dodging more reliable? | Conflict detection, consensus handling | boss_strategy | If sources disagree, the answer should surface that disagreement instead of pretending there is one perfect answer. A visible consensus/conflict signal is a strong pass. |
| E8 | What exact punish window, in seconds, do I get after Tiger Vanguard's delayed slam? | Honest uncertainty, non-fabrication under thin evidence | boss_strategy | If the corpus does not support an exact time value, the answer should say so clearly and fall back to grounded qualitative guidance. A precise unsupported number would be a fail. |

## Why These Questions Cover The Implemented Design

This set is intentionally mixed across workflows and evidence shapes.

- E1 and E7 test the full bounded multi-agent path: routing, retrieval, optional community evidence, optional consensus analysis, and cited synthesis.
- E2 and E3 test whether count/list questions are correctly forced into fact_lookup instead of drifting into strategy mode.
- E4 tests whether the planner can keep execution minimal and skip low-value agents.
- E5 tests decision-making rather than pure fact answering.
- E6 tests whether player profile information can actually change the final recommendation.
- E8 tests whether the system stays honest when the retrieval evidence is not strong enough for an exact claim.

## Strong Demo Outcome

If the system performs well on this set, reviewers should be able to see that:

- the assistant is not just a single RAG answerer
- the workflow is controlled and observable
- different question types trigger different bounded agent paths
- answers are grounded in sources
- uncertainty is surfaced honestly
- English questions now stay English across both the answer and the debug surface