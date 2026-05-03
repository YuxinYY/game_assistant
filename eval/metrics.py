"""
Evaluation metrics for the agent system.

Key metrics (all reportable as hard numbers in the final paper):
  - citation_rate: fraction of answers with ≥1 citation
  - spoiler_violation_rate: fraction of answers referencing content beyond player chapter
  - keyword_coverage: fraction of ideal_answer_includes keywords found in answer
  - workflow_accuracy: fraction of queries routed to the correct workflow
  - avg_react_steps: average number of tool calls per query
"""

from dataclasses import dataclass


@dataclass
class EvalResult:
    citation_rate: float
    spoiler_violation_rate: float
    keyword_coverage: float
    workflow_accuracy: float
    avg_trace_steps: float


def citation_rate(results: list[dict]) -> float:
    """Fraction of answers that include at least one citation."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("citations")) / len(results)


def spoiler_violation_rate(results: list[dict]) -> float:
    """Fraction of answers where retrieved docs contain spoiler chapters."""
    if not results:
        return 0.0
    violations = 0
    for r in results:
        max_ch = r.get("max_chapter", 6)
        retrieved_chs = r.get("retrieved_chapters", [])
        if any(ch > max_ch for ch in retrieved_chs if ch is not None):
            violations += 1
    return violations / len(results)


def keyword_coverage(results: list[dict]) -> float:
    """Fraction of ideal keywords present in the generated answer."""
    if not results:
        return 0.0
    scores = []
    for r in results:
        keywords = r.get("ideal_includes", [])
        if not keywords:
            continue
        answer = r.get("answer", "").lower()
        found = sum(1 for kw in keywords if kw.lower() in answer)
        scores.append(found / len(keywords))
    return sum(scores) / len(scores) if scores else 0.0


def workflow_accuracy(results: list[dict], eval_items: list[dict]) -> float:
    """Fraction of queries routed to the expected workflow/category."""
    if not results:
        return 0.0
    correct = sum(
        1 for r, item in zip(results, eval_items)
        if r.get("workflow") == item.get("category")
    )
    return correct / len(results)
