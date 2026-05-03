"""
Run the full evaluation suite and print/save results.
Usage: python eval/run_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.core.orchestrator import Orchestrator
from src.core.state import PlayerProfile
from eval.metrics import (
    citation_rate,
    spoiler_violation_rate,
    keyword_coverage,
    workflow_accuracy,
    EvalResult,
)

EVAL_SET = Path("eval/eval_set.jsonl")
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_eval_set() -> list[dict]:
    items = []
    with open(EVAL_SET, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def run_single(item: dict, orchestrator: Orchestrator) -> dict:
    profile = PlayerProfile.from_dict(item["player_profile"])
    state = orchestrator.run(query=item["query"], profile=profile)
    return {
        "query": item["query"],
        "category": item["category"],
        "answer": state.final_answer or "",
        "citations": [c.__dict__ for c in state.citations],
        "workflow": state.workflow,
        "trace_steps": len(state.trace),
        "ideal_includes": item["ideal_answer_includes"],
        "max_chapter": item["must_not_include_spoiler_after_chapter"],
        "retrieved_chapters": [d.chapter for d in state.retrieved_docs if d.chapter],
    }


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    orchestrator = Orchestrator(config)
    eval_items = load_eval_set()
    print(f"Running eval on {len(eval_items)} items...")

    results = []
    for item in eval_items:
        print(f"  query: {item['query'][:40]}...")
        result = run_single(item, orchestrator)
        results.append(result)

    # Compute aggregate metrics
    metrics = {
        "citation_rate": citation_rate(results),
        "spoiler_violation_rate": spoiler_violation_rate(results),
        "keyword_coverage": keyword_coverage(results),
        "workflow_accuracy": workflow_accuracy(results, eval_items),
        "avg_trace_steps": sum(r["trace_steps"] for r in results) / len(results),
        "n": len(results),
    }

    print("\n=== Eval Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    out = RESULTS_DIR / "latest.json"
    out.write_text(json.dumps({"metrics": metrics, "results": results}, ensure_ascii=False, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
