"""
Semi-automatic eval set construction.
Sources: NGA "求助" post titles (real player pain points), manually curated answers.
Output: eval/eval_set.jsonl

Run: python scripts/build_eval_set.py
"""

import json
from pathlib import Path

OUTPUT = Path("eval/eval_set.jsonl")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Seed set: manually written ground-truth QA pairs
SEED_QA = [
    {
        "query": "虎先锋那个跳起来蓄力的招怎么躲？",
        "category": "boss_strategy",
        "player_profile": {"chapter": 1, "build": "dodge", "staff_level": 1,
                           "unlocked_skills": [], "unlocked_spells": [], "unlocked_transformations": []},
        "ideal_answer_includes": ["虎跃斩", "侧闪", "蓄力"],
        "must_not_include_spoiler_after_chapter": 1,
        "ground_truth_move_name": "虎跃斩",
        "source": "manual",
    },
    {
        "query": "广智什么时候解锁？",
        "category": "fact_lookup",
        "player_profile": {"chapter": 2, "build": "dodge", "staff_level": 1,
                           "unlocked_skills": [], "unlocked_spells": [], "unlocked_transformations": []},
        "ideal_answer_includes": ["第三章", "小西天", "观音禅院"],
        "must_not_include_spoiler_after_chapter": 2,
        "ground_truth_move_name": None,
        "source": "manual",
    },
    {
        "query": "闪身流怎么加点？",
        "category": "decision_making",
        "player_profile": {"chapter": 1, "build": "dodge", "staff_level": 1,
                           "unlocked_skills": [], "unlocked_spells": [], "unlocked_transformations": []},
        "ideal_answer_includes": ["闪避", "完美闪避", "身法"],
        "must_not_include_spoiler_after_chapter": 1,
        "ground_truth_move_name": None,
        "source": "manual",
    },
]


def main():
    # TODO: auto-extract more queries from NGA 求助 post titles
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for qa in SEED_QA:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"Wrote {len(SEED_QA)} eval items to {OUTPUT}")


if __name__ == "__main__":
    main()
