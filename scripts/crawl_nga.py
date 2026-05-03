"""
One-time script: scrape NGA posts for each boss → data/raw/nga/<boss>_posts.jsonl
Run: python scripts/crawl_nga.py
"""

import json
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

OUTPUT_DIR = Path("data/raw/nga")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://nga.178.com"
GAME_BOARD = "621"   # Black Myth: Wukong board ID
HEADERS = {"User-Agent": "Mozilla/5.0 (research project)"}

BOSS_LIST = ["虎先锋", "广智", "黄风大圣", "蜘蛛精", "百眼魔君", "红孩儿", "牛魔王"]


def search_posts(boss_name: str, page: int = 1) -> list[dict]:
    """Search NGA for posts about a boss. Returns list of post metadata."""
    # TODO: use NGA search API endpoint
    params = {
        "stid": GAME_BOARD,
        "searchword": boss_name,
        "page": page,
    }
    # resp = requests.get(f"{BASE_URL}/thread.php", params=params, headers=HEADERS)
    # soup = BeautifulSoup(resp.text, "lxml")
    # return _parse_thread_list(soup)
    raise NotImplementedError("NGA search not yet implemented")


def fetch_post(post_id: str) -> dict:
    """Fetch full post content."""
    # TODO: fetch thread page, extract main post + top replies
    raise NotImplementedError


def main():
    for boss in BOSS_LIST:
        out_path = OUTPUT_DIR / f"{boss}_posts.jsonl"
        if out_path.exists():
            print(f"skip {boss}")
            continue
        print(f"crawling NGA for {boss}...")
        # TODO: paginate through search results, fetch each post
        time.sleep(1.5)


if __name__ == "__main__":
    main()
