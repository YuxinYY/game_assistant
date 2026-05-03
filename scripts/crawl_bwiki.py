"""
One-time script: scrape bwiki boss pages → data/raw/wiki/<boss>.json
Run: python scripts/crawl_bwiki.py
"""

import json
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

OUTPUT_DIR = Path("data/raw/wiki")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://wiki.biligame.com/wukong"
HEADERS = {"User-Agent": "Mozilla/5.0 (research project)"}

BOSS_LIST = [
    "虎先锋", "广智", "黄风大圣", "蜘蛛精",
    "百眼魔君", "红孩儿", "牛魔王",
]


def scrape_boss(name: str) -> dict:
    url = f"{BASE_URL}/{name}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # TODO: locate 招式 table in wiki HTML structure, extract rows
    moves = _parse_moves(soup)
    raw_text = soup.get_text(separator="\n", strip=True)

    return {
        "boss_name": name,
        "url": url,
        "source": "wiki",
        "chapter": _infer_chapter(name),
        "moves": moves,
        "raw_text": raw_text[:5000],  # truncate to avoid giant files
    }


def _parse_moves(soup: BeautifulSoup) -> list[dict]:
    # TODO: find the moves/招式 section in bwiki page structure
    return []


def _infer_chapter(boss_name: str) -> int:
    chapter_map = {
        "虎先锋": 1, "广智": 1, "黄风大圣": 2,
        "蜘蛛精": 4, "百眼魔君": 4,
        "红孩儿": 5, "牛魔王": 5,
    }
    return chapter_map.get(boss_name, 1)


def main():
    for name in BOSS_LIST:
        out_path = OUTPUT_DIR / f"{name}.json"
        if out_path.exists():
            print(f"skip {name} (already scraped)")
            continue
        print(f"scraping {name}...")
        try:
            data = scrape_boss(name)
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"  ERROR: {e}")
        time.sleep(1.5)


if __name__ == "__main__":
    main()
