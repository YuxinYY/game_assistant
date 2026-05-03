"""
One-time script: scrape boss pages from the Black Myth Wukong Fextralife wiki
into data/raw/wiki/<boss>.json.

Run:
    python scripts/crawl_bwiki.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup, Tag

OUTPUT_DIR = Path("data/raw/wiki")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_WIKIS = [
    "https://blackmythwukong.wiki.fextralife.com/Black+Myth+Wukong+Wiki",
]
BASE_URL = "https://blackmythwukong.wiki.fextralife.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (research project; educational use)"}

BOSS_PAGES = [
    {"name_zh": "虎先锋", "name_en": "Tiger Vanguard", "slug": "Tiger+Vanguard"},
    {"name_zh": "广智", "name_en": "Guangzhi", "slug": "Guangzhi"},
    {"name_zh": "黄风大圣", "name_en": "Yellow Wind Sage", "slug": "Yellow+Wind+Sage"},
    {"name_zh": "蜘蛛精", "name_en": "The Second Sister", "slug": "The+Second+Sister"},
    {"name_zh": "百眼魔君", "name_en": "The Hundred-Eyed Daoist Master", "slug": "The+Hundred-Eyed+Daoist+Master"},
    {"name_zh": "红孩儿", "name_en": "Red Boy", "slug": "Red+Boy"},
    {"name_zh": "牛魔王", "name_en": "Bull King", "slug": "Bull+King"},
]

SECTION_ALIASES = {
    "location": [
        "where to find",
        "location",
    ],
    "combat_info": [
        "combat information",
    ],
    "rewards": [
        "rewards",
    ],
    "strategy": [
        "fight strategy",
        "best tips",
    ],
    "attacks": [
        "attacks & counters",
        "attacks and counters",
    ],
    "lore": [
        "lore, notes & other trivia",
        "lore",
        "notes",
        "other trivia",
    ],
}

CHAPTER_MAP = {
    "虎先锋": 2,
    "广智": 1,
    "黄风大圣": 2,
    "蜘蛛精": 4,
    "百眼魔君": 4,
    "红孩儿": 5,
    "牛魔王": 5,
}


def scrape_boss(page: dict[str, str], session: requests.Session | None = None) -> dict:
    session = session or requests.Session()
    url = f"{BASE_URL}/{page['slug']}"
    resp = session.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    content_root = _find_content_root(soup)
    title = _extract_title(soup)
    infobox = _extract_infobox(content_root)
    sections = _extract_sections(content_root)
    attacks = _parse_attack_table(content_root)

    raw_text = _normalize_text(content_root.get_text("\n", strip=True))

    return {
        "boss_name": page["name_zh"],
        "boss_name_en": page["name_en"],
        "slug": page["slug"],
        "url": url,
        "source": "wiki",
        "source_site": "fextralife",
        "source_wikis": SOURCE_WIKIS,
        "chapter": _infer_chapter(page["name_zh"]),
        "title": title,
        "summary": _extract_summary(content_root),
        "infobox": infobox,
        "location_text": sections.get("location", ""),
        "combat_info": sections.get("combat_info", ""),
        "rewards_text": sections.get("rewards", ""),
        "strategy_text": sections.get("strategy", ""),
        "lore_text": sections.get("lore", ""),
        "moves": attacks,
        "sections": sections,
        "raw_text": raw_text[:8000],
    }


def _find_content_root(soup: BeautifulSoup) -> Tag:
    for selector in [
        ".wiki-page-content",
        ".WikiaArticle",
        ".mw-parser-output",
        "#wiki-content-block",
        "#content",
    ]:
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            return node
    if soup.body is None:
        raise ValueError("Unable to locate page body for parsing.")
    return soup.body


def _extract_title(soup: BeautifulSoup) -> str:
    for selector in ["h1.page-header__title", "h1#firstHeading", "h1"]:
        node = soup.select_one(selector)
        if node:
            return _normalize_text(node.get_text(" ", strip=True))
    return ""


def _extract_summary(content_root: Tag) -> str:
    paragraphs: list[str] = []
    for node in content_root.find_all(["p", "blockquote"], recursive=True):
        text = _normalize_text(node.get_text(" ", strip=True))
        if not text:
            continue
        if len(text) < 30:
            continue
        paragraphs.append(text)
        if len(paragraphs) == 2:
            break
    return "\n".join(paragraphs)


def _extract_infobox(content_root: Tag) -> dict[str, str | list[str]]:
    infobox: dict[str, str | list[str]] = {}

    for table in content_root.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        for row in rows:
            cells = row.find_all(["th", "td"], recursive=False)
            if len(cells) != 2:
                continue
            key = _normalize_key(cells[0].get_text(" ", strip=True))
            values = _collect_list_items(cells[1])
            if not key:
                continue
            infobox[key] = values if len(values) > 1 else (values[0] if values else "")
        if infobox:
            return infobox

    block = content_root.find(["ul", "div"])
    if not isinstance(block, Tag):
        return infobox

    text = _normalize_text(block.get_text("\n", strip=True))
    for key in ["location", "drops", "type"]:
        match = re.search(rf"{key}\s*\|\s*(.+)", text, flags=re.IGNORECASE)
        if match:
            infobox[key] = match.group(1).strip()
    return infobox


def _extract_sections(content_root: Tag) -> dict[str, str]:
    sections: dict[str, str] = {}
    headings = content_root.find_all(re.compile(r"^h[2-4]$"))

    for heading in headings:
        title = _normalize_key(heading.get_text(" ", strip=True))
        if not title:
            continue
        canonical = _match_section(title)
        if canonical is None:
            continue
        sections[canonical] = _collect_section_text(heading)
    return sections


def _collect_section_text(heading: Tag) -> str:
    parts: list[str] = []
    for sibling in heading.next_siblings:
        if isinstance(sibling, Tag) and re.match(r"^h[2-4]$", sibling.name or ""):
            break
        if not isinstance(sibling, Tag):
            continue
        text = _normalize_text(sibling.get_text("\n", strip=True))
        if text:
            parts.append(text)
    return "\n".join(parts)


def _parse_attack_table(content_root: Tag) -> list[dict[str, str]]:
    heading = _find_section_heading(content_root, "attacks")
    if heading is None:
        return []

    for sibling in heading.next_siblings:
        if isinstance(sibling, Tag) and re.match(r"^h[2-4]$", sibling.name or ""):
            break
        if not isinstance(sibling, Tag):
            continue
        if sibling.name != "table":
            continue
        moves = _rows_from_table(sibling)
        if moves:
            return moves

    return []


def _match_section(title: str) -> str | None:
    for canonical, aliases in SECTION_ALIASES.items():
        if any(alias in title for alias in aliases):
            return canonical
    return None


def _find_section_heading(content_root: Tag, canonical_name: str) -> Tag | None:
    for heading in content_root.find_all(re.compile(r"^h[2-4]$")):
        title = _normalize_key(heading.get_text(" ", strip=True))
        if _match_section(title) == canonical_name:
            return heading
    return None


def _rows_from_table(table: Tag) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) != 3:
            continue
        values = [_normalize_text(cell.get_text(" ", strip=True)) for cell in cells]
        if not all(values):
            continue
        if values[0].lower() == "attack" and values[1].lower() == "description":
            continue
        rows.append(
            {
                "attack": values[0],
                "description": values[1],
                "counter": values[2],
            }
        )
    return rows


def _collect_list_items(node: Tag) -> list[str]:
    items = [
        _normalize_text(item.get_text(" ", strip=True))
        for item in node.find_all("li")
        if _normalize_text(item.get_text(" ", strip=True))
    ]
    if items:
        return items

    text = _normalize_text(node.get_text("\n", strip=True))
    if not text:
        return []
    return [part.strip() for part in text.split("\n") if part.strip()]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_key(text: str) -> str:
    text = _normalize_text(text).lower()
    text = text.replace("black myth wukong", "")
    return text.strip(" :-")


def _infer_chapter(boss_name: str) -> int:
    return CHAPTER_MAP.get(boss_name, 1)


def _iter_pages() -> Iterable[dict[str, str]]:
    seen: set[str] = set()
    for page in BOSS_PAGES:
        slug = page["slug"]
        if slug in seen:
            continue
        seen.add(slug)
        yield page


def main():
    with requests.Session() as session:
        for page in _iter_pages():
            out_path = OUTPUT_DIR / f"{page['name_zh']}.json"
            if out_path.exists():
                print(f"skip {page['name_zh']} (already scraped)")
                continue
            print(f"scraping {page['name_zh']} from Fextralife...")
            try:
                data = scrape_boss(page, session=session)
                out_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                print(f"  ERROR: {exc}")
            time.sleep(1.5)


if __name__ == "__main__":
    main()
