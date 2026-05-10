"""
One-time script: scrape boss pages from the Black Myth: Wukong BWIKI
into data/raw/wiki/<boss>.json.

The crawler auto-discovers pages from the BWIKI index pages for 妖王 and 头目,
then fetches each detail page through the MediaWiki parse API.

Run:
    python scripts/crawl_bwiki.py
    python scripts/crawl_bwiki.py --limit 5
    python scripts/crawl_bwiki.py --force
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup, Tag

OUTPUT_DIR = Path("data/raw/wiki")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://wiki.biligame.com/wukong"
API_URL = f"{BASE_URL}/api.php"
RAW_PAGE_URL = f"{BASE_URL}/index.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (research project; educational use)"}

INDEX_PAGES = [
    ("妖王", "妖王"),
    ("头目", "头目"),
]
SOURCE_WIKIS = [f"{BASE_URL}/{quote(page_title)}" for page_title, _ in INDEX_PAGES]

CHAPTER_SECTION_ORDER = {
    "火照黑云": 1,
    "风起黄昏": 2,
    "夜生白露": 3,
    "曲度紫鸳": 4,
    "日落红尘": 5,
    "未竟": 6,
}

LOCATION_TO_CHAPTER = {
    "黑风山": 1,
    "苍狼林": 1,
    "翠竹林": 1,
    "观音禅院": 1,
    "黄风岭": 2,
    "卧虎寺": 2,
    "挟魂崖": 2,
    "沙门村": 2,
    "小西天": 3,
    "浮屠界": 3,
    "苦海": 3,
    "极乐谷": 3,
    "盘丝岭": 4,
    "盘丝洞": 4,
    "黄花观": 4,
    "紫云山": 4,
    "火焰山": 5,
    "丹灶谷": 5,
    "碧水洞": 5,
    "花果山": 6,
    "水帘洞": 6,
}

DISCOVERY_NAME_PATTERN = re.compile(r"\|名称=([^|\n}]+)")
INDEX_SECTION_PATTERN = re.compile(r"^\|([^=\n]+)=\s*$")
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')
SKIP_SECTION_NAMES = {"目录", "影神图", "背景", "杂谈", "参考", "相关链接", "台词", "图鉴"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Black Myth BWIKI boss pages.")
    parser.add_argument("--limit", type=int, default=None, help="Only scrape the first N discovered pages.")
    parser.add_argument("--delay", type=float, default=0.5, help="Sleep between pages to avoid hammering the site.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing JSON files.")
    return parser.parse_args()


def discover_boss_pages(session: requests.Session | None = None) -> list[dict[str, object]]:
    session = session or requests.Session()
    discovered: dict[str, dict[str, object]] = {}

    for index_title, page_kind in INDEX_PAGES:
        raw_text = _fetch_raw_page(index_title, session=session)
        for page_info in _iter_index_entries(raw_text, page_kind):
            discovered.setdefault(page_info["boss_name"], page_info)

    return list(discovered.values())


def _iter_index_entries(raw_text: str, page_kind: str) -> Iterable[dict[str, object]]:
    current_chapter: int | None = None

    for line in raw_text.splitlines():
        stripped = line.strip()
        section_match = INDEX_SECTION_PATTERN.match(stripped)
        if section_match:
            section_name = _strip_section_count(section_match.group(1))
            current_chapter = CHAPTER_SECTION_ORDER.get(section_name)
            continue

        name_match = DISCOVERY_NAME_PATTERN.search(stripped)
        if name_match is None:
            continue

        page_title = _normalize_page_title(name_match.group(1))
        boss_name = _normalize_entity_name(page_title)
        if not boss_name:
            continue

        yield {
            "page_title": page_title,
            "boss_name": boss_name,
            "page_kind": page_kind,
            "chapter": current_chapter,
        }


def scrape_boss(page: dict[str, object], session: requests.Session | None = None) -> dict:
    session = session or requests.Session()
    page_title = str(page["page_title"])
    html = _fetch_parsed_html(page_title, session=session)

    soup = BeautifulSoup(html, "lxml")
    content_root = _find_content_root(soup)
    _strip_noise(content_root)

    infobox = _extract_infobox(content_root)
    sections = _extract_sections(content_root)
    moves = _extract_moves(content_root)
    related_links = _extract_related_links(content_root)
    summary = _extract_summary(content_root)

    chapter = int(page.get("chapter") or 0) or _infer_chapter(infobox, sections, summary)
    page_kind = _normalize_text(str(infobox.get("分类", "") or page.get("page_kind", "")))
    lore_text = _join_non_empty(
        sections.get("影神图", ""),
        sections.get("背景", ""),
        sections.get("杂谈", ""),
    )
    strategy_titles = [
        item["title"]
        for item in related_links
        if item["section"] in {"招式拆解", "打法攻略"}
    ]
    strategy_text = "；".join(strategy_titles)

    raw_text = _build_raw_text(
        boss_name=str(page["boss_name"]),
        page_kind=page_kind,
        chapter=chapter,
        summary=summary,
        infobox=infobox,
        sections=sections,
        moves=moves,
    )

    return {
        "boss_name": page["boss_name"],
        "page_title": page_title,
        "page_kind": page_kind,
        "slug": quote(page_title),
        "url": f"{BASE_URL}/{quote(page_title)}",
        "source": "wiki",
        "source_site": "bwiki",
        "source_wikis": SOURCE_WIKIS,
        "chapter": chapter,
        "title": page_title,
        "summary": summary,
        "infobox": infobox,
        "location_text": sections.get("位置", ""),
        "combat_info": _collect_gameplay_sections(sections),
        "rewards_text": sections.get("击败奖励", ""),
        "strategy_text": strategy_text,
        "lore_text": lore_text,
        "moves": moves,
        "tips": strategy_text,
        "sections": sections,
        "related_links": related_links,
        "raw_text": raw_text,
    }


def _fetch_raw_page(page_title: str, session: requests.Session) -> str:
    resp = _request_with_retry(
        session,
        RAW_PAGE_URL,
        params={"title": page_title, "action": "raw"},
    )
    resp.encoding = resp.encoding or "utf-8"
    return resp.text


def _fetch_parsed_html(page_title: str, session: requests.Session) -> str:
    resp = _request_with_retry(
        session,
        API_URL,
        params={
            "action": "parse",
            "page": page_title,
            "prop": "text",
            "format": "json",
            "formatversion": 2,
        },
    )
    payload = resp.json()
    if "error" in payload:
        raise ValueError(payload["error"].get("info", f"parse failed for {page_title}"))
    return payload["parse"]["text"]


def _request_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, object],
    max_attempts: int = 3,
    timeout: int = 30,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=timeout,
            )
            if resp.status_code >= 500:
                resp.raise_for_status()
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
            time.sleep(1.5 * attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"request failed without a captured exception: {url}")


def _find_content_root(soup: BeautifulSoup) -> Tag:
    node = soup.select_one(".mw-parser-output")
    if isinstance(node, Tag):
        return node
    if soup.body is None:
        raise ValueError("Unable to locate BWIKI content root.")
    return soup.body


def _strip_noise(content_root: Tag) -> None:
    for selector in [
        "style",
        "script",
        "noscript",
        ".wkbreadcrumb",
        "#toc",
        ".noprint",
        ".mw-editsection",
        ".errorbox",
        ".thumbcaption",
    ]:
        for node in content_root.select(selector):
            node.decompose()


def _extract_infobox(content_root: Tag) -> dict[str, str]:
    infobox: dict[str, str] = {}
    box = content_root.select_one(".wk-infobox")
    if not isinstance(box, Tag):
        return infobox

    poem = box.select_one(".wk-infobox-shi")
    if isinstance(poem, Tag):
        poem_text = _normalize_text(poem.get_text(" ", strip=True))
        if poem_text:
            infobox["诗句"] = poem_text

    for row in box.select(".wk-2row"):
        left = row.select_one(".ib-l")
        right = row.select_one(".ib-r")
        if not isinstance(left, Tag) or not isinstance(right, Tag):
            continue
        key = _normalize_text(left.get_text(" ", strip=True))
        value = _normalize_text(right.get_text(" ", strip=True))
        if key and value:
            infobox[key] = value

    return infobox


def _extract_summary(content_root: Tag) -> str:
    for child in content_root.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "h2":
            break
        if child.name != "p":
            continue
        text = _normalize_text(child.get_text(" ", strip=True))
        if text:
            return text
    return ""


def _extract_sections(content_root: Tag) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_heading: str | None = None
    parts: list[str] = []

    for child in content_root.children:
        if not isinstance(child, Tag):
            continue

        if child.name == "h2":
            if current_heading is not None and parts:
                sections[current_heading] = "\n".join(parts).strip()
            current_heading = _normalize_heading(child.get_text(" ", strip=True))
            parts = []
            continue

        if current_heading is None:
            continue

        if child.name == "h3":
            subheading = _normalize_heading(child.get_text(" ", strip=True))
            if subheading:
                parts.append(f"{subheading}：")
            continue

        text = _normalize_text(child.get_text("\n", strip=True))
        if text:
            parts.append(text)

    if current_heading is not None and parts:
        sections[current_heading] = "\n".join(parts).strip()

    return sections


def _extract_moves(content_root: Tag) -> list[dict[str, str]]:
    move_heading = _find_heading(content_root, "招式")
    if move_heading is None:
        return []

    moves: list[dict[str, str]] = []
    for sibling in move_heading.next_siblings:
        if isinstance(sibling, Tag) and sibling.name == "h2":
            break
        if not isinstance(sibling, Tag):
            continue
        if "div-box" not in (sibling.get("class") or []):
            continue
        move = _parse_move_box(sibling)
        if move is not None:
            moves.append(move)
    return moves


def _find_heading(content_root: Tag, title: str) -> Tag | None:
    for heading in content_root.find_all("h2"):
        normalized = _normalize_heading(heading.get_text(" ", strip=True))
        if normalized == title:
            return heading
    return None


def _parse_move_box(node: Tag) -> dict[str, str] | None:
    skill = node.select_one(".infobox-bossskill")
    if not isinstance(skill, Tag):
        return None

    name_nodes = skill.select(".bossskill-name")
    block_nodes = skill.select(".bossskill-blocks")
    if not name_nodes:
        return None

    primary_name = ""
    parts: list[str] = []
    for index, name_node in enumerate(name_nodes):
        label = _normalize_text(name_node.get_text(" ", strip=True))
        block_text = ""
        if index < len(block_nodes):
            block_text = _normalize_text(block_nodes[index].get_text(" ", strip=True))

        if index == 0 and label:
            primary_name = label

        if not block_text:
            continue

        if index == 0 or not label:
            parts.append(block_text)
        else:
            parts.append(f"{label}：{block_text}")

    if not primary_name:
        return None

    return {
        "name": primary_name,
        "description": " ".join(parts).strip(),
    }


def _extract_related_links(content_root: Tag) -> list[dict[str, str]]:
    section_heading = _find_heading(content_root, "相关链接")
    if section_heading is None:
        return []

    related_links: list[dict[str, str]] = []
    current_section = "相关链接"

    for sibling in section_heading.next_siblings:
        if isinstance(sibling, Tag) and sibling.name == "h2":
            break
        if not isinstance(sibling, Tag):
            continue

        if sibling.name == "h3":
            current_section = _normalize_heading(sibling.get_text(" ", strip=True)) or current_section
            continue

        for link in sibling.find_all("a", href=True):
            title = _normalize_text(link.get_text(" ", strip=True))
            href = link["href"].strip()
            if not title or not href:
                continue
            if href.startswith("//"):
                href = f"https:{href}"
            elif href.startswith("/"):
                href = f"https://wiki.biligame.com{href}"
            if not href.startswith("http"):
                continue
            related_links.append({
                "section": current_section,
                "title": title,
                "url": href,
            })

    return related_links


def _collect_gameplay_sections(sections: dict[str, str]) -> str:
    parts = []
    for name, text in sections.items():
        if not text or name in SKIP_SECTION_NAMES or name == "招式":
            continue
        if any(keyword in name for keyword in ("位置", "耐性", "奖励", "打法", "机制", "属性", "技能")):
            parts.append(f"{name}：{text}")
    return "\n".join(parts)


def _build_raw_text(
    boss_name: str,
    page_kind: str,
    chapter: int | None,
    summary: str,
    infobox: dict[str, str],
    sections: dict[str, str],
    moves: list[dict[str, str]],
) -> str:
    parts: list[str] = [f"Boss：{boss_name}。"]

    if page_kind:
        parts.append(f"分类：{page_kind}。")
    if chapter:
        parts.append(f"章节：第{chapter}章。")

    infobox_parts = []
    for key in ("地点", "生命", "分类", "名称"):
        value = infobox.get(key, "")
        if value:
            infobox_parts.append(f"{key}：{value}")
    if infobox_parts:
        parts.append("；".join(infobox_parts) + "。")

    if summary:
        parts.append(f"简介：{summary}")

    for name, text in sections.items():
        if not text or name in SKIP_SECTION_NAMES or name == "招式":
            continue
        if any(keyword in name for keyword in ("位置", "耐性", "奖励", "打法", "机制", "属性", "技能")):
            parts.append(f"{name}：{text}")

    for move in moves:
        name = move.get("name", "")
        description = move.get("description", "")
        if name and description:
            parts.append(f"招式 {name}：{description}")

    return _normalize_text("\n".join(parts))


def _infer_chapter(infobox: dict[str, str], sections: dict[str, str], summary: str) -> int:
    search_space = _join_non_empty(
        infobox.get("地点", ""),
        sections.get("位置", ""),
        summary,
    )
    for location, chapter in LOCATION_TO_CHAPTER.items():
        if location in search_space:
            return chapter
    return 1


def _strip_section_count(text: str) -> str:
    return re.sub(r"（\d+）$", "", text).strip()


def _normalize_heading(text: str) -> str:
    return _normalize_text(re.sub(r"\[\s*编辑\s*\]", "", text))


def _normalize_page_title(text: str) -> str:
    return _normalize_text(text)


def _normalize_entity_name(text: str) -> str:
    return _normalize_page_title(text).strip('“”"')


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _join_non_empty(*parts: str) -> str:
    return "\n".join(part for part in parts if part)


def _sanitize_filename(name: str) -> str:
    cleaned = INVALID_FILENAME_CHARS.sub("_", name).strip().rstrip(".")
    return cleaned or "wiki_page"


def main() -> None:
    args = parse_args()

    with requests.Session() as session:
        pages = discover_boss_pages(session=session)
        if args.limit is not None:
            pages = pages[: args.limit]

        print(f"discovered {len(pages)} boss pages from BWIKI indexes")
        for page in pages:
            out_path = OUTPUT_DIR / f"{_sanitize_filename(str(page['boss_name']))}.json"
            if out_path.exists() and not args.force:
                print(f"skip {page['boss_name']} (already scraped)")
                continue

            print(f"scraping {page['boss_name']} from BWIKI...")
            try:
                data = scrape_boss(page, session=session)
                out_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                print(f"  ERROR: {exc}")
            time.sleep(max(args.delay, 0.0))


if __name__ == "__main__":
    main()
