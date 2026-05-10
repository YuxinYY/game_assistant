"""
One-time script: scrape English boss pages from IGN's Black Myth: Wukong wiki
into data/raw/wiki/<boss>.json.

The crawler discovers boss guides from IGN's Boss_List_and_Guides index page,
then fetches each detail page and extracts retrieval-friendly English text.

Run:
    python scripts/crawl_ign_wiki.py
    python scripts/crawl_ign_wiki.py --limit 5
    python scripts/crawl_ign_wiki.py --force
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import unquote, urljoin

import requests
from bs4 import BeautifulSoup, Tag

OUTPUT_DIR = Path("data/raw/wiki")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.ign.com"
BOSS_LIST_URL = f"{BASE_URL}/wikis/black-myth-wukong/Boss_List_and_Guides"
HEADERS = {"User-Agent": "Mozilla/5.0 (research project; educational use)"}
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')
CHAPTER_PATTERN = re.compile(r"Chapter\s+(\d+)", re.IGNORECASE)

KIND_MAP = {
    "Yaoguai Chief Boss List": "Yaoguai Chief",
    "Yaoguai King List": "Yaoguai King",
    "Character Boss List": "Character Boss",
}

LOCATION_TO_CHAPTER = {
    "Black Wind Mountain": 1,
    "Forest of Wolves": 1,
    "Bamboo Grove": 1,
    "Black Wind Cave": 1,
    "Ancient Guanyin Temple": 1,
    "Yellow Wind Ridge": 2,
    "Sandgate Village": 2,
    "Fright Cliff": 2,
    "Crouching Tiger Temple": 2,
    "Kingdom of Sahali": 2,
    "The New West": 3,
    "Pagoda Realm": 3,
    "Valley of Ecstasy": 3,
    "Bitter Lake": 3,
    "New Thunderclap Temple": 3,
    "Snowhill Path": 3,
    "Warding Temple": 3,
    "Mahavira Hall": 3,
    "Webbed Hollow": 4,
    "Village of Lanxi": 4,
    "Temple of Yellow Flowers": 4,
    "Purple Cloud Mountain": 4,
    "The Flaming Mountains": 5,
    "Woods of Ember": 5,
    "Furnace Valley": 5,
    "Field of Fire": 5,
    "Bishui Cave": 5,
    "Mount Huaguo": 6,
    "Foothills": 6,
    "Water Curtain Cave": 6,
    "Mount Mei": 6,
    "Zodiac Village": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape English Black Myth wiki guides from IGN.")
    parser.add_argument("--limit", type=int, default=None, help="Only scrape the first N discovered pages.")
    parser.add_argument("--delay", type=float, default=0.5, help="Sleep between pages to avoid hammering the site.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing JSON files.")
    return parser.parse_args()


def discover_boss_pages(session: requests.Session | None = None) -> list[dict[str, object]]:
    session = session or requests.Session()
    html_text = _fetch_page_html(BOSS_LIST_URL, session=session)
    soup = BeautifulSoup(html_text, "lxml")
    content_root = _find_page_content_root(soup)
    return _extract_discovery_entries(content_root)


def _extract_discovery_entries(content_root: Tag) -> list[dict[str, object]]:
    discovered: dict[str, dict[str, object]] = {}

    for table in content_root.find_all("table"):
        heading = table.find_previous("h3")
        heading_text = _normalize_text(heading.get_text(" ", strip=True)) if isinstance(heading, Tag) else ""
        page_kind = KIND_MAP.get(heading_text)
        if not page_kind:
            continue

        current_chapter: int | None = None
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"], recursive=False)
            if not cells:
                continue

            row_text = _normalize_text(" ".join(cell.get_text(" ", strip=True) for cell in cells))
            chapter = _parse_chapter(row_text)
            if chapter is not None and len(cells) == 1:
                current_chapter = chapter
                continue

            boss_cell = _pick_boss_cell(cells)
            if boss_cell is None:
                continue

            href = _extract_detail_href(boss_cell)
            if not href:
                continue

            page_title = _href_to_page_title(href)
            if not page_title or _is_non_boss_page(page_title):
                continue

            location_text = _normalize_text(cells[1].get_text(" ", strip=True)) if len(cells) > 1 else ""
            reward_text = _normalize_text(cells[2].get_text(" ", strip=True)) if len(cells) > 2 else ""
            alias = _normalize_alias_name(boss_cell.get_text(" ", strip=True))

            page_info = discovered.setdefault(
                page_title,
                {
                    "page_title": page_title,
                    "boss_name": page_title,
                    "page_kind": page_kind,
                    "chapter": current_chapter,
                    "url": urljoin(BASE_URL, href),
                    "aliases": [],
                    "index_locations": [],
                    "index_rewards": [],
                },
            )

            if page_info.get("chapter") is None and current_chapter is not None:
                page_info["chapter"] = current_chapter
            if alias and alias not in page_info["aliases"]:
                page_info["aliases"].append(alias)
            if location_text and location_text not in page_info["index_locations"]:
                page_info["index_locations"].append(location_text)
            if reward_text and reward_text not in page_info["index_rewards"]:
                page_info["index_rewards"].append(reward_text)

    return sorted(
        discovered.values(),
        key=lambda item: (
            int(item.get("chapter") or 99),
            str(item.get("boss_name") or ""),
        ),
    )


def scrape_boss(page: dict[str, object], session: requests.Session | None = None) -> dict:
    session = session or requests.Session()
    html_text = _fetch_page_html(str(page["url"]), session=session)
    return _parse_boss_html(html_text, page)


def _parse_boss_html(html_text: str, page: dict[str, object]) -> dict:
    soup = BeautifulSoup(html_text, "lxml")
    page_content = _find_page_content_root(soup)
    article_root = _find_article_root(page_content)

    title = _extract_page_title(soup) or str(page.get("boss_name") or page.get("page_title") or "")
    intro_texts, sections = _extract_sections(article_root)
    summary = next((text for text in intro_texts if not text.startswith("Rewards:")), "")
    rewards_text = _extract_rewards(intro_texts) or " ; ".join(page.get("index_rewards") or [])
    location_text = _get_section_value(sections, prefix="Where to Find") or " ; ".join(page.get("index_locations") or [])
    strategy_text = _get_section_value(sections, suffix="Boss Fight and Guide")
    chapter = int(page.get("chapter") or 0) or _infer_chapter(
        summary,
        location_text,
        rewards_text,
        page.get("index_locations") or [],
        title,
    )
    aliases = [alias for alias in page.get("aliases") or [] if alias and alias != title]

    raw_text = _build_raw_text(
        boss_name=title,
        aliases=aliases,
        page_kind=str(page.get("page_kind") or ""),
        chapter=chapter,
        summary=summary,
        location_text=location_text,
        rewards_text=rewards_text,
        sections=sections,
    )

    return {
        "boss_name": title,
        "page_title": str(page.get("page_title") or title),
        "page_kind": str(page.get("page_kind") or ""),
        "slug": _page_title_to_slug(str(page.get("page_title") or title)),
        "url": page["url"],
        "source": "wiki",
        "source_site": "ign",
        "source_language": "en",
        "source_wikis": [BOSS_LIST_URL],
        "chapter": chapter,
        "title": title,
        "aliases": aliases,
        "summary": summary,
        "infobox": {
            "Type": str(page.get("page_kind") or ""),
            "Location": location_text,
            "Rewards": rewards_text,
        },
        "location_text": location_text,
        "combat_info": strategy_text,
        "rewards_text": rewards_text,
        "strategy_text": strategy_text,
        "lore_text": _get_section_value(sections, prefix="Lore"),
        "moves": [],
        "tips": strategy_text,
        "sections": sections,
        "related_links": [],
        "index_locations": page.get("index_locations") or [],
        "index_rewards": page.get("index_rewards") or [],
        "raw_text": raw_text,
    }


def _fetch_page_html(url: str, session: requests.Session) -> str:
    response = _request_with_retry(session, url)
    response.encoding = "utf-8"
    return response.text


def _request_with_retry(
    session: requests.Session,
    url: str,
    *,
    max_attempts: int = 3,
    timeout: int = 30,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, headers=HEADERS, timeout=timeout)
            if response.status_code >= 500:
                response.raise_for_status()
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
            time.sleep(1.5 * attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"request failed without captured exception: {url}")


def _find_page_content_root(soup: BeautifulSoup) -> Tag:
    node = soup.select_one(".content-wrapper.page-content")
    if isinstance(node, Tag):
        return node
    if soup.body is None:
        raise ValueError("Unable to locate IGN content root.")
    return soup.body


def _find_article_root(page_content: Tag) -> Tag:
    node = page_content.select_one(".desktop-wiki-group .content")
    if isinstance(node, Tag):
        return node
    return page_content


def _extract_page_title(soup: BeautifulSoup) -> str:
    node = soup.select_one(".page-header h1")
    if isinstance(node, Tag):
        return _normalize_text(node.get_text(" ", strip=True))
    return ""


def _extract_sections(article_root: Tag) -> tuple[list[str], dict[str, str]]:
    intro_texts: list[str] = []
    sections: dict[str, list[str]] = {}
    current_heading: str | None = None

    for section in _iter_article_sections(article_root):
        heading = section.find("h2")
        if isinstance(heading, Tag):
            current_heading = _normalize_text(heading.get_text(" ", strip=True))
            sections.setdefault(current_heading, [])
            continue

        text = _extract_section_text(section)
        if not text:
            continue
        if current_heading is None:
            intro_texts.append(text)
            continue
        sections[current_heading].append(text)

    return intro_texts, {
        heading: "\n".join(parts).strip()
        for heading, parts in sections.items()
        if heading and any(parts)
    }


def _iter_article_sections(article_root: Tag) -> list[Tag]:
    return [
        child
        for child in article_root.find_all("section", recursive=False)
        if isinstance(child, Tag) and "wiki-section" in (child.get("class") or [])
    ]


def _extract_section_text(section: Tag) -> str:
    text = _normalize_text(section.get_text(" ", strip=True))
    if text.startswith("Go To Map "):
        return ""
    return text


def _extract_rewards(intro_texts: list[str]) -> str:
    for text in intro_texts:
        if text.startswith("Rewards:"):
            return text[len("Rewards:") :].strip()
    return ""


def _get_section_value(sections: dict[str, str], prefix: str = "", suffix: str = "") -> str:
    for heading, value in sections.items():
        if prefix and heading.startswith(prefix):
            return value
        if suffix and heading.endswith(suffix):
            return value
    return ""


def _infer_chapter(
    summary: str,
    location_text: str,
    rewards_text: str,
    index_locations: list[str],
    title: str,
) -> int:
    for text in [summary, location_text, rewards_text, " ".join(index_locations), title]:
        chapter = _parse_chapter(text)
        if chapter is not None:
            return chapter
        for location_name, location_chapter in LOCATION_TO_CHAPTER.items():
            if location_name in text:
                return location_chapter
    return 1


def _build_raw_text(
    boss_name: str,
    aliases: list[str],
    page_kind: str,
    chapter: int | None,
    summary: str,
    location_text: str,
    rewards_text: str,
    sections: dict[str, str],
) -> str:
    parts = [f"Boss: {boss_name}."]
    if aliases:
        parts.append(f"Aliases: {'; '.join(aliases)}.")
    if page_kind:
        parts.append(f"Type: {page_kind}.")
    if chapter:
        parts.append(f"Chapter: {chapter}.")
    if summary:
        parts.append(f"Summary: {summary}")
    if location_text:
        parts.append(f"Location: {location_text}")
    if rewards_text:
        parts.append(f"Rewards: {rewards_text}")

    for heading, text in sections.items():
        if not text:
            continue
        parts.append(f"{heading}: {text}")

    return _normalize_text("\n".join(parts))


def _pick_boss_cell(cells: list[Tag]) -> Tag | None:
    if not cells:
        return None
    if len(cells) == 1:
        return None
    return cells[0]


def _extract_detail_href(cell: Tag) -> str:
    for link in cell.find_all("a", href=True):
        href = link["href"].strip()
        if "/wikis/black-myth-wukong/" in href:
            return href
    return ""


def _href_to_page_title(href: str) -> str:
    if "/wikis/black-myth-wukong/" not in href:
        return ""
    slug = href.split("/wikis/black-myth-wukong/", 1)[1].split("?", 1)[0].split("#", 1)[0]
    if not slug:
        return ""
    return _normalize_page_title(unquote(slug).replace("_", " "))


def _page_title_to_slug(page_title: str) -> str:
    return page_title.replace(" ", "_")


def _is_non_boss_page(page_title: str) -> bool:
    return page_title.endswith("Walkthrough") or page_title.endswith("Boss List") or page_title == "topcontributors"


def _parse_chapter(text: str) -> int | None:
    match = CHAPTER_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(1))


def _normalize_page_title(text: str) -> str:
    return _normalize_text(text).strip('"')


def _normalize_alias_name(text: str) -> str:
    cleaned = _normalize_text(text).replace(" *", "")
    return cleaned.strip('"')


def _normalize_text(text: str) -> str:
    cleaned = text.replace("\xa0", " ")
    if "\\u" in cleaned:
        try:
            cleaned = cleaned.encode("utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            pass
    cleaned = html.unescape(cleaned)
    cleaned = (
        cleaned.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _sanitize_filename(name: str) -> str:
    cleaned = INVALID_FILENAME_CHARS.sub("_", name).strip().rstrip(".")
    return cleaned or "ign_wiki_page"


def main() -> None:
    args = parse_args()

    with requests.Session() as session:
        pages = discover_boss_pages(session=session)
        if args.limit is not None:
            pages = pages[: args.limit]

        print(f"discovered {len(pages)} boss guide pages from IGN")
        for page in pages:
            out_path = OUTPUT_DIR / f"{_sanitize_filename(str(page['boss_name']))}.json"
            if out_path.exists() and not args.force:
                print(f"skip {page['boss_name']} (already scraped)")
                continue

            print(f"scraping {page['boss_name']} from IGN...")
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