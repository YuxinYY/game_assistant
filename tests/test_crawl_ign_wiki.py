import importlib.util
from pathlib import Path

from bs4 import BeautifulSoup


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "crawl_ign_wiki.py"
SPEC = importlib.util.spec_from_file_location("crawl_ign_wiki", MODULE_PATH)
crawl_ign_wiki = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(crawl_ign_wiki)


def test_extract_discovery_entries_tracks_kind_chapter_and_aliases():
    html = """
    <div class="content-wrapper page-content">
      <div class="desktop-wiki-group">
        <div class="content">
          <h3>Yaoguai Chief Boss List</h3>
          <table>
            <tr><th>Name</th><th>Location</th><th>Reward</th><th>Checkbox</th></tr>
            <tr><td>Chapter 2 - Yellow Wind Ridge Yaoguai Chiefs</td></tr>
            <tr>
              <td><a href="/wikis/black-myth-wukong/Tiger_Vanguard">Tiger Vanguard</a></td>
              <td>Crouching Tiger Temple</td>
              <td>Rock Solid Spell</td>
              <td><a href="/wikis/black-myth-wukong/Tiger_Vanguard">Tiger Vanguard</a></td>
            </tr>
            <tr>
              <td><a href="/wikis/black-myth-wukong/Poison_Chiefs">Poison Chief 1</a></td>
              <td>Foothills</td>
              <td>Mind Core</td>
              <td><a href="/wikis/black-myth-wukong/Poison_Chiefs">Poison Chiefs</a></td>
            </tr>
            <tr>
              <td><a href="/wikis/black-myth-wukong/Poison_Chiefs">Poison Chief 2</a></td>
              <td>Foothills</td>
              <td>Mind Core</td>
              <td><a href="/wikis/black-myth-wukong/Poison_Chiefs">Poison Chiefs</a></td>
            </tr>
          </table>
        </div>
      </div>
    </div>
    """

    soup = BeautifulSoup(html, "lxml")
    content_root = crawl_ign_wiki._find_page_content_root(soup)

    entries = crawl_ign_wiki._extract_discovery_entries(content_root)

    assert entries == [
        {
            "page_title": "Poison Chiefs",
            "boss_name": "Poison Chiefs",
            "page_kind": "Yaoguai Chief",
            "chapter": 2,
            "url": "https://www.ign.com/wikis/black-myth-wukong/Poison_Chiefs",
            "aliases": ["Poison Chief 1", "Poison Chief 2"],
            "index_locations": ["Foothills"],
            "index_rewards": ["Mind Core"],
        },
        {
            "page_title": "Tiger Vanguard",
            "boss_name": "Tiger Vanguard",
            "page_kind": "Yaoguai Chief",
            "chapter": 2,
            "url": "https://www.ign.com/wikis/black-myth-wukong/Tiger_Vanguard",
            "aliases": ["Tiger Vanguard"],
            "index_locations": ["Crouching Tiger Temple"],
            "index_rewards": ["Rock Solid Spell"],
        },
    ]


def test_parse_boss_html_extracts_intro_sections_and_decodes_quotes():
    html = """
    <div class="content-wrapper page-content">
      <div class="page-header"><h1>Tiger Vanguard</h1></div>
      <div class="desktop-wiki-group">
        <div class="content">
          <section class="wiki-section wiki-html"><p>The Tiger Vanguard is a required Yaoguai King boss fight in Chapter 2.</p></section>
          <section class="wiki-section wiki-html"><ul><li><b>Rewards:</b> Rock Solid Spell, Keeness of Tiger</li></ul></section>
          <section class="wiki-section wiki-html"><h2><span>Where to Find the Tiger Vanguard</span></h2></section>
          <section class="wiki-section wiki-html"><p>You can find him inside the Crouching Tiger Temple.</p></section>
          <section class="wiki-section wiki-html"><h2><span>Tiger Vanguard Boss Fight and Guide</span></h2></section>
          <section class="wiki-section wiki-html"><p>You\\u0026#x2019;ll need to dodge the spinning kick at the last moment.</p></section>
        </div>
      </div>
    </div>
    """
    page = {
        "page_title": "Tiger Vanguard",
        "boss_name": "Tiger Vanguard",
        "page_kind": "Yaoguai King",
        "chapter": 2,
        "url": "https://www.ign.com/wikis/black-myth-wukong/Tiger_Vanguard",
        "aliases": ["Tiger Vanguard"],
        "index_locations": ["Crouching Tiger Temple"],
        "index_rewards": ["Rock Solid Spell"],
    }

    data = crawl_ign_wiki._parse_boss_html(html, page)

    assert data["boss_name"] == "Tiger Vanguard"
    assert data["source_site"] == "ign"
    assert data["source_language"] == "en"
    assert data["chapter"] == 2
    assert data["location_text"] == "You can find him inside the Crouching Tiger Temple."
    assert data["rewards_text"] == "Rock Solid Spell, Keeness of Tiger"
    assert "You'll need to dodge the spinning kick" in data["strategy_text"]
    assert "Boss: Tiger Vanguard." in data["raw_text"]