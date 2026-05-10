import importlib.util
from pathlib import Path

from bs4 import BeautifulSoup


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "crawl_bwiki.py"
SPEC = importlib.util.spec_from_file_location("crawl_bwiki", MODULE_PATH)
crawl_bwiki = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(crawl_bwiki)


def test_iter_index_entries_extracts_chapter_and_normalized_name():
    raw_text = """
|火照黑云（4）=
{{模板:妖魔展示|名称=广智|影神图=头目-广智.png}}
|风起黄昏（9）=
{{模板:妖魔展示|名称=“虎先锋”|影神图=头目-虎先锋.png}}
"""

    entries = list(crawl_bwiki._iter_index_entries(raw_text, "头目"))

    assert entries == [
        {
            "page_title": "广智",
            "boss_name": "广智",
            "page_kind": "头目",
            "chapter": 1,
        },
        {
            "page_title": "“虎先锋”",
            "boss_name": "虎先锋",
            "page_kind": "头目",
            "chapter": 2,
        },
    ]


def test_extract_infobox_and_moves_from_bwiki_markup():
    html = """
    <div class="mw-parser-output">
      <div class="wk-infobox">
        <div class="wk-2row"><div class="ib-l">名称</div><div class="ib-r">广智</div></div>
        <div class="wk-2row"><div class="ib-l">分类</div><div class="ib-r">头目</div></div>
        <div class="wk-2row"><div class="ib-l">地点</div><div class="ib-r">黑风山</div></div>
      </div>
      <p>广智是黑风山中的头目。</p>
      <h2>招式 [编辑]</h2>
      <div class="div-box">
        <div class="infobox-bossskill">
          <div class="bossskill-name bossskill-block-red">冲刺连劈</div>
          <div class="bossskill-blocks">起手蓄力后前冲三连攻击。</div>
          <div class="bossskill-name bossskill-block-blue">应对</div>
          <div class="bossskill-blocks">第二段需注意快慢刀。</div>
        </div>
      </div>
      <h2>参考</h2>
    </div>
    """

    soup = BeautifulSoup(html, "lxml")
    root = crawl_bwiki._find_content_root(soup)

    infobox = crawl_bwiki._extract_infobox(root)
    moves = crawl_bwiki._extract_moves(root)

    assert infobox["名称"] == "广智"
    assert infobox["地点"] == "黑风山"
    assert moves == [
        {
            "name": "冲刺连劈",
            "description": "起手蓄力后前冲三连攻击。 应对：第二段需注意快慢刀。",
        }
    ]


def test_build_raw_text_prefers_gameplay_sections_and_moves():
    raw_text = crawl_bwiki._build_raw_text(
        boss_name="广智",
        page_kind="头目",
        chapter=1,
        summary="广智是黑风山中的头目。",
        infobox={"地点": "黑风山", "生命": "1944"},
        sections={
            "位置": "位于苍狼林。",
            "击败奖励": "赤潮。",
            "背景": "很长的背景故事。",
        },
        moves=[{"name": "冲刺连劈", "description": "前冲后连续下劈。"}],
    )

    assert "章节：第1章。" in raw_text
    assert "位置：位于苍狼林。" in raw_text
    assert "击败奖励：赤潮。" in raw_text
    assert "招式 冲刺连劈：前冲后连续下劈。" in raw_text
    assert "背景" not in raw_text