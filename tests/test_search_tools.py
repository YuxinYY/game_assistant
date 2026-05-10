from unittest.mock import patch

from src.tools.search import _get_wiki_entities, entity_lookup, wiki_search


class FakeRetriever:
    def __init__(self):
        self.calls = []
        self.bm25_documents = []

    def search(self, query, top_k, filters):
        self.calls.append((query, top_k, filters))
        return []


def test_wiki_search_filters_pure_english_queries_to_english_docs():
    retriever = FakeRetriever()
    _get_wiki_entities.cache_clear()

    with patch("src.retrieval.hybrid_retriever.get_retriever", return_value=retriever):
        wiki_search("How do I beat Tiger Vanguard?", top_k=3)

    assert retriever.calls == [
        (
            "How do I beat Tiger Vanguard?",
            3,
            {"source": "wiki", "language": "en"},
        )
    ]


def test_wiki_search_filters_pure_chinese_queries_to_chinese_docs():
    retriever = FakeRetriever()
    retriever.bm25_documents = [{"source": "wiki", "entity": "虎先锋"}]
    _get_wiki_entities.cache_clear()

    with patch("src.retrieval.hybrid_retriever.get_retriever", return_value=retriever):
        wiki_search("虎先锋怎么打？")

    assert retriever.calls[0][2] == {"source": "wiki", "language": "zh", "entity": "虎先锋"}


def test_wiki_search_keeps_mixed_language_queries_unfiltered():
    retriever = FakeRetriever()
    _get_wiki_entities.cache_clear()

    with patch("src.retrieval.hybrid_retriever.get_retriever", return_value=retriever):
        wiki_search("Tiger Vanguard 怎么打？")

    assert retriever.calls[0][2] == {"source": "wiki"}


def test_entity_lookup_prefers_direct_chunk_match_before_retrieval_search():
    retriever = FakeRetriever()
    _get_wiki_entities.cache_clear()
    retriever.bm25_documents = [
        {
            "text": "Boss：虎先锋。招式包括乌鸦坐飞机与血龙卷。",
            "source": "wiki",
            "url": "http://wiki/tiger-vanguard",
            "entity": "虎先锋",
            "metadata": {"language": "zh"},
        }
    ]

    with patch("src.retrieval.hybrid_retriever.get_retriever", return_value=retriever):
        payload = entity_lookup("虎先锋")

    assert payload == {
        "entity": "虎先锋",
        "text": "实体：虎先锋\n\nBoss：虎先锋。招式包括乌鸦坐飞机与血龙卷。",
        "url": "http://wiki/tiger-vanguard",
    }
    assert retriever.calls == []


def test_entity_lookup_uses_query_to_prefer_relevant_chunks():
    retriever = FakeRetriever()
    _get_wiki_entities.cache_clear()
    retriever.bm25_documents = [
        {
            "text": "招式 乌鸦坐飞机：开局跃起砸地。",
            "source": "wiki",
            "url": "http://wiki/tiger-vanguard",
            "entity": "虎先锋",
            "metadata": {"language": "zh"},
        },
        {
            "text": "招式 耗油跟：短暂蓄力后猛地砸下，需要注意蓄力节奏。",
            "source": "wiki",
            "url": "http://wiki/tiger-vanguard",
            "entity": "虎先锋",
            "metadata": {"language": "zh"},
        },
    ]

    with patch("src.retrieval.hybrid_retriever.get_retriever", return_value=retriever):
        payload = entity_lookup("虎先锋", query="虎先锋那个蓄力的大招怎么躲")

    assert "耗油跟" in payload["text"]
    assert "蓄力" in payload["text"]
    assert retriever.calls == []


def test_entity_lookup_count_queries_include_move_summary():
    retriever = FakeRetriever()
    _get_wiki_entities.cache_clear()
    retriever.bm25_documents = [
        {
            "text": "招式 乌鸦坐飞机：开局跃起砸地。 招式 后撤步7777：拉开距离。",
            "source": "wiki",
            "url": "http://wiki/tiger-vanguard",
            "entity": "虎先锋",
            "metadata": {"language": "zh"},
        },
        {
            "text": "招式 血龙卷：远程血气攻击。 招式 卧虎石：制造假身。",
            "source": "wiki",
            "url": "http://wiki/tiger-vanguard",
            "entity": "虎先锋",
            "metadata": {"language": "zh"},
        },
    ]

    with patch("src.retrieval.hybrid_retriever.get_retriever", return_value=retriever):
        payload = entity_lookup("虎先锋", query="虎先锋有几个大招")

    assert "当前 wiki 页面列出的招式条目数：4" in payload["text"]
    assert "命名质量警告" in payload["text"]
    assert "不宜直接当成严格官方大招数量" in payload["text"]
    assert "招式清单" not in payload["text"]


def test_entity_lookup_move_listing_queries_return_structured_names_without_raw_body():
    retriever = FakeRetriever()
    _get_wiki_entities.cache_clear()
    retriever.bm25_documents = [
        {
            "text": "招式 闪电旋风劈：虎先锋迅速拔刀三连斩。\"老子这三刀，超音速不是对手，蓝毒兽斩于虎下，今天就是老子起兵能源城之日，哈哈哈哈\"。 招式 卧虎石：制造假身。",
            "source": "wiki",
            "url": "http://wiki/tiger-vanguard",
            "entity": "虎先锋",
            "metadata": {"language": "zh"},
        }
    ]

    with patch("src.retrieval.hybrid_retriever.get_retriever", return_value=retriever):
        payload = entity_lookup("虎先锋", query="虎先锋的大招都叫什么")

    assert "较稳定的招式条目：卧虎石" in payload["text"]
    assert "已降权的可疑条目：闪电旋风劈" in payload["text"]
    assert "起兵能源城之日" not in payload["text"]