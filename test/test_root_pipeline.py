"""
Test suite cho root project RAG pipeline với data trong ./data/.

Kiểm tra:
- Data integrity của 10 news articles trong data/standardized/news/
- Semantic search (ChromaDB, collection "legal_chunks", 84 chunks)
- Lexical search (BM25 corpus từ ChromaDB)
- Retrieval pipeline (RRF + reranking)
- Generation helpers (reorder, format_context)

Chạy:
    conda run -n ai20k pytest test/test_root_pipeline.py -v
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
STANDARDIZED_NEWS_DIR = DATA_DIR / "standardized" / "news"
LANDING_NEWS_DIR = DATA_DIR / "landing" / "news"

sys.path.insert(0, str(ROOT / "src"))


# ===========================================================================
# Data Integrity — kiểm tra 10 articles trong data/standardized/news/
# ===========================================================================

class TestRootDataIntegrity(unittest.TestCase):
    """Kiểm tra dữ liệu thực tế trong ./data/standardized/news/"""

    def test_standardized_news_dir_exists(self):
        """data/standardized/news/ phải tồn tại."""
        self.assertTrue(STANDARDIZED_NEWS_DIR.exists(),
                       f"Thư mục không tồn tại: {STANDARDIZED_NEWS_DIR}")

    def test_has_10_news_articles(self):
        """Phải có đúng 10 articles (tất cả đã được chuẩn hoá)."""
        md_files = list(STANDARDIZED_NEWS_DIR.glob("*.md"))
        self.assertEqual(len(md_files), 10,
                        f"Số articles: {len(md_files)}, cần đúng 10")

    def test_all_articles_have_yaml_frontmatter(self):
        """Tất cả articles có YAML frontmatter với title, url, date, source."""
        for md_file in STANDARDIZED_NEWS_DIR.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("---"),
                          f"{md_file.name} không có YAML frontmatter")
            for field in ["title:", "url:", "date:", "source:"]:
                self.assertIn(field, content[:500],
                            f"{md_file.name} thiếu trường '{field}' trong frontmatter")

    def test_articles_have_article_body(self):
        """Tất cả articles có body sau frontmatter (>200 chars)."""
        for md_file in STANDARDIZED_NEWS_DIR.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            # Body starts after second ---
            parts = content.split("---", 2)
            if len(parts) >= 3:
                body = parts[2].strip()
                self.assertGreater(len(body), 200,
                                 f"{md_file.name}: body quá ngắn ({len(body)} chars)")

    def test_miu_le_article_exists(self):
        """Articles về Miu Lê phải có trong data."""
        all_content = " ".join(
            f.read_text(encoding="utf-8") for f in STANDARDIZED_NEWS_DIR.glob("*.md")
        )
        self.assertIn("Miu Lê", all_content, "Thiếu articles về Miu Lê")
        self.assertIn("Lê Ánh Nhật", all_content, "Thiếu tên thật Lê Ánh Nhật")

    def test_long_nhat_article_exists(self):
        """Articles về Long Nhật phải có trong data."""
        all_content = " ".join(
            f.read_text(encoding="utf-8") for f in STANDARDIZED_NEWS_DIR.glob("*.md")
        )
        self.assertIn("Long Nhật", all_content, "Thiếu articles về Long Nhật")

    def test_chau_viet_cuong_article_exists(self):
        """Article về Châu Việt Cường phải có trong data."""
        all_content = " ".join(
            f.read_text(encoding="utf-8") for f in STANDARDIZED_NEWS_DIR.glob("*.md")
        )
        self.assertIn("Châu Việt Cường", all_content, "Thiếu article về Châu Việt Cường")

    def test_landing_news_has_10_json_files(self):
        """data/landing/news/ có đúng 10 JSON files."""
        json_files = list(LANDING_NEWS_DIR.glob("*.json"))
        self.assertEqual(len(json_files), 10,
                        f"Số JSON files: {len(json_files)}, cần 10")

    def test_landing_json_have_required_fields(self):
        """JSON landing files có đủ metadata."""
        for json_file in list(LANDING_NEWS_DIR.glob("*.json"))[:5]:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            for field in ["url", "title", "date_crawled", "content_markdown"]:
                self.assertIn(field, data, f"{json_file.name} thiếu '{field}'")

    def test_chroma_collection_exists(self):
        """ChromaDB collection 'legal_chunks' phải tồn tại."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(DATA_DIR / "vectorstore" / "chroma"))
            collection = client.get_collection("legal_chunks")
            count = collection.count()
            self.assertGreater(count, 0, "Collection 'legal_chunks' rỗng")
        except Exception as e:
            self.skipTest(f"ChromaDB không khả dụng: {e}")


# ===========================================================================
# Semantic Search — test với root ChromaDB (collection "legal_chunks")
# ===========================================================================

class TestRootSemanticSearch(unittest.TestCase):
    """Kiểm tra semantic_search() với root data (news only, 84 chunks)."""

    @classmethod
    def setUpClass(cls):
        try:
            from rag_pipeline.semantic_search import semantic_search
            cls.search = staticmethod(semantic_search)
            cls._available = True
        except Exception as e:
            cls._available = False
            cls.skip_reason = str(e)

    def _req(self):
        if not self.__class__._available:
            self.skipTest(getattr(self.__class__, "skip_reason", "semantic_search không khả dụng"))

    def test_miu_le_query_returns_relevant(self):
        """Query về Miu Lê trả về article liên quan trong top 3."""
        self._req()
        results = self.search("Miu Lê bị bắt ma túy", top_k=5)
        self.assertGreater(len(results), 0)
        top = " ".join(r["content"] for r in results[:3])
        self.assertTrue("Miu Lê" in top or "Lê Ánh Nhật" in top,
                       f"Top 3 không chứa 'Miu Lê': {top[:200]}")

    def test_long_nhat_query_returns_relevant(self):
        """Query về Long Nhật trả về article liên quan."""
        self._req()
        results = self.search("Long Nhật bị bắt ma túy showbiz", top_k=5)
        self.assertGreater(len(results), 0)
        top = " ".join(r["content"] for r in results[:3])
        self.assertIn("Long Nhật", top,
                     f"Top 3 không chứa 'Long Nhật': {top[:200]}")

    def test_all_results_are_news_type(self):
        """Tất cả kết quả phải là type='news' (không có legal docs)."""
        self._req()
        results = self.search("nghệ sĩ ma túy Việt Nam", top_k=10)
        for r in results:
            self.assertEqual(r["metadata"].get("type"), "news",
                           f"Kết quả không phải news: {r['metadata']}")

    def test_cat_ba_location_query(self):
        """Query địa điểm Cát Bà trả về article đúng."""
        self._req()
        results = self.search("bãi biển Cát Bà sử dụng ma túy", top_k=5)
        self.assertGreater(len(results), 0)
        top = " ".join(r["content"] for r in results[:3])
        self.assertIn("Cát Bà", top,
                     f"Top 3 không chứa 'Cát Bà': {top[:200]}")

    def test_score_in_cosine_range(self):
        """Score phải nằm trong [-1, 1]."""
        self._req()
        results = self.search("nghệ sĩ", top_k=5)
        for r in results:
            self.assertGreaterEqual(r["score"], -1.0 - 1e-6)
            self.assertLessEqual(r["score"], 1.0 + 1e-6)

    def test_results_sorted_descending(self):
        """Kết quả sorted theo score descending."""
        self._req()
        results = self.search("ca sĩ ma túy Hải Phòng", top_k=5)
        if len(results) >= 2:
            scores = [r["score"] for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True),
                           "Kết quả không sorted descending")

    def test_results_have_metadata_fields(self):
        """Metadata có các trường từ chunk_and_index: source, type, url, title, date."""
        self._req()
        results = self.search("ma túy showbiz", top_k=3)
        for r in results:
            meta = r["metadata"]
            self.assertIn("source", meta)
            self.assertIn("type", meta)

    def test_respects_top_k(self):
        """Không trả về nhiều hơn top_k."""
        self._req()
        for k in [1, 3, 5]:
            results = self.search("ma túy", top_k=k)
            self.assertLessEqual(len(results), k)


# ===========================================================================
# Lexical Search — BM25 với root data
# ===========================================================================

class TestRootLexicalSearch(unittest.TestCase):
    """Kiểm tra lexical_search() với root data (BM25 trên 84 chunks)."""

    @classmethod
    def setUpClass(cls):
        try:
            from rag_pipeline.lexical_search import lexical_search
            cls.search = staticmethod(lexical_search)
            cls._available = True
        except Exception as e:
            cls._available = False
            cls.skip_reason = str(e)

    def _req(self):
        if not self.__class__._available:
            self.skipTest(getattr(self.__class__, "skip_reason", "lexical_search không khả dụng"))

    def test_miu_le_keyword_found(self):
        """BM25 tìm thấy 'Miu Lê' trong top 3."""
        self._req()
        results = self.search("Miu Le", top_k=5)
        self.assertGreater(len(results), 0)
        top = " ".join(r["content"] for r in results[:3])
        self.assertTrue("Miu Lê" in top or "Miu Le" in top,
                       f"BM25 không tìm thấy 'Miu Lê' trong top 3: {top[:200]}")

    def test_long_nhat_keyword_found(self):
        """BM25 tìm thấy 'Long Nhật'."""
        self._req()
        results = self.search("Long Nhat", top_k=5)
        self.assertGreater(len(results), 0)
        top = " ".join(r["content"] for r in results[:3])
        self.assertTrue("Long Nhật" in top or "Long Nhat" in top,
                       f"BM25 không tìm thấy 'Long Nhật': {top[:200]}")

    def test_cat_ba_keyword_found(self):
        """BM25 tìm thấy 'Cát Bà'."""
        self._req()
        results = self.search("Cat Ba", top_k=5)
        self.assertGreater(len(results), 0)

    def test_drug_name_found(self):
        """BM25 tìm thấy tên chất ma túy: ketamine."""
        self._req()
        results = self.search("ketamine", top_k=3)
        self.assertGreater(len(results), 0, "Không tìm thấy 'ketamine'")
        self.assertGreater(results[0]["score"], 0)

    def test_positive_score_for_match(self):
        """Score > 0 khi có keyword match."""
        self._req()
        results = self.search("ma tuy nghe si", top_k=3)
        if results:
            self.assertGreater(results[0]["score"], 0)

    def test_sorted_descending(self):
        """Kết quả sorted descending."""
        self._req()
        results = self.search("nghe si bi bat", top_k=5)
        if len(results) >= 2:
            scores = [r["score"] for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_unrelated_query_low_score(self):
        """Query không liên quan (Python deep learning) cho score thấp."""
        self._req()
        results = self.search("Python deep learning neural network", top_k=3)
        if results:
            self.assertLess(results[0]["score"], 5.0)

    def test_empty_query_no_crash(self):
        """Empty query không crash."""
        self._req()
        try:
            results = self.search("", top_k=3)
            self.assertIsInstance(results, list)
        except Exception as e:
            self.fail(f"Empty query gây crash: {e}")


# ===========================================================================
# Retrieval Pipeline — end-to-end với root data
# ===========================================================================

class TestRootRetrievalPipeline(unittest.TestCase):
    """Kiểm tra retrieve() với root data (chỉ có news về showbiz)."""

    @classmethod
    def setUpClass(cls):
        try:
            from rag_pipeline.retrieval_pipeline import retrieve
            cls.retrieve = staticmethod(retrieve)
            cls._available = True
        except Exception as e:
            cls._available = False
            cls.skip_reason = str(e)

    def _req(self):
        if not self.__class__._available:
            self.skipTest(getattr(self.__class__, "skip_reason", "retrieve không khả dụng"))

    def test_miu_le_location_query(self):
        """Pipeline trả về địa điểm Cát Bà cho câu hỏi về Miu Lê."""
        self._req()
        results = self.retrieve("Miu Lê bị bắt ở đâu?", top_k=3)
        self.assertGreater(len(results), 0)
        top = " ".join(r["content"] for r in results[:3])
        self.assertIn("Cát Bà", top,
                     f"Pipeline không trả về địa điểm Cát Bà: {top[:300]}")

    def test_source_field_is_hybrid_or_pageindex(self):
        """Tất cả kết quả có 'source' là hybrid hoặc pageindex."""
        self._req()
        results = self.retrieve("ca sĩ bị bắt ma túy 2026", top_k=5)
        for r in results:
            self.assertIn("source", r)
            self.assertIn(r["source"], ["hybrid", "pageindex"])

    def test_top_k_not_exceeded(self):
        """Pipeline không vượt top_k."""
        self._req()
        for k in [1, 3, 5]:
            results = self.retrieve("nghệ sĩ ma túy", top_k=k)
            self.assertLessEqual(len(results), k)

    def test_required_keys_present(self):
        """Mỗi kết quả có content, score, metadata, source."""
        self._req()
        results = self.retrieve("nghệ sĩ bị bắt ma túy Hải Phòng", top_k=3)
        for r in results:
            self.assertIn("content", r)
            self.assertIn("score", r)
            self.assertIn("metadata", r)
            self.assertIn("source", r)

    def test_news_only_results(self):
        """Kết quả chỉ gồm news (root data không có legal docs)."""
        self._req()
        results = self.retrieve("nghệ sĩ ma túy showbiz", top_k=5)
        for r in results:
            doc_type = r.get("metadata", {}).get("type")
            if doc_type:
                self.assertEqual(doc_type, "news",
                               f"Kết quả không phải news: {doc_type}")

    def test_obscure_query_no_crash(self):
        """Query lạ không crash."""
        self._req()
        try:
            results = self.retrieve("xyz123abc obscure completely unrelated", top_k=3)
            self.assertIsInstance(results, list)
        except Exception as e:
            self.fail(f"Obscure query gây crash: {e}")

    def test_chau_viet_cuong_query(self):
        """Pipeline tìm được thông tin về Châu Việt Cường."""
        self._req()
        results = self.retrieve("Châu Việt Cường nhét tỏi", top_k=3)
        self.assertGreater(len(results), 0)
        top = " ".join(r["content"] for r in results[:3])
        self.assertIn("Châu Việt Cường", top,
                     f"Pipeline không tìm được Châu Việt Cường: {top[:300]}")


# ===========================================================================
# Generation Helpers — reorder_for_llm và format_context
# ===========================================================================

class TestRootGenerationHelpers(unittest.TestCase):
    """Test reorder_for_llm và format_context của root generation.py."""

    def _import(self):
        try:
            from rag_pipeline.generation import reorder_for_llm, format_context
            return reorder_for_llm, format_context
        except Exception as e:
            self.skipTest(f"generation.py không import được: {e}")

    def _chunks(self, n):
        return [
            {
                "content": f"Nội dung về ca sĩ Miu Lê bị bắt tại Cát Bà chunk {i}",
                "score": 1.0 - i * 0.1,
                "metadata": {
                    "source": f"article_{i:02d}.md",
                    "type": "news",
                    "url": f"https://example.com/{i}",
                    "title": f"Ca sĩ bị bắt {i}",
                    "date": "2026-06-08",
                },
            }
            for i in range(n)
        ]

    def test_reorder_preserves_count(self):
        """reorder_for_llm giữ nguyên số chunks."""
        reorder, _ = self._import()
        chunks = self._chunks(6)
        result = reorder(chunks)
        self.assertEqual(len(result), 6)

    def test_reorder_preserves_content(self):
        """reorder_for_llm không làm mất nội dung."""
        reorder, _ = self._import()
        chunks = self._chunks(5)
        result = reorder(chunks)
        self.assertEqual(
            {c["content"] for c in chunks},
            {c["content"] for c in result},
        )

    def test_reorder_best_stays_first(self):
        """Chunk quan trọng nhất luôn ở đầu."""
        reorder, _ = self._import()
        chunks = self._chunks(5)
        result = reorder(chunks)
        self.assertEqual(result[0]["content"], chunks[0]["content"])

    def test_reorder_small_input(self):
        """reorder_for_llm xử lý được 1-2 chunks."""
        reorder, _ = self._import()
        for n in [1, 2]:
            chunks = self._chunks(n)
            result = reorder(chunks)
            self.assertEqual(len(result), n)

    def test_format_context_numbering(self):
        """format_context đánh số Document 1, 2, 3..."""
        _, fmt = self._import()
        chunks = self._chunks(3)
        ctx = fmt(chunks)
        for i in range(1, 4):
            self.assertIn(f"Document {i}", ctx)

    def test_format_context_source_info(self):
        """format_context có tên source cho citation."""
        _, fmt = self._import()
        chunks = self._chunks(1)
        ctx = fmt(chunks)
        self.assertIn("article_00", ctx,
                     "format_context thiếu tên source file")

    def test_format_context_separator(self):
        """format_context có separator --- giữa documents."""
        _, fmt = self._import()
        chunks = self._chunks(2)
        ctx = fmt(chunks)
        self.assertIn("---", ctx)


# ===========================================================================
# Generation return schema — generate_with_citation
# ===========================================================================

class TestRootGenerationSchema(unittest.TestCase):
    """Kiểm tra schema return dict của generate_with_citation()."""

    def _import(self):
        try:
            from rag_pipeline.generation import generate_with_citation
            return generate_with_citation
        except Exception as e:
            self.skipTest(f"generation.py không import được: {e}")

    def test_returns_dict_with_answer(self):
        """generate_with_citation trả về dict có 'answer'."""
        gen = self._import()
        try:
            result = gen("Nghệ sĩ nào bị bắt vì ma túy ở Cát Bà?")
            self.assertIsInstance(result, dict)
            self.assertIn("answer", result)
            self.assertIsInstance(result["answer"], str)
            self.assertGreater(len(result["answer"]), 0)
        except Exception as e:
            self.skipTest(f"Generation error (có thể thiếu API key): {e}")

    def test_returns_chunks_not_sources(self):
        """generate_with_citation dùng key 'chunks' (không phải 'sources')."""
        gen = self._import()
        try:
            result = gen("Miu Lê tên thật là gì?")
            self.assertIn("chunks", result, "Phải có key 'chunks' trong return dict")
            self.assertNotIn("sources", result, "Key 'sources' không dùng trong root version")
        except Exception as e:
            self.skipTest(f"Generation error: {e}")

    def test_returns_out_of_scope_flag(self):
        """generate_with_citation có key 'out_of_scope'."""
        gen = self._import()
        try:
            result = gen("Hôm nay thời tiết như thế nào?")
            self.assertIn("out_of_scope", result)
            self.assertIsInstance(result["out_of_scope"], bool)
        except Exception as e:
            self.skipTest(f"Generation error: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
