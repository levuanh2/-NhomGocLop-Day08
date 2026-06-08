"""Mock responses used while the RAG pipeline is not connected."""

from __future__ import annotations


EXAMPLE_QUESTIONS = [
    "Dieu kien de duoc huong an treo la gi?",
    "Toi cuop giat tai san bi phat tu bao nhieu nam?",
    "Nguoi lao dong nghi viec co duoc nhan tro cap thoi viec khong?",
    "Vu viec dua tin sai su that tren mang co the bi xu ly nhu the nao?",
]


MOCK_CHUNKS = [
    {
        "content": (
            "Nguoi bi xu phat tu co the duoc huong an treo khi dap ung cac dieu kien ve "
            "muc hinh phat, nhan than, tinh tiet giam nhe va xet thay khong can bat chap "
            "hanh hinh phat tu."
        ),
        "score": 0.94,
        "metadata": {
            "source": "bo_luat_hinh_su_2015.md",
            "type": "legal",
            "chunk_index": 12,
            "url": "",
            "title": "Bo luat Hinh su 2015, sua doi 2017",
            "date": "2015-11-27",
            "article_id": "Dieu 65",
        },
        "source": "hybrid",
    },
    {
        "content": (
            "Hoi dong Tham phan Toa an nhan dan toi cao huong dan viec ap dung an treo, "
            "trong do nhan manh dieu kien ve noi cu tru ro rang va kha nang tu cai tao."
        ),
        "score": 0.88,
        "metadata": {
            "source": "nghi_quyet_02_2018_nq_hdttp.md",
            "type": "legal",
            "chunk_index": 4,
            "url": "",
            "title": "Nghi quyet 02/2018/NQ-HDTP ve an treo",
            "date": "2018-05-15",
            "article_id": "Dieu 2",
        },
        "source": "pageindex",
    },
    {
        "content": (
            "Mot so vu an duoc toa xem xet cho huong an treo khi bi cao co nhieu tinh tiet "
            "giam nhe, boi thuong thiet hai va co noi cu tru on dinh."
        ),
        "score": 0.79,
        "metadata": {
            "source": "bao_phap_luat_demo.md",
            "type": "news",
            "chunk_index": 2,
            "url": "https://example.com/vu-an-an-treo-demo",
            "title": "Toa xem xet cac dieu kien khi cho bi cao huong an treo",
            "date": "2024-03-15",
            "article_id": "news_demo_001",
        },
        "source": "hybrid",
    },
]


def mock_chat(query: str, history: list[dict] | None = None) -> dict:
    lowered = query.lower()
    if any(word in lowered for word in ["thoi tiet", "bong da", "nau an", "du lich"]):
        return {
            "answer": (
                "Cau hoi nam ngoai pham vi ho tro.\n\n"
                "LegalBot chi ho tro cac cau hoi ve phap luat Viet Nam hoac vu viec vi pham "
                "phap luat duoc dang tren bao chi."
            ),
            "chunks": [],
            "out_of_scope": True,
            "query": query,
        }

    return {
        "answer": (
            "Theo quy dinh phap luat Viet Nam, mot nguoi co the duoc xem xet **huong an "
            "treo** khi dap ung dong thoi cac dieu kien chinh sau:\n\n"
            "1. Bi xu phat tu khong qua 03 nam.\n"
            "2. Co nhan than tot va co nhieu tinh tiet giam nhe.\n"
            "3. Co noi cu tru ro rang hoac noi lam viec on dinh.\n"
            "4. Toa an xet thay khong can bat chap hanh hinh phat tu va viec cho huong an "
            "treo khong gay nguy hiem cho xa hoi [1][2].\n\n"
            "Trong thuc tien, toa an con xem xet boi canh vu viec, hau qua, thai do khac "
            "phuc va nguy co tai pham cua nguoi bi ket an [3]."
        ),
        "chunks": MOCK_CHUNKS,
        "out_of_scope": False,
        "query": query,
    }
