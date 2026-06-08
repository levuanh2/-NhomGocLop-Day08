"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve, _add_continuations


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 7

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Trả lời câu hỏi dưới đây bằng tiếng Việt, dựa hoàn toàn vào ngữ cảnh được cung cấp.

Quy tắc bắt buộc:
1. CHỈ dùng thông tin từ context. Mọi sự kiện phải có trích dẫn nguồn ngay sau đó, ví dụ [Luật Phòng, chống ma túy, Điều 5] hoặc [VnExpress, 2026].
2. Liệt kê ĐẦY ĐỦ tất cả các mục khi câu hỏi yêu cầu (ví dụ: "những hành vi nào", "các nguồn nào", "những người nào") — không rút gọn hay bỏ qua mục nào. Liệt kê từng mục riêng biệt theo số thứ tự.
3. TUYỆT ĐỐI BẮT BUỘC — Tên người: Nếu context chứa câu như "Nhóm gồm: A (X tuổi), B (Y tuổi), C..." thì PHẢI liệt kê từng tên một. VÍ DỤ SAI: "bắt cùng 5 người khác". VÍ DỤ ĐÚNG: "bắt cùng 5 người: (1) Vũ Khương An, (2) Trần Minh Trang, (3) Vũ Tài Nam, (4) Trần Đức Phong, (5) Đoàn Thị Thúy An". KHÔNG bao giờ viết "X người khác" khi có tên trong context.
4. Địa điểm và thời gian: Ghi chính xác địa chỉ đầy đủ (thôn, đặc khu, thành phố), giờ giấc cụ thể có trong context — không được viết tắt hay lược bỏ.
5. KHÔNG suy diễn hay kết hợp thông tin từ các phần KHÁC nhau của context để tạo ra sự kiện mới. Ví dụ: nếu chất ma túy được tìm thấy khi KHÁM XÉT và chất ma túy được tìm thấy khi XÉT NGHIỆM CƠ THỂ là hai việc khác nhau, đừng trộn lẫn chúng.
6. Nếu context không chứa thông tin cần thiết, ghi rõ: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
7. Cấu trúc câu trả lời rõ ràng bằng đoạn văn hoặc danh sách đánh số."""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    # Even indices (0,2,4,...) → front; odd indices (1,3,...) → tail reversed
    first_half = chunks[::2]    # 0, 2, 4, ...
    second_half = chunks[1::2]  # 1, 3, 5, ...
    return first_half + second_half[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")
        # Strip extension for cleaner citation labels
        source_label = source.rsplit(".", 1)[0] if "." in source else source
        context_parts.append(
            f"[Document {i} | Source: {source_label} | Type: {doc_type}]\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve + add continuation chunks (chunk_index+1 per source)
    # Guarantees that content split across chunk boundaries is included
    # (e.g., article lists that continue in the next chunk).
    chunks = retrieve(query, top_k=top_k)
    chunks = _add_continuations(chunks)

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    # Step 5: Call LLM
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content

    # Step 6: Return
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
