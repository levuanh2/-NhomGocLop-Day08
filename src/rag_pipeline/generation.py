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
from pathlib import Path
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()
else:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

from .retrieval_pipeline import retrieve, _add_continuations


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# Giới hạn context trước khi gọi LLM. Fallback BM25 có thể lấy từ markdown thô,
# nên cần chặn kích thước để tránh lỗi context_length_exceeded.
MAX_CONTEXT_CHARS = 24000
MAX_CHUNK_CHARS = 2500


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý tư vấn pháp luật Việt Nam.

## Khi nào dùng "NGOÀI PHẠM VI:"
Chỉ bắt đầu bằng "NGOÀI PHẠM VI:" khi câu hỏi hoàn toàn không liên quan đến pháp luật — ví dụ: thời tiết, nấu ăn, thể thao, giải trí thuần túy, v.v.

Các câu hỏi SAU ĐÂY đều hợp lệ, KHÔNG dùng "NGOÀI PHẠM VI:":
- Hỏi về điều khoản, điều luật, hình phạt, tội danh trong bộ luật bất kỳ
- Hỏi về vụ việc vi phạm pháp luật của bất kỳ ai
- Hỏi kết hợp vụ việc cụ thể với quy định pháp luật (ví dụ: "tội của anh X theo bộ luật hình sự là gì?")

## Khi không tìm thấy trong context
Nếu câu hỏi hợp lệ nhưng context không có đủ thông tin, trả lời bình thường và ghi: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
KHÔNG dùng "NGOÀI PHẠM VI:" trong trường hợp này.

## Quy tắc trả lời
1. CHỈ dùng thông tin từ context. Mọi sự kiện phải có trích dẫn nguồn, ví dụ [Luật Phòng, chống ma túy, Điều 5] hoặc [VnExpress, 2026].
2. Liệt kê ĐẦY ĐỦ tất cả các mục khi câu hỏi yêu cầu — không rút gọn hay bỏ qua mục nào.
3. Tên người: PHẢI liệt kê từng tên một nếu có trong context. KHÔNG viết "X người khác".
4. Địa điểm và thời gian: ghi chính xác, không viết tắt.
5. KHÔNG suy diễn hay kết hợp thông tin từ các phần khác nhau của context để tạo ra sự kiện mới.
6. Cấu trúc câu trả lời rõ ràng bằng đoạn văn hoặc danh sách đánh số."""

_OUT_OF_SCOPE_PREFIX = "NGOÀI PHẠM VI:"


# =============================================================================
# SESSION NAMING
# =============================================================================

_SESSION_NAMING_PROMPT = """Dựa vào câu hỏi và câu trả lời bên dưới, hãy tạo một tên ngắn gọn cho đoạn hội thoại này.

Yêu cầu bắt buộc:
- Bắt đầu bằng một ĐỘNG TỪ (ví dụ: Phân tích, Tra cứu, Giải thích, Tìm hiểu, So sánh, Xác định...)
- Tiếp theo là DANH TỪ hoặc cụm danh từ mô tả chủ đề
- KHÔNG phải câu hỏi (không kết thúc bằng dấu ?)
- Ngắn gọn: 3–7 từ
- Trả về DUY NHẤT tên đó, không giải thích thêm

Ví dụ: Phân tích tội phạm ma túy | Tra cứu hình phạt tù chung thân | Xác định tội danh buôn bán chất cấm"""


def _generate_session_name(query: str, answer: str, client) -> str:
    """Tạo tên session từ câu hỏi và câu trả lời đầu tiên. Chỉ gọi 1 lần."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SESSION_NAMING_PROMPT},
            {"role": "user", "content": f"Câu hỏi: {query}\n\nCâu trả lời: {answer[:500]}"},
        ],
        temperature=0.4,
        max_tokens=30,
    )
    name = response.choices[0].message.content.strip()
    return name.rstrip("?")


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
            f"{chunk['content'][:MAX_CHUNK_CHARS]}"
        )
    return "\n\n---\n\n".join(context_parts)


def _trim_chunks_for_context(chunks: list[dict]) -> list[dict]:
    """Giữ context trong giới hạn ổn định trước khi đưa vào prompt."""
    trimmed: list[dict] = []
    used_chars = 0

    for chunk in chunks:
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue

        remaining = MAX_CONTEXT_CHARS - used_chars
        if remaining <= 0:
            break

        clipped = content[: min(MAX_CHUNK_CHARS, remaining)].strip()
        if not clipped:
            continue

        safe_chunk = chunk.copy()
        safe_chunk["content"] = clipped
        trimmed.append(safe_chunk)
        used_chars += len(clipped)

    return trimmed


# =============================================================================
# QUERY CONTEXTUALIZATION
# =============================================================================

_CONTEXTUALIZE_PROMPT = """Dựa vào lịch sử hội thoại bên dưới, hãy viết lại câu hỏi cuối cùng thành một câu hỏi độc lập, đầy đủ ngữ cảnh — có thể hiểu được mà không cần đọc lịch sử hội thoại.

Quy tắc:
- Thay thế đại từ ("cô ấy", "anh ta", "nó", "tội đó", "điều đó"...) bằng tên/sự việc cụ thể từ lịch sử.
- Nếu câu hỏi đã rõ ràng và không cần ngữ cảnh, trả về nguyên văn.
- Chỉ trả về câu hỏi đã viết lại, không giải thích thêm."""


def _contextualize_query(query: str, history: list[dict], client) -> str:
    """
    Viết lại query thành câu hỏi độc lập khi có history.
    Ví dụ: "cô ấy tội gì?" + history về Miu Lê → "Miu Lê bị buộc tội gì?"
    """
    if not history:
        return query

    history_text = "\n".join(
        f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content']}"
        for m in history[-6:]  # chỉ dùng 3 lượt gần nhất để tránh bloat
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _CONTEXTUALIZE_PROMPT},
            {"role": "user", "content": f"Lịch sử hội thoại:\n{history_text}\n\nCâu hỏi cần viết lại: {query}"},
        ],
        temperature=0,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(
    query: str,
    history: list[dict] | None = None,
    top_k: int = TOP_K,
) -> dict:
    """
    End-to-end RAG generation có citation và memory hội thoại.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build messages: system + history + user(context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query:   Câu hỏi của user
        history: Lịch sử hội thoại, list of {"role": "user"|"assistant", "content": str}
                 Chỉ chứa text thuần — KHÔNG kèm RAG context để tránh thổi phồng context window.

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if history is None:
        history = []

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Step 1: Viết lại query nếu có history (tránh đại từ mơ hồ như "cô ấy", "tội đó")
    retrieval_query = _contextualize_query(query, history, client)

    # Step 2: Retrieve + add continuation chunks
    chunks = retrieve(retrieval_query, top_k=top_k)
    chunks = _add_continuations(chunks)
    chunks = _trim_chunks_for_context(chunks)

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build messages
    # History (các lượt trước) đặt giữa system và câu hỏi hiện tại.
    # Câu hỏi hiện tại mang RAG context — các lượt trước không cần vì LLM
    # đã có câu trả lời cũ trong history để tham chiếu.
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + list(history)
        + [{"role": "user", "content": user_message}]
    )

    # Step 5: Call LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content
    out_of_scope = answer.strip().startswith(_OUT_OF_SCOPE_PREFIX)

    # Tạo tên session chỉ ở lượt đầu tiên (history rỗng)
    session_name = _generate_session_name(query, answer, client) if not history else None

    # Step 6: Return theo BotResponse schema
    used_chunks = reordered
    return {
        "answer": answer,
        "chunks": [] if out_of_scope else used_chunks,
        "out_of_scope": out_of_scope,
        "query": query,
        "session_name": session_name,
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
        print(f"\n[Sources: {len(result['chunks'])} chunks]")
