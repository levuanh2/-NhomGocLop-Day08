# Test CLI — LegalBot RAG Pipeline

## Yêu cầu
- Conda environment: `ai20k`
- `OPENAI_API_KEY` trong file `LeTrungKien_2A202600834/.env`

## Chạy từ thư mục gốc project

### Chế độ Retrieve-only (không cần OpenAI cho generation)
Hiển thị các chunks được retrieve kèm score và nguồn.
```bash
PYTHONUTF8=1 /c/Users/Admin/miniconda3/envs/ai20k/python.exe test/cli_rag.py --retrieve
```

### Chế độ Full RAG (retrieve + generate với GPT-4o-mini)
Hiển thị chunks + câu trả lời có citation từ LLM.
```bash
PYTHONUTF8=1 /c/Users/Admin/miniconda3/envs/ai20k/python.exe test/cli_rag.py
```

## Ví dụ câu hỏi thử

| Loại | Câu hỏi |
|---|---|
| Hình phạt | `tội tàng trữ ma túy bị phạt tù bao nhiêu năm?` |
| Định nghĩa | `cai nghiện ma túy là gì?` |
| Sự kiện | `nghệ sĩ nào bị bắt vì ma túy?` |
| Out-of-scope | `thời tiết hôm nay thế nào?` |

## Ghi chú
- Jina API key không bắt buộc — reranker tự fallback sang score sort.
- `PYTHONUTF8=1` cần thiết trên Windows để hiển thị đúng tiếng Việt và emoji.
