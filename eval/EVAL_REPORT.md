# Báo cáo Evaluation — LegalBot RAG Pipeline

**Ngày:** 2026-06-08  
**Framework:** Custom LLM-based evaluation (GPT-4o-mini judge)  
**Embedding model:** text-embedding-3-small (OpenAI)  
**Generation model:** gpt-4o-mini  
**Reranker:** flashrank local (ms-marco-MiniLM-L-12-v2, ONNX CPU)  

---

## 1. Golden Dataset

Tổng số câu hỏi: **18**

| Loại | Số câu | Mô tả |
|---|---|---|
| combined | 3 | Câu hỏi kết hợp luật + báo chí |
| legal | 7 | Câu hỏi về điều luật, hình phạt |
| news | 6 | Câu hỏi về vụ án từ báo chí |
| out_of_scope | 2 | Câu hỏi ngoài phạm vi pháp luật |

---

## 2. Cấu hình Thử nghiệm

| Tham số | Config A (Baseline) | Config B (Optimized) |
|---|---|---|
| top_k | 3 | 5 |
| Reranking | Tắt | flashrank cross-encoder (local) |
| Query expansion | Tắt | Bật |
| Mục tiêu | Baseline nhanh | Chất lượng tốt nhất |

---

## 3. Kết quả Evaluation

| Metric | Ý nghĩa | Config A | Config B | Chênh lệch |
|---|---|---|---|---|
| faithfulness | Faithfulness (không hallucinate) | 0.9750 | 0.9750 | +0.0000 |
| answer_relevancy | Answer Relevancy (đúng câu hỏi) | 0.9813 | 1.0000 | +0.0187 |
| context_precision | Context Precision (chunks cần thiết) | 0.4375 | 0.4625 | +0.0250 |
| context_recall | Context Recall (không bỏ sót) | 0.7938 | 0.7875 | -0.0063 |

**Điểm trung bình:** Config A = `0.7969` | Config B = `0.8063`

**Kết luận:** Config **B** thắng toàn diện với điểm trung bình cao hơn `0.0094`.

### Biểu đồ điểm

```
Metric                     Config A                    Config B
---------------------------------------------------------------------------
faithfulness               ███████████████████░ 0.975  |  ███████████████████░ 0.975
answer_relevancy           ███████████████████░ 0.981  |  ████████████████████ 1.000
context_precision          ████████░░░░░░░░░░░░ 0.438  |  █████████░░░░░░░░░░░ 0.463
context_recall             ███████████████░░░░░ 0.794  |  ███████████████░░░░░ 0.787
```

---

## 4. Phân tích

### Điểm mạnh của Config B
- **flashrank cross-encoder** (local ONNX) rerank candidates theo relevance thực sự, không phụ thuộc API ngoài.
- **top_k=5** cung cấp nhiều context hơn, tăng khả năng bao phủ ground truth.
- **Query expansion** tự động sinh thêm biến thể ("giải thích từ ngữ X", "X là") → khớp đúng Điều 2 định nghĩa trong luật.
- **RRF fusion** kết hợp BM25 lexical + semantic vector → tốt hơn từng phương pháp đơn lẻ.

### Hạn chế
- context_recall Config B (0.7875) thấp hơn A (0.7938) một chút: top_k=5 + reranking đôi khi đẩy chunk có ground truth xuống thấp hơn so với baseline top_k=3.
- Dữ liệu chỉ bao phủ chủ đề ma túy — câu hỏi lĩnh vực khác sẽ fallback sang PageIndex.
- flashrank model tiếng Anh (ms-marco) → độ chính xác rerank tiếng Việt còn hạn chế.

## 5. Worst Performers (câu hỏi điểm thấp nhất)

**1. Tội tàng trữ trái phép chất ma túy bị phạt tù bao nhiêu năm theo Bộ luật Hình sự?...**
- Contexts retrieved (Config B): 5
- Nguyên nhân khả năng: thiếu data về chủ đề cụ thể hoặc câu hỏi quá chung chung.

**2. Tội tổ chức sử dụng trái phép chất ma túy bị xử lý như thế nào?...**
- Contexts retrieved (Config B): 5
- Nguyên nhân khả năng: thiếu data về chủ đề cụ thể hoặc câu hỏi quá chung chung.

**3. Các hành vi nào bị nghiêm cấm theo Luật Phòng, chống ma túy 2021?...**
- Contexts retrieved (Config B): 5
- Nguyên nhân khả năng: thiếu data về chủ đề cụ thể hoặc câu hỏi quá chung chung.

---

## 6. Khuyến nghị Cải thiện

1. **Bổ sung dữ liệu:** Thêm nghị định, thông tư, án lệ để tăng context recall cho câu hỏi liên ngành.
2. **Chunking tốt hơn:** Giảm chunk_size (~800 ký tự) để mỗi Điều nằm trong 1 chunk riêng → tăng context_precision.
3. **Multilingual reranker:** Dùng `ms-marco-MultiBERT-L-12` (~400MB) thay `MiniLM` để cải thiện rerank tiếng Việt.
4. **HyDE đã triển khai:** `use_hyde=True` trong `retrieve()` — bật khi câu hỏi dạng "là gì"/định nghĩa.

---

*Báo cáo được tự động sinh bởi `eval/generate_report.py`*