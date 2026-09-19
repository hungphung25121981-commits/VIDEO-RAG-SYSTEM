# Phase 3 — Search & Generation

## 1. Mục Tiêu

Đây là "trái tim truy vấn" của hệ thống — nhận câu hỏi ngôn ngữ tự nhiên, tìm đúng ngữ cảnh liên quan nhất trong Qdrant + Kuzu Graph, rồi sinh câu trả lời có trích dẫn chính xác đến từng mốc thời gian.

**Input:** `--query [text]`, dữ liệu từ Qdrant + Kuzu (Phase 2).
**Output:** câu trả lời (nếu không dùng `-s`) + `trake` (mảng truy vết).

## 2. Sơ Đồ DAG Bất Biến (4 Trạm — Luôn Theo Đúng Thứ Tự)

```
                        Câu hỏi người dùng
                                │
                                ▼
                 ┌─────────────────────────┐
                 │  Micro-Router (DeBERTa) │
                 │  CPU — phân loại modal  │
                 └───────────┬─────────────┘
                             │
                 confidence < 0.4 cho mọi cờ?
                   │                  │
                  Có                 Không
                   │                  │
                   ▼                  ▼
           Bật CẢ 3 cờ              Chỉ bật cờ 
          (Fallback an toàn)        được chọn 
                    │                  │
                    └────────┬─────────┘
                             ▼
         TRẠM 1 — Quét diện rộng (tối đa 3 Collection: Semantic, Visual, Sparse)
                             │
                             ▼
         TRẠM 2 — Nhảy cóc qua Kuzu Graph (Cypher, hop_limit clamp [1,5])
                             │
                             ▼
         TRẠM 3 — Xếp hạng toàn cục (RRF, dùng α, β đã tune theo domain)
                             │
                             ▼
         TRẠM 4 — Cắt gọt (0/1 Knapsack DP, capacity = --max_context_images)
                             │
                             ▼
                  Qwen 2.5 VL (ghim GPU)
                  → Chain-of-Thought → Câu trả lời + trake
```

> **Nguyên tắc bất biến:** dù Router mở bao nhiêu cờ ở Trạm 1, hay Graph Hop mở rộng bao nhiêu candidate ở Trạm 2, số ảnh cuối cùng đưa vào VLM **không bao giờ vượt quá** `--max_context_images` — đảm bảo VRAM ổn định tuyệt đối.

## 3. Công Nghệ & Thuật Toán Áp Dụng

| Thành phần | Công nghệ | Ghi chú kỹ thuật |
|---|---|---|
| Routing | DeBERTa (CPU) | Multi-label Fallback: nếu độ tin cậy < 0.4 cho mọi cờ → mở toàn bộ 3 modal |
| Graph traversal | Kuzu Cypher, Parameterized Query | Chống Cypher Injection; `safe_hop = max(1, min(int(hop_limit), 5))` |
| Ranking | Tri-Search RRF | `S_total = α·S_semantic + β·S_visual + (1−α−β)·S_keyword` — 2 bậc tự do, grid search trên mặt phẳng (α, β) |
| Cut-off | 0/1 Knapsack (Dynamic Programming) | Weight = 1 (chunk đơn) hoặc 2 (chunk cặp vắt cảnh); Value = điểm RRF (cặp lấy Value từ sub-chunk match độc lập) |
| Rerank tùy chọn | Cross-Encoder (`-rr`) | Chạy CPU, chấm lại Top-K trước khi vào RRF |
| Generation | Qwen 2.5 VL (3B INT4) | Ghim cố định GPU trong suốt Phase 3, nhận `{type: image}` qua PIL.Image |

## 4. Công Thức Ranking Chi Tiết

```
S_total = α · S_semantic + β · S_visual + (1 − α − β) · S_keyword
```

`α, β` được tune và lưu theo domain (xem `--tune_weights` trong `backend/README.md`), đọc từ `config/settings.yaml`.

## 5. Thuật Toán Cut-off (0/1 Knapsack)

- **Capacity `W`** = `--max_context_images` (mặc định 4).
- **Item = Logical Chunk:**
  - Chunk đơn: `Weight = 1`, `Value = điểm RRF`.
  - Chunk cặp (vắt cảnh, có `parent_chunk_id`): `Weight = 2`, `Value = điểm RRF của sub-chunk được match độc lập`.
- DP đảm bảo dùng tối đa ngân sách `W`, tối ưu tổng Value — không bỏ sót slot như thuật toán greedy đơn thuần.

## 6. Cách Chạy Độc Lập

```bash
# Tìm kiếm dùng cả Graph và VLM
python cli_pipeline.py --run_phase 3 \
  --query "Nhân vật chính nói gì ở phút thứ 5?" \
  --use_graph --hop_limit 2

# Chỉ trích xuất bối cảnh (Context) dạng truy vết, không chạy VLM sinh văn bản
python cli_pipeline.py --run_phase 3 \
  --query "Find the explosion scene" -s -tr

# Truy vấn với rerank Cross-Encoder + giới hạn 4 ảnh vào VLM
python cli_pipeline.py --run_phase 3 \
  --query "Phân tích chiến thuật bù giờ" \
  --use_graph --hop_limit 2 -rr --max_context_images 4
```

**Output mẫu (JSON, `-s -tr`):**
```json
{
  "status": "success",
  "answer": null,
  "trake": [
    {"chunk_id": "c_0301", "timestamp": "00:04:58", "keyframe_path": "data/temp_workspace/kf_0301.jpg", "rrf_score": 0.812},
    {"chunk_id": "c_0302", "timestamp": "00:05:02", "keyframe_path": "data/temp_workspace/kf_0302.jpg", "rrf_score": 0.799}
  ]
}
```

## 7. Lưu Ý Vận Hành

- Nếu kết quả tìm kiếm quá thưa (ít candidate), kiểm tra `α, β` đã được tune cho đúng domain hiện tại chưa (`--save_domain`).
- Nếu Router thường xuyên rơi vào Fallback (mở cả 3 cờ), có thể cần fine-tune lại DeBERTa hoặc kiểm tra câu hỏi đầu vào có quá mơ hồ so với domain.
- `--hop_limit` cao (gần 5) làm tăng số candidate ở Trạm 2 nhưng **không** ảnh hưởng đến VRAM cuối cùng nhờ Trạm 4 (Knapsack) luôn cắt về đúng `--max_context_images`.

- Nếu `ocr_fallback_easyocr` chiếm tỷ lệ lớn bất thường, kiểm tra lại `variance_boundary` trong `gamma_matrix` — có thể domain hiện tại cần ngưỡng khác với mặc định.
- Nếu `rms_gate_skipped` gần bằng tổng số chunk (video gần như không có tiếng ồn môi trường), đây là hành vi bình thường — không phải lỗi.
- Video không có thoại (chỉ nhạc nền/tiếng động) vẫn xử lý đúng: `clean_speech.wav` sẽ gần như trống, Whisper trả về ít/không có timestamp, phần lớn tín hiệu dồn vào `background_noise.wav` cho CLAP phân loại.
