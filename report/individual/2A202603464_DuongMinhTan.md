# Báo cáo cá nhân — Day 6: Data Ingestion Engineering

**Học viên:** Dương Minh Tân · **MSSV:** `2A202603464` · **Lớp:** `L3A`

---

## Dữ liệu và pipeline

### Chủ đề, nguồn/quyền, số trang và quy tắc tính trang
- **Chủ đề:** Phân tích kinh tế vĩ mô Việt Nam, thị trường vốn và chuyển đổi giao thông điện (e-mobility) qua hai ấn phẩm công khai cùng chuỗi *Taking Stock* của Ngân hàng Thế giới (World Bank).
- **Nguồn & giấy phép (`data/manifest.json`):**
  1. `wb_taking_stock_aug2024`: *Taking Stock August 2024: Reaching New Heights in Capital Markets* (`DOI: 10.1596/42084`, `https://hdl.handle.net/10986/42084`, lưu tại `starter/data/wb_taking_stock_aug2024.pdf`), giấy phép **CC BY-NC 3.0 IGO**. Gồm **64 trang vật lý**, **46 trang nội dung** được tính (`129,041` ký tự sau chuẩn hóa).
  2. `wb_taking_stock_mar2025`: *Taking Stock March 2025: Electrifying Journeys - E-mobility Transition in Viet Nam* (`Report ID: 099919303102515323`, `https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099919303102515323`, lưu tại `starter/data/wb_taking_stock_mar2025.pdf`), giấy phép **CC BY-NC 3.0 IGO**. Gồm **62 trang vật lý**, **48 trang nội dung** được tính.
- **Tổng số trang nội dung hợp lệ:** `46 + 48 = 94` trang (vượt ngưỡng tối thiểu 30 trang trong `README.md`).
- **Quy tắc tính trang & loại trừ:** Đánh số 1-indexed theo trang vật lý PDF; loại trừ tường minh trang bìa, trang trắng, trang tựa phụ, bản quyền, mục lục (`toc`), danh mục hình/bảng, bảng viết tắt, lời cảm ơn, bìa sau và các trang phân chương (`empty_or_divider`) có `< 150` ký tự sau chuẩn hóa. Toàn bộ quyết định từng trang được ghi vào `starter/results/ingestion_log.json`.

### Chuẩn hóa, lỗi và giới hạn
- **Chuẩn hóa văn bản (`starter/src/ingestion.py`):**
  - Chuẩn hóa Unicode `NFC`, khử ký tự null `\x00` và hyphen ngắt dòng cuối dòng, gom khoảng trắng liên tiếp nhưng **giữ ranh giới đoạn** (`\n\n`).
  - Loại bỏ running headers (`TAKING STOCK | ...`) và số trang đứng độc lập để tránh lặp rác vào vector chunk.
- **Lỗi runtime & fallback đã gặp (ghi đầy đủ trong `starter/results/provider_error_log.json`):**
  1. *Docker Desktop runtime (`2026-11-13T17:18:21+07:00`):* `com.docker.service` ở trạng thái dừng và không thể khởi động nếu thiếu UAC admin trên host -> Provision native **PostgreSQL 18.6 + pgvector 0.8.1** trong WSL2 Ubuntu lắng nghe trên `127.0.0.1:55433` với cùng thông tin `day6 / day6_lab`; giữ nguyên `starter/docker-compose.yml` để tái lập chuẩn qua Docker.
  2. *Gemini Embedding API (`2026-11-13T17:33:40+07:00`):* Thử nghiệm ban đầu với `gemini-embedding-001` đạt 560 chunk của A và 315/407 chunk của B trước khi chạm trần Free Tier **1,000 requests/day (`429 RESOURCE_EXHAUSTED`, `retryDelay: 76127s`)** -> Xóa sạch các vector Gemini dở dang và chuyển đồng nhất toàn bộ pipeline (`A`, `B`, `C`, `D` và 20 câu hỏi) sang **model thật chạy local `BAAI/bge-base-en-v1.5`** để bảo đảm 100% cùng embedding model và không tốn phí hay chạm rate limit.
- **Giới hạn xử lý PDF:** `pypdf` đọc layout 2 cột/bảng theo thứ tự text stream nên các trang có bảng (`Table 0.1` ở trang 15 báo cáo 2025) bị làm phẳng hàng-cột, và các câu dài vắt qua ranh giới trang PDF còn phần đoạn nối đầu trang (`pos_02`, `pos_07`, `pos_09`).

### Embedding model, vector store, metric, lệnh chạy và chi phí
- **Embedding model:** `BAAI/bge-base-en-v1.5` (`local_sentence_transformers`, **768 chiều**, L2-normalized). File cấu hình model: `starter/results/active_embedding_model.json`.
- **Vector store:** PostgreSQL `18.6` + `pgvector 0.8.1` (`starter/src/store.py`, bảng `document_chunks` dùng kiểu `vector(768)` và `JSONB` metadata), metric **exact cosine similarity** (`1 - (embedding <=> query_vector)`).
- **Idempotency & cleanup:** Khi ingest lại chiến lược `B` (`reingest_check.json`), số lượng giữ nguyên `407 -> 407`, `duplicates_created = 0`; kiểm tra co dữ liệu giả lập (`stale_cleanup_test`) xóa đúng `1` chunk lỗi thời (`3 -> 2`).
- **Lệnh chạy tái lập (từ thư mục `starter/`):**
  ```powershell
  .\.venv\Scripts\python.exe scripts\check_setup.py
  .\.venv\Scripts\python.exe -m pytest -v tests
  .\.venv\Scripts\python.exe -u -m src.run --split dev
  .\.venv\Scripts\python.exe -u -m src.run --split all --k-rrf 60 --bm25-weight 0.8
  ```
- **Thời gian & chi phí thực tế:**
  - Chạy toàn bộ pipeline `--split all` tái sử dụng cache DB cho chunk đã nhúng mất `97.649s` ingest/kiểm tra (`25.612s` chiến lược A, `18.644s` chiến lược B/C/D), `1.051s` nhúng 5 câu test mới, và `3.028s` (A), `2.818s` (B), `2.770s` (C), `5.888s` (D) cho tìm kiếm top-3 toàn 20 câu.
  - Chi phí API USD: **`$0.00`** (toàn bộ vector đều chạy local bằng `BAAI/bge-base-en-v1.5`, không gọi LLM trả lời).

### Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn
- **Metadata bắt buộc (`models.py`):** `document_id`, `source`, `page` (trang vật lý 1-indexed), `chunk_id` (`<strategy>__<doc_id>__p<page>__c<idx>`), `strategy`.
- **Trường lọc bổ sung:** `year` (`2024` | `2025`), `section` (`overview`, `executive_summary`, `economic_developments`, `special_focus_capital_markets`, `special_focus_emobility`), `category` (`capital_markets`, `emobility_transition`), `language` (`en`).
- **Nguồn giá trị & lý do chọn:** Ánh xạ xác định từ dải trang trong `data/manifest.json`. Trong corpus gồm 2 ấn phẩm *cùng chủ đề kinh tế Việt Nam qua hai năm liên tiếp*, các số liệu như dự báo GDP, cán cân tài khóa hay lạm phát lặp lại ở cả báo cáo 2024 lẫn 2025; thiếu lọc `year` sẽ gây trộn số liệu cũ/mới trong top-3.

---

## Cấu hình và benchmark

### Các chiến lược A / B / C và mở rộng D (`artifacts/experiment_log.csv`)
| Chiến lược | Chunking | Tham số | Bộ lọc | Vectorizer / Retrieval | Khác biệt kiểm soát |
|---|---|---|---|---|---|
| **A** | `fixed_size` | `size=500`, `overlap=50` | Không | `BAAI/bge-base-en-v1.5` | Baseline cửa sổ cố định |
| **B** | `by_paragraph` | `max_size=800`, `overlap=0` | Không | `BAAI/bge-base-en-v1.5` | Tách tác động chunking (A vs B) |
| **C** | `by_paragraph` | `max_size=800`, `overlap=0` | SQL `metadata @> filter::jsonb` | `BAAI/bge-base-en-v1.5` | **Chỉ khác B ở SQL metadata filter** (cô lập tác động lọc) |
| **D** *(mở rộng)* | `by_paragraph` | `max_size=800`, `overlap=0` | SQL `metadata @> filter::jsonb` | Dense (`pgvector` top-15) + `BM25Okapi` qua RRF (`k_rrf=60`, `dense_weight=1.0`, `bm25_weight=0.8`) | Tác động hybrid lexical + dense khi giữ nguyên chunk và filter như C |

### Thiết kế benchmark & kiểm chứng bằng chứng (`benchmark/questions.json`)
- **Tổng số câu:** **20 câu hỏi**, chia cố định **15 câu `dev` (`q01`–`q15`) / 5 câu `test` (`q16`–`q20`)** trước khi chạy.
- **Phân loại câu hỏi:**
  - *Trực tiếp:* `q01`, `q02`, `q05`, `q08`, `q11`, `q16`
  - *Diễn đạt lại:* `q03`, `q06`, `q09`, `q17`
  - *Điều kiện / ngoại lệ / so sánh kịch bản:* `q04`, `q07`, `q10`, `q12`, `q18`
  - *Metadata filter (`year` / `section`):* `q13`, `q14`, `q15` (dev) và `q19`, `q20` (test) — tổng 5 câu.
- **Kiểm chứng evidence:** Script kiểm định đối chiếu từng `evidence_quote` (và `alternative_gold` với các số liệu có mặt ở cả tóm tắt lẫn thân bài, ví dụ `p15` và `p32`/`p37`) khớp chính xác trên văn bản chuẩn hóa của trang nguồn (`20/20` câu qua kiểm chứng).

### Tiêu chí chọn chiến lược đã chốt trước khi xem `test`
1. Chọn chiến lược làm việc tốt nhất trên tập `dev` theo thứ tự ưu tiên: **Evidence Hit@3 ≥ 0.90**, **100% Filter Compliance** trên các câu có filter, và **Hit@1 cao nhất**, kết hợp đánh giá độ mạch lạc (Coherence).
2. Khi tinh chỉnh trọng số `bm25_weight` của mở rộng **D** chỉ trên `dev` (`--skip-ingest --split dev`):
   - `bm25_weight=0.5`: `dev` Hit@1 = `11/15 (0.7333)`, Evidence Hit@3 = `15/15 (1.0000)`
   - `bm25_weight=0.8`: `dev` Hit@1 = `11/15 (0.7333)`, Evidence Hit@3 = `15/15 (1.0000)`
   - `bm25_weight=1.2`: `dev` Hit@1 = `12/15 (0.8000)`, nhưng Evidence Hit@3 tụt xuống `14/15 (0.9333)`
   -> Chốt **`k_rrf = 60`, `bm25_weight = 0.8`** vì giữ trọn **100% Evidence Hit@3 trên `dev`**, rồi mới khóa cấu hình và chạy đánh giá tập `test` một lần duy nhất (`--split all`).

---

## Kết quả

### 1. Độ chính xác truy xuất, số chunk, thời gian và chi phí (`starter/results/metrics_summary.json`)
| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian search | Chi phí |
|---|---|---|---|---|---|---|
| A | dev | 0.3333 (5/15) | 0.6000 (9/15) | 560 | — | \$0.00 |
| A | test | 0.6000 (3/5) | 0.8000 (4/5) | 560 | — | \$0.00 |
| A | all | **0.4000 (8/20)** | **0.6500 (13/20)** | 560 (min 60, med 500, max 500) | 3.028s | \$0.00 |
| B | dev | 0.6667 (10/15) | 0.9333 (14/15) | 407 | — | \$0.00 |
| B | test | 0.8000 (4/5) | 1.0000 (5/5) | 407 | — | \$0.00 |
| B | all | **0.7000 (14/20)** | **0.9500 (19/20)** | 407 (min 8, med 649, max 800) | 2.818s | \$0.00 |
| C | dev | 0.7333 (11/15) | 0.9333 (14/15) | 407 | — | \$0.00 |
| C | test | 0.8000 (4/5) | 1.0000 (5/5) | 407 | — | \$0.00 |
| C | all | **0.7500 (15/20)** | **0.9500 (19/20)** | 407 (dùng chung với B) | 2.770s | \$0.00 |
| D | dev | 0.7333 (11/15) | 1.0000 (15/15) | 407 | — | \$0.00 |
| D | test | 1.0000 (5/5) | 1.0000 (5/5) | 407 | — | \$0.00 |
| D | all | **0.8000 (16/20)** | **1.0000 (20/20)** | 407 (dùng chung với B) | 5.888s | \$0.00 |

### 2. Đánh giá thủ công (Coherence, Metadata usefulness, Grounding quality)
| Strategy | Coherence TB /2 (n=10) | Metadata đủ và đúng (n/N test hits) | Filter tuân thủ (n/N) | Grounding 0/1/2 (số câu test) & TB /2 |
|---|---|---|---|---|
| A | **0.40** (4/20; 0:6, 1:4, 2:0) | 15/15* | Test chunks: 6/6 · **All queries: 3/5 (60%)** | 0:0, 1:3, 2:2 → **1.40 / 2.0** (7/10) |
| B | **1.50** (15/20; 0:0, 1:5, 2:5) | 15/15* | Test chunks: 5/6 · **All queries: 2/5 (40%)** | 0:0, 1:1, 2:4 → **1.80 / 2.0** (9/10) |
| C | **1.50** (dùng chung chunk B) | 15/15* | Test chunks: 6/6 · **All queries: 5/5 (100%)** | 0:0, 1:0, 2:5 → **2.00 / 2.0** (10/10) |
| D | **1.50** (dùng chung chunk B) | 15/15* | Test chunks: 6/6 · **All queries: 5/5 (100%)** | 0:0, 1:0, 2:5 → **2.00 / 2.0** (10/10) |

*\*Ghi chú kiểm toán metadata (`audit_metadata_labeled.csv`):* 15/15 dòng top-3 trên tập test có đủ 7 trường (`document_id`, `source`, `page`, `year`, `section`, `category`, `chunk_id`) và chính xác ở mức trang/section; riêng trường `category` được gán ở **cấp tài liệu** theo chuyên đề Part 2 (`capital_markets` hoặc `emobility_transition`) nên các chunk kinh tế vĩ mô Part 1 vẫn kế thừa `category` của toàn báo cáo.

- **Đường dẫn trace & nhãn thủ công:**
  - Trace từng câu cho 4 chiến lược: `starter/results/traces_A.json`, `starter/results/traces_B.json`, `starter/results/traces_C.json`, `starter/results/traces_D.json`.
  - Nhãn Coherence (10 vị trí trang cố định `pos_01`–`pos_10` từ trang 13–17 báo cáo 2024 và trang 11–15 báo cáo 2025): `starter/results/audit_coherence_sample.csv` & `starter/results/audit_coherence_labeled.csv`.
  - Nhãn Metadata (5 câu test × top-3 = 15 dòng/chiến lược, tổng 60 dòng): `starter/results/audit_metadata_sample.csv` & `starter/results/audit_metadata_labeled.csv`.
  - Nhãn Grounding (5 câu test/chiến lược, tổng 20 dòng): `starter/results/audit_grounding_sample.csv` & `starter/results/audit_grounding_labeled.csv`.
- **Cảnh báo thiên lệch đánh giá:** Toàn bộ điểm Coherence, Metadata và Grounding do chính tác giả chấm dựa trên tiêu chí cố định (`GUIDE.md`) và các đoạn `evidence_quote` đã khóa sẵn; dù kiểm tra chéo nghiêm ngặt theo từng dòng text thực tế, vẫn tồn tại nguy cơ thiên lệch tự chấm (*author bias*).

---

## Phân tích và kết luận

### So sánh A–B, B–C và C–D cùng các ví dụ lỗi cụ thể
1. **Tác động chunking (A vs B):** Chuyển từ cửa sổ cứng 500 ký tự sang gom đoạn theo ngưỡng tối đa 800 ký tự làm **giảm 27.3% số chunk (560 → 407)** nhưng **tăng vọt Coherence trung bình từ `0.40/2` lên `1.50/2`**, đẩy **Hit@1 từ `40%` (8/20) lên `70%` (14/20)** và **Evidence Hit@3 từ `65%` (13/20) lên `95%` (19/20)**.
   - *Ví dụ lỗi 1 — `q16` (tập `test`, `wb_taking_stock_aug2024` trang 52):* Với **A**, chunk rank #1 (`A__wb_taking_stock_aug2024__p052__c002`) cắt đôi giữa chữ **`"US$31 bill | ion) in total assets..."`**, làm mất hoàn toàn con số `"VND 786 trillion (US$31 billion)"` khỏi top-3 (`Evidence Hit@3 = False`, Grounding = `1/2`). Với **B**, đoạn văn nguyên vẹn (`B__wb_taking_stock_aug2024__p052__c001`) đứng hạng #1 đầy đủ cả giá trị tài sản lẫn vị trí xếp hạng thứ 2 (`Grounding = 2/2`).
2. **Tác động metadata filter (B vs C):** Giữ nguyên tập chunk 407 đoạn như B và chỉ thêm lọc SQL `WHERE metadata @> ...` loại bỏ triệt để sự nhầm lẫn giữa báo cáo 2024 và 2025 (**Filter Compliance tăng từ `40%` (2/5) lên `100%` (5/5)**, **Hit@1 tăng `70% → 75%`**, **Grounding test đạt tối đa `2.00/2.0`**).
   - *Ví dụ lỗi 2 — `q13` (tập `dev`, hỏi dự báo năm 2024 trong báo cáo tháng 8/2024, `filter={"year": 2024}`):* Do không lọc năm, **B** trả về chunk của **báo cáo tháng 3/2025** (`wb_taking_stock_mar2025 p37`, dự báo `6.8%` cho 2025) ở **rank #1** (`Hit@1 = False`, `filter_compliant = False`). Tương tự ở tập test (`q19`), **B** chèn chunk năm 2024 vào rank #2 cạnh chunk 2025 gây mâu thuẫn số liệu. **C** loại bỏ hoàn toàn rò rỉ chéo phiên bản ngay ở tầng SQL.
3. **Tác động Hybrid Dense + BM25 RRF (C vs D — hai chiều):**
   - *Nơi Hybrid giúp ích rõ rệt (`q10`, `q07`, `q18`):* Trên câu `q10` (thắt chặt tài khóa 1.8% GDP bất chấp tăng lương khu vực công), dense semantic retrieval (`A/B/C`) bị hút vào các chunk `"current account surplus"` và trượt cả Hit@3; **D** nhờ khớp từ khóa BM25 (`fiscal`, `tighten`, `1.8`, `wage`) kéo được `wb_taking_stock_mar2025 p14` vào rank #3, đưa **Evidence Hit@3 toàn bộ 20 câu lên `100% (20/20)`** và nâng Hit@1 câu `q07`, `q18` lên #1.
   - *Nơi Hybrid làm giảm hạng #1 (`q08`):* `q08` đạt Hit@1 ở B và C (`wb_taking_stock_mar2025 p19`), nhưng ở **D** trọng số BM25 kéo chunk triển vọng của báo cáo 2024 (`aug2024 p16`) lên rank #1 (`score 0.028266` vs `0.027988`) do lặp dày các từ phổ biến `"GDP"`, `"growth"`, `"exports"`, `"2024"` khi câu hỏi không gắn `year filter`.

### Chiến lược chọn, đánh đổi và điều kiện áp dụng
- **Chiến lược khuyến nghị (cốt lõi A/B/C):** Chọn **C (`by_paragraph` 800 + SQL metadata filter)** làm cấu hình cốt lõi đạt chuẩn, cân bằng tốt giữa độ chính xác (`75%` Hit@1, `95%` Hit@3), **100% tuân thủ filter**, tốc độ search nhanh nhất (`2.770s`) và độ trung thực ngữ cảnh (`2.00/2.0`).
  - *Áp tiêu chí chốt trước trên `dev`:* cả C (`14/15`) và D (`15/15`) đều qua ngưỡng Evidence Hit@3 ≥ 0.90, cùng 100% filter compliance, **hòa Hit@1 (`11/15`)** và dùng chung chunk nên hòa Coherence. Quy tắc phân xử khi hòa: ưu tiên cấu hình đơn giản hơn và search nhanh hơn (C chỉ dùng pgvector, không cần BM25 trong Python) → C là cấu hình cốt lõi; D là phương án mở rộng khi cần recall cao hơn.
- **Mở rộng đề xuất khi cần recall tối đa:** Dùng **D (Hybrid BGE + BM25, `bm25_weight=0.8`)** nếu hệ thống ưu tiên **100% Evidence Hit@3** (`20/20`) cho các truy vấn chứa số liệu/thuật ngữ chuyên ngành hẹp (`NDC`, `e-2Ws`, `fiscal`), chấp nhận tăng thời gian truy xuất gấp ~2.1 lần (`5.888s` vs `2.770s` do tính BM25 trong bộ nhớ trên candidate đã lọc). Với khối dữ liệu lớn hơn nhiều lần, cần chuyển BM25 sang `tsvector` / GIN index trực tiếp trong PostgreSQL thay vì chấm trên Python.

### Hạn chế hiện tại
- Tập `test` chỉ có `5` câu (đủ ngưỡng tối thiểu benchmark nhưng phương sai mỗi câu là 20%, như mức 60%→80%→100% test Hit@1).
- Tự chấm tay Coherence/Grounding bởi người xây dựng benchmark có thể thiên lệch; cần thêm người chấm chéo độc lập (blind review).
- Bảng biểu (`Table 0.1`) và chữ vắt qua trang PDF chưa được ghép lại ở bước ingestion.
- Chạy local trên CPU mất `97.649s` cho khâu ingest/kiểm tra 2 tài liệu ở lần chạy `--split all` (đã tái sử dụng vector có sẵn trong DB) (chậm hơn GPU nhưng chi phí `0` và không lệ thuộc quota API).

### Hướng cải tiến nếu làm lại / mở rộng
1. Dùng PDF layout/table parser (như `pdfplumber` / Docling) để trích xuất bảng thành Markdown/JSON giữ nguyên tiêu đề cột và ghép câu vắt qua ranh giới trang (`page n` → `page n+1`).
2. Gán `category` ở cấp **section/chunk** (`macroeconomic_update`, `fiscal_policy`, `capital_markets`, `emobility_transition`) thay vì kế thừa từ chủ đề chính của cả PDF.
3. Chuyển tìm kiếm từ khóa của D sang Postgres `GIN(to_tsvector('english', text))` + RRF trong SQL để mở rộng hàng chục nghìn trang không cần load corpus về Python.

---

## Nhật ký thực hiện theo 6 Checkpoint (`CHECKPOINTS.md`)
| Checkpoint | Mốc | Những gì đã thực hiện và kiểm chứng |
|---|---|---|
| **CP1** | `0:25` | Thiết lập `.venv`, cài dependencies (`psycopg`, `pypdf`, `sentence-transformers`, `rank-bm25`, `pytest`), dựng PostgreSQL 18.6 + `pgvector 0.8.1` trên WSL2 (`check_setup.py -> Setup OK`), tải 2 PDF World Bank CC BY-NC 3.0 IGO, sinh `data/manifest.json` (`94` trang nội dung hợp lệ). |
| **CP2** | `1:00` | Xây dựng `benchmark/questions.json` với `20` câu hỏi (`15 dev` / `5 test`, 5 câu filter, có `alternative_gold`), xác minh `100%` trích dẫn bằng script trên text chuẩn hóa, và khóa trước giả thuyết A/B/C/D trong `artifacts/experiment_log.csv`. |
| **CP3** | `1:50` | Hiện thực toàn bộ pipeline (`models.py`, `ingestion.py`, `chunking.py`, `embeddings.py`, `store.py`) với `fixed_size` (A) và `by_paragraph` (B), upsert bất biến (`reingest_check.json`: 0 bản trùng), log sự cố Docker & Gemini Free Tier 1,000 RPD (`provider_error_log.json`), chuyển sạch sang local `BAAI/bge-base-en-v1.5`. |
| **CP4** | `2:40` | Hoàn thiện `evaluation.py` (Hit@1, Evidence Hit@3, Filter Compliance, Hybrid RRF D), `run.py`, `tests/test_pipeline.py` (`5 passed`), tinh chỉnh `bm25_weight` chỉ trên `dev` (`0.5 / 0.8 / 1.2` -> khóa `0.8` vì giữ `15/15` Hit@3), rồi chạy `--split all` đánh giá một lần toàn bộ 20 câu. |
| **CP5** | `3:30` | Xuất và đọc trực tiếp 3 tập mẫu kiểm toán (`audit_coherence_sample.csv`, `audit_metadata_sample.csv`, `audit_grounding_sample.csv`) + trace lỗi; chấm tay đầy đủ 100% từng dòng vào `*_labeled.csv`, ghi nhận rõ sự cố cắt chữ `q16 [A]`, rò rỉ năm `q13/q19 [B]`, và tác động 2 chiều của Hybrid `D` (`q10` vs `q08`). |
| **CP6** | `4:00` | Hoàn tất `artifacts/experiment_log.csv`, `report/individual/2A202603464_DuongMinhTan.md`, `STUDENT.md`, cập nhật `README.md`, đối chiếu checklist trong `SUBMISSION.md`, chạy lại `pytest` + pipeline toàn phần đảm bảo không còn lỗi. |

