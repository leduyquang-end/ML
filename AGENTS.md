# AGENTS.md

Hướng dẫn cho AI coding agent làm việc trong repo này. Đọc kỹ trước khi sửa code.

## Dự án

Phân loại tin giả tiếng Việt: PhoBERT (đóng băng) tạo embedding → **router** chọn lane → **Lane A** (Logistic Regression) hoặc **Lane B** (SVM) → **fusion** ra nhãn cuối. Xem `README.md` để biết tổng quan.

## Quy trình bắt buộc: lập kế hoạch trước khi code

Dùng skill `planning-and-task-breakdown` cho mọi yêu cầu thêm tính năng, sửa lỗi hoặc refactor không tầm thường.

1. **Chưa viết code khi đang lập kế hoạch.** Đọc code liên quan, xác định phụ thuộc, rủi ro, câu hỏi mở.
2. Kế hoạch lưu ở `tasks/plan.md`; danh sách task lưu ở `tasks/todo.md` (mỗi task có acceptance criteria, verification, dependencies, file liên quan, mức độ lớn).
3. **Không ghi đè** `tasks/plan.md` hoặc `tasks/todo.md` nếu còn task chưa tick cho công việc khác. Dừng lại và hỏi người dùng.
4. Tóm tắt kế hoạch và **chờ người dùng duyệt** trước khi code.
5. Làm từng task nhỏ (ưu tiên cỡ S/M, không chạm quá ~5 file). Xong task nào thì tick `- [x]` ngay trong `tasks/todo.md`.
6. Ở mỗi checkpoint: chạy test, kiểm tra ranh giới module và báo cáo trước khi sang giai đoạn kế tiếp.
7. Cắt lát dọc (một đường chạy đầy đủ cho mỗi task), không làm "toàn bộ tầng dưới rồi mới làm tầng trên".

## Kiến trúc: Modular Monolith

Mã nguồn ở `src/fakenews/`:

- `shared/`: kiểu dữ liệu chung (`Sample`, `Label`, `LaneId`, `Prediction`), cấu hình, seed, logging, lỗi. Chỉ chứa thứ trung lập, ít thay đổi.
- `modules/`: `dataset`, `preprocessing`, `embedding`, `lanes`, `routing`, `fusion`, `evaluation`, `inference`.
- `app/`: Composition Root, nơi duy nhất nối các module (pipeline huấn luyện, đánh giá, ablation, CLI).

Cấu trúc mỗi module:

```
modules/<tên>/
├── __init__.py    # chỉ re-export từ api.py
├── api.py         # cổng công khai (hàm, lớp, Protocol)
├── _internal/     # chi tiết riêng, có thể đổi tự do
└── tests/         # test đơn vị của module
```

### Luật phụ thuộc (không được vi phạm)

- Module nghiệp vụ (`dataset`, `preprocessing`, `embedding`, `lanes`, `routing`, `fusion`, `evaluation`) chỉ được import `shared`. **Không import module nghiệp vụ khác.**
- **Không import `_internal/` của module khác.** Chỉ dùng `api.py`.
- `inference` được import `api.py` của các module khác; `app/` được nối mọi module qua `api.py`.
- `shared` không import module nào.
- Dữ liệu đi giữa các module bằng numpy array và kiểu trong `shared/types.py`; `app/` chuyển dữ liệu từ module này sang module kia. Ví dụ `routing` không import `lanes`: `app/` lấy nhãn OOF từ `lanes` rồi đưa cho `routing`.
- Cần dùng chung logic giữa hai module nghiệp vụ? Hỏi người dùng trước; thường nên đưa vào `shared/` (nếu thật sự trung lập) hoặc để `app/` điều phối, không tạo import chéo.

Luật được ép bằng `import-linter` (`.importlinter`). Chạy `lint-imports` và phải xanh.

## Lệnh thường dùng

> Chỉnh lại nếu tên lệnh thật khác.

```bash
pip install -e ".[dev]"                         # cài đặt
pytest                                          # toàn bộ test
pytest src/fakenews/modules/<tên>/tests         # test một module
lint-imports                                    # kiểm tra ranh giới module
ruff check . && ruff format --check .           # lint và format
python -m fakenews.app.cli prepare|train|evaluate|predict
```

## Quy tắc về ML (quan trọng)

- **Không rò rỉ dữ liệu.** Chia train/val/test một lần, phân tầng nhãn, seed cố định, không trùng mẫu giữa các tập.
- **Nhãn của router chỉ lấy từ dự đoán out-of-fold** trên tập train, tuyệt đối không dùng dự đoán in-sample của lane.
- **Không dùng tập test để chọn cấu hình hay siêu tham số.** Chỉ chạy test một lần ở bước đánh giá cuối, sau khi đã chốt cấu hình. Chọn lựa dùng validation hoặc cross-validation trên train.
- PhoBERT chỉ trích xuất đặc trưng (đóng băng). Văn bản phải được tách từ trước khi mã hóa; văn bản dài hơn 256 token thì chia đoạn thay vì cắt cụt âm thầm.
- Embedding được cache ra `data/embeddings/`; không tính lại nếu cache còn hợp lệ.
- Luôn so sánh với baseline: LR đơn, SVM đơn, trung bình LR+SVM, oracle router. Không tuyên bố router "tốt hơn" nếu không có số liệu so sánh.
- Báo cáo **macro-F1** và **F1 lớp tin giả**, không chỉ accuracy (dữ liệu có thể mất cân bằng).
- Đặt seed qua `shared/seed.py`; kết quả phải tái lập được.
- Báo cáo trung thực: nếu router không vượt baseline thì nói rõ, đừng chỉnh để "đẹp số".

## Quy ước code

- Python 3.10+, có type hints cho hàm công khai trong `api.py`.
- Mỗi hành vi mới phải có test; test của module nằm trong `modules/<tên>/tests/`, test xuyên module nằm trong `tests/integration/` hoặc `tests/e2e/`.
- Cấu hình (đường dẫn, siêu tham số, loại router, kiểu fusion) đặt trong `configs/*.yaml`, không hard-code trong logic.
- Đặt tên theo ý nghĩa; ưu tiên hàm nhỏ, ít tác dụng phụ. Không thêm phụ thuộc mới khi chưa cần; nếu cần, nêu lý do cho người dùng.
- Log qua `shared/logging.py`, không dùng `print` trong mã thư viện.
- Commit nhỏ, mỗi commit tương ứng một task hoặc một phần task có ý nghĩa.

## Không được làm

- Không commit `data/`, `artifacts/`, file model, embedding hay dữ liệu thô.
- Không sửa hoặc xóa `tasks/plan.md`, `tasks/todo.md` của công việc khác khi chưa hỏi.
- Không bỏ qua hoặc tắt test/`lint-imports` để "cho qua".
- Không đưa dữ liệu thật hoặc thông tin nhạy cảm vào repo, log hay ví dụ.
- Không tự ý đổi cấu trúc module hoặc ranh giới đã thống nhất; đề xuất và chờ duyệt.

## Khi chưa chắc

Hỏi người dùng thay vì đoán, nhất là các điểm còn mở trong `tasks/plan.md` (dataset, tiêu chí router, kiểu fusion, nhị phân hay nhiều lớp).