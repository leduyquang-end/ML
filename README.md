# Fake News Router

Phân loại tin giả tiếng Việt bằng kiến trúc **router**: văn bản được mã hóa bằng PhoBERT, một router quyết định đưa mẫu vào **Lane A** (Logistic Regression) hoặc **Lane B** (SVM), sau đó kết quả hai lane được **kết hợp (fusion)** để ra nhãn tin thật / tin giả.

```
Văn bản ─► Tiền xử lý ─► PhoBERT embedding ─► Router ─┬─► Lane A: Logistic Regression ─┐
                                                      └─► Lane B: SVM ─────────────────┴─► Fusion ─► Nhãn
```

> Trạng thái: đang phát triển. Tiến độ xem tại [`tasks/todo.md`](tasks/todo.md).

## Ý tưởng chính

- **PhoBERT đóng băng** (`vinai/phobert-base-v2`) chỉ làm bộ trích xuất đặc trưng; embedding được cache để thử nghiệm nhanh.
- **Lane A (Logistic Regression)**: nhanh, xác suất ổn định, hợp với mẫu có biên rõ ràng.
- **Lane B (SVM RBF + hiệu chỉnh xác suất)**: bắt biên phi tuyến, hợp với mẫu sát biên.
- **Router**: chọn lane cho từng mẫu. Có router luật (baseline) và router học được, huấn luyện trên nhãn "lane nào đúng hơn" lấy từ dự đoán out-of-fold.
- **Fusion**: hard (lấy kết quả lane được chọn), soft (trộn xác suất có trọng số) hoặc stacking.
- **Baseline để so sánh**: LR đơn, SVM đơn, trung bình LR+SVM và oracle router (giới hạn trên nếu router chọn lane hoàn hảo).

## Kiến trúc: Modular Monolith

Một codebase, một lần triển khai, nhưng chia thành các module có ranh giới rõ. Mỗi module chỉ lộ ra một cổng công khai (`api.py`); phần còn lại nằm trong `_internal/`.

```
src/fakenews/
├── shared/        # kiểu dữ liệu, cấu hình, seed, logging (trung lập)
├── modules/
│   ├── dataset/        # nạp, làm sạch, chia train/val/test
│   ├── preprocessing/  # chuẩn hóa Unicode, tách từ
│   ├── embedding/      # PhoBERT + cache
│   ├── lanes/          # Lane A, Lane B, dự đoán out-of-fold
│   ├── routing/        # router luật và router học được
│   ├── fusion/         # hard / soft / stacking
│   ├── evaluation/     # metric, oracle router, báo cáo, ablation
│   └── inference/      # lưu/nạp mô hình, predict(text)
└── app/           # Composition Root: nối các module, pipeline và CLI
```

Quy tắc phụ thuộc:
- Các module nghiệp vụ chỉ được import `shared`, **không import module khác**.
- Không ai được import `_internal/` của module khác.
- Chỉ `app/` (và `inference/` qua `api.py`) được nối nhiều module với nhau.
- Luật này được kiểm tra tự động bằng `import-linter` (xem `.importlinter`).

## Cấu trúc thư mục

```
fake-news-router/
├── pyproject.toml
├── README.md
├── AGENTS.md              # quy tắc cho AI coding agent
├── .importlinter          # luật ranh giới module
├── configs/               # default.yaml, ablation.yaml
├── tasks/                 # plan.md, todo.md
├── data/                  # raw/, processed/, embeddings/  (không commit)
├── artifacts/             # models/, reports/              (không commit)
├── notebooks/
├── src/fakenews/          # mã nguồn (xem phần trên)
└── tests/                 # integration/, e2e/, architecture/
```

## Cài đặt

Yêu cầu: Python 3.10+, GPU là tùy chọn (giúp trích xuất embedding nhanh hơn).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Sử dụng

> Các lệnh dưới đây là kế hoạch; chỉnh lại nếu tên lệnh trong `app/cli.py` khác.

```bash
# 1. Chuẩn bị dữ liệu và cache embedding
python -m fakenews.app.cli prepare --config configs/default.yaml

# 2. Huấn luyện lane, router, fusion
python -m fakenews.app.cli train --config configs/default.yaml

# 3. Đánh giá (validation hoặc test)
python -m fakenews.app.cli evaluate --split val
python -m fakenews.app.cli evaluate --split test   # chỉ chạy khi đã chốt cấu hình

# 4. Dự đoán
python -m fakenews.app.cli predict --text "Nội dung bài viết cần kiểm tra"
```

Kết quả `predict` gồm nhãn, xác suất và lane đã xử lý mẫu.

Dùng như thư viện:

```python
from fakenews.modules.inference.api import FakeNewsService

service = FakeNewsService.load("artifacts/models/latest")
result = service.predict("Nội dung bài viết cần kiểm tra")
print(result.label, result.probability, result.lane)
```

## Dữ liệu

Đặt dữ liệu thô vào `data/raw/` và khai báo đường dẫn, định dạng cột trong `configs/default.yaml`. Dataset cụ thể chưa được chốt (xem Open Questions trong `tasks/plan.md`). Dữ liệu và artifact không được commit lên git.

## Đánh giá

Chỉ số chính: **macro-F1** và **F1 của lớp tin giả**; thêm precision/recall, ma trận nhầm lẫn và tỉ lệ mẫu đi vào mỗi lane. Mọi lựa chọn cấu hình dựa trên validation/cross-validation; **tập test chỉ dùng một lần** sau khi đã chốt cấu hình.

| Mô hình | Macro-F1 | F1 (tin giả) |
|---|---|---|
| LR đơn | _chưa có_ | _chưa có_ |
| SVM đơn | _chưa có_ | _chưa có_ |
| Trung bình LR+SVM | _chưa có_ | _chưa có_ |
| Router + hard fusion | _chưa có_ | _chưa có_ |
| Router + soft fusion | _chưa có_ | _chưa có_ |
| Oracle router (giới hạn trên) | _chưa có_ | _chưa có_ |

## Phát triển

```bash
pytest                      # chạy toàn bộ test
lint-imports                # kiểm tra luật ranh giới module
ruff check . && ruff format --check .
```

Quy trình làm việc: lập kế hoạch trong `tasks/plan.md`, theo dõi task trong `tasks/todo.md`, làm từng task nhỏ, chạy test và kiểm tra ranh giới module trước khi đánh dấu hoàn thành. Chi tiết quy tắc cho AI agent xem [`AGENTS.md`](AGENTS.md).

## Giấy phép

_Chưa xác định._