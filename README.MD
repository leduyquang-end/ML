# Trích đặc trưng tin tức tiếng Việt bằng PhoBERT

## 1. Mục tiêu và trạng thái

Dự án đọc dữ liệu tin tức, tiền xử lý, chia bài dài thành các chunk và dùng
`vinai/phobert-base` đã đóng băng để tạo **một vector cho mỗi bài báo**.

Đã triển khai đến bước trích đặc trưng. **Chưa huấn luyện Logistic Regression
hoặc SVM**, chưa có kết quả đánh giá phân loại. Nhãn trong các file đầu ra là
nhãn gốc của dữ liệu: `0 = Real`, `1 = Fake`.

```text
CSV → làm sạch, bỏ trùng → chia train/test theo bài → PyVi → tokenize/chunk
    → PhoBERT → vector chunk → gộp có trọng số → vector bài
```

Có hai chế độ chạy: chỉ xuất dữ liệu sau chunk, hoặc chạy đầy đủ để xuất vector.

## 2. Cấu trúc thư mục

```text
data/vn_news_226_tlfr.csv       # Dữ liệu đầu vào
phobert_features.py            # Điểm chạy chính, nhận tham số dòng lệnh
phobert_pipeline/
    config.py                 # Checkpoint, đường dẫn và cache
    preprocessing.py          # Đọc CSV, làm sạch, bỏ trùng, tách từ
    splitting.py              # Chia train/test theo bài
    chunking.py               # Tokenize, giới hạn độ dài, chia cửa sổ overlap
    encoder.py                # Nạp PhoBERT đóng băng, tạo vector chunk
    aggregation.py            # Gộp vector chunk thành vector bài
    pipeline.py               # Điều phối các bước trích đặc trưng
    export_chunks.py          # Xuất dữ liệu sau chunk thành CSV
    outputs.py                # Lưu vector, nhãn, split và encoder
outputs/                      # Dữ liệu sinh ra khi chạy, không đưa lên Git
.cache/                       # Cache mô hình và file tạm, không đưa lên Git
run.ps1                       # Chạy trên Windows bằng .venv của dự án
Dockerfile / compose.yaml     # Môi trường chạy CPU để chia sẻ
```

## 3. Tiền xử lý dữ liệu

CSV cần có hai cột `text` và `label`. Chương trình:

- Chuẩn hóa Unicode NFC và khoảng trắng; giữ cả tiêu đề lẫn thân bài.
- Bỏ bản ghi thiếu văn bản/nhãn; chỉ chấp nhận nhãn `0` hoặc `1`.
- Bỏ văn bản trùng sau chuẩn hóa; báo lỗi nếu bài trùng nhưng khác nhãn.
- Giữ dấu tiếng Việt, chữ hoa, số, dấu câu và từ phủ định; không bỏ stopwords.
- Tách từ bằng `pyvi.ViTokenizer` trước khi tokenize cho PhoBERT.

Dữ liệu hiện tại: **226 bài → bỏ 1 bài trùng → 225 bài**, gồm 123 Real và 102 Fake.
`article_id` là chỉ số bản ghi CSV ban đầu, tính từ 0; sau khi bỏ trùng ID có thể bị khuyết.

## 4. Chia train/test

Mặc định chia **80% train / 20% test**, phân tầng theo nhãn, `seed = 42`.
Với dữ liệu hiện tại: **180 bài train và 45 bài test**.

Chia theo bài **trước khi tạo chunk**. Tất cả chunk của cùng một bài thuộc cùng
một tập, tránh việc mô hình học một phần bài rồi được kiểm tra trên phần khác
của chính bài đó. Cùng dữ liệu, tham số và thứ tự bài sẽ tạo cùng một split.

PhoBERT đóng băng trích vector cho cả hai tập. Khi bổ sung bộ phân loại,
chỉ dùng train để học/chọn cấu hình; dành test cho đánh giá cuối.

## 5. Tokenize và chia chunk

Tokenize toàn bài với `truncation=False`, sau đó dùng cửa sổ trượt có overlap.
Độ dài được tính bằng tokenizer của checkpoint, không suy từ số ký tự.

| Tham số mặc định với PhoBERT base | Giá trị |
| --- | --- |
| Tổng token mỗi chunk, gồm token đặc biệt | Tối đa 256 |
| Token nội dung mỗi chunk | Tối đa 254 |
| Token đặc biệt | `<s>` và `</s>` |
| Overlap | 32 token nội dung |
| Bước dịch cửa sổ | 254 − 32 = 222 token |

Giới hạn được đối chiếu với bảng vị trí/config và tokenizer. Token đặc biệt
được thêm riêng sau khi chia chunk. Không bỏ phần đuôi bài dài.
Cấu hình mặc định hiện tạo **621 chunk từ 225 bài**.

## 6. Đưa dữ liệu qua PhoBERT

Encoder dùng `requires_grad_(False)`, `eval()` và `torch.no_grad()`;
không cập nhật trọng số và không fine-tune.

Mỗi chunk cho hidden state của từng token. Masked mean pooling lấy trung bình
các token nội dung, loại padding và token đặc biệt, tạo một vector chunk.
Số chiều đọc từ cấu hình checkpoint; PhoBERT base có **768 chiều**.

## 7. Gộp thành vector bài

```text
vector_bài = tổng(trọng_số_chunk × vector_chunk) / tổng(trọng_số_chunk)
```

Trọng số là số token mới: chunk đầu tính toàn bộ nội dung, chunk sau trừ phần
overlap. Mỗi bài cuối cùng có đúng một vector, dù gồm một hay nhiều chunk.

Đây là baseline: phần overlap vẫn góp ngữ cảnh trong vector chunk;
trọng số không loại trùng chính xác từng vector token và phép gộp không giữ thứ tự đoạn.

## 8. Các file đầu ra dùng để làm gì?

| File | Nội dung | Kết quả hiện tại |
| --- | --- | --- |
| `outputs/phobert/articles.csv` | Văn bản đã làm sạch/bỏ trùng, chưa tách từ/chunk; gồm ID và nhãn gốc | 225 bài |
| `outputs/chunks.csv` | Mỗi dòng là một chunk trước encoder: ID bài, split, nhãn, ID chunk, vị trí token, văn bản tách từ và `input_ids` dạng JSON | 621 chunk |
| `outputs/phobert/embeddings.csv` | Mỗi dòng là vector bài sau PhoBERT và gộp chunk; gồm ID, split, nhãn, số chunk, `feature_0`…`feature_767` | 225 × 768 đặc trưng |
| `outputs/phobert/features.npz` | Cùng vector/nhãn ở dạng NumPy, tách sẵn train/test | Train `(180, 768)`, test `(45, 768)` |
| `outputs/phobert/metadata.json` | Checkpoint/revision, seed, cấu hình chunk, thống kê và ID từng split | Dùng đối chiếu, tái lập |
| `outputs/phobert/encoder/` | Encoder và tokenizer được lưu lại | Dùng trích vector sau này |

`chunks.csv` được tạo bằng lệnh xuất chunk riêng. Các file trong
`outputs/phobert/` được tạo khi chạy đầy đủ. CSV dùng UTF-8 BOM để mở bằng Excel.
`embeddings.csv` chưa qua scaler và không chứa nhãn dự đoán hay xác suất phân loại.

Đọc dữ liệu cho bước học máy tiếp theo:

```python
import numpy as np

with np.load("outputs/phobert/features.npz", allow_pickle=False) as data:
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
```

NPZ còn có `train_ids`, `test_ids`, `train_chunk_counts`, `test_chunk_counts`.
Dùng `article_id` để đối chiếu một bài giữa các file, không dùng số dòng CSV.

## 9. Chạy bằng Python trên Windows

Dùng Python 3.13 như môi trường đã kiểm tra. Mở terminal tại thư mục dự án trên ổ E:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --cache-dir .cache/pip -r requirements.txt

# Chỉ kiểm tra dữ liệu, không tải mô hình
.\run.ps1 --check-data

# Tiền xử lý và xuất chunk; chỉ tải tokenizer/config
.\run.ps1 --export-chunks outputs/chunks.csv

# Chạy PhoBERT và xuất vector bài
.\run.ps1
```

Có thể điều chỉnh tham số:

```powershell
.\run.ps1 --batch-size 4 --overlap 32 --test-size 0.2 --seed 42
```

Lần đầu cần Internet để tải tokenizer và trọng số. Chương trình dùng CUDA nếu
PyTorch đang cài hỗ trợ GPU, nếu không dùng CPU. `.venv`, cache mô hình,
file tạm và đầu ra được đặt trong dự án; `run.ps1` đặt TEMP/TMP trước khi chạy Python.
PyVi trên Windows nạp CRF vào RAM để hỗ trợ đường dẫn có dấu.

## 10. Chạy và chia sẻ bằng Docker

```powershell
docker compose build
docker compose run --rm phobert --check-data
docker compose run --rm phobert --export-chunks outputs/chunks.csv
docker compose run --rm phobert
```

Image chạy bằng CPU. Compose gắn `data/` chỉ đọc và lưu `outputs/`, `.cache/`
về thư mục dự án trên máy chủ. Lần đầu vẫn cần Internet để tải checkpoint.
Docker Desktop quản lý nơi lưu image/máy ảo riêng; bind mount không đổi vị trí đó.

Xem [RUN.md](RUN.md) để xuất/nhập image và cấu hình lưu trữ trên Windows.
Cấu hình Docker đã được kiểm tra tĩnh; chưa build/chạy thử trong môi trường hiện tại.

## 11. Kiểm tra

```powershell
.\.venv\Scripts\python.exe -B -m unittest -v test_phobert_features
```

Hiện có **11 kiểm thử**, gồm làm sạch/nhãn, coverage và trọng số chunk, giới hạn
vị trí, chia tập, PyVi trên đường dẫn tiếng Việt và dữ liệu xuất CSV.
Nếu đã cài Pyright: `.\.venv\Scripts\python.exe -m pyright`.

Lượt chạy dữ liệu thật đã xác nhận 225 vector × 768 chiều, không NaN/Inf;
vector và nhãn trong CSV khớp NPZ, cùng 621 chunk và split 180/45.
Các số này kiểm tra luồng dữ liệu, không phải chất lượng phân loại tin giả.

## 12. Hướng phát triển

Bước tiếp theo là Logistic Regression và SVM dùng chung vector/split.
Nếu chuẩn hóa vector, đặt scaler trong từng fold train để tránh leakage;
chọn cấu hình bằng cross-validation và chỉ đánh giá test sau cùng.
Dữ liệu hiện ít; mới loại trùng sau chuẩn hóa, chưa tự gom bài gần trùng/cùng sự kiện.

Tham khảo: [PhoBERT — VinAI Research](https://github.com/VinAIResearch/PhoBERT).
