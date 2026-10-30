# Trích đặc trưng PhoBERT (dừng trước hai bộ phân loại)

`phobert_features.py` là điểm chạy chính, gọi các module trong `phobert_pipeline/`.
Các module tương ứng mục 3–7 của `README.MD`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --cache-dir .cache/pip -r requirements.txt
.\run.ps1 --check-data
.\run.ps1
```

Mở terminal tại thư mục dự án trên ổ E. Môi trường `.venv`, đầu ra `outputs/`,
cache Hugging Face/PyTorch/pip và thư mục tạm `.cache/tmp` đều ở trong dự án.
`run.ps1` đặt thư mục làm việc và biến TEMP/TMP trước khi khởi động Python;
chương trình đặt thêm các cache mô hình trước khi import thư viện.

| Module | Chức năng theo README |
| --- | --- |
| `preprocessing.py` | Mục 3: đọc CSV, làm sạch, kiểm tra nhãn, bỏ trùng, tách từ |
| `splitting.py` | Mục 4: chia train/test theo bài |
| `chunking.py` | Mục 5: giới hạn token và cửa sổ overlap |
| `encoder.py` | Mục 6: nạp/đóng băng PhoBERT, vector từng chunk |
| `aggregation.py` | Mục 7: gộp vector chunk theo số token mới |
| `pipeline.py` | Ghép các bước để trích vector toàn bộ bài |
| `outputs.py` | Lưu đặc trưng, nhãn, split, cấu hình và encoder |
| `config.py` | Checkpoint, đường dẫn và cache trong dự án |

Kiểm tra logic dữ liệu và chunking (không cần tải mô hình):
`.\.venv\Scripts\python.exe -B -m unittest -v test_phobert_features`.

VS Code đã có cấu hình môi trường riêng trong `.vscode/settings.json`.
Nếu trước đó đã chọn Python toàn hệ thống, dùng `Python: Select Interpreter`
và chọn `.venv\Scripts\python.exe` để chạy bằng đúng môi trường đã cài thư viện.

Lần đầu cần Internet để tải `vinai/phobert-base`. Mặc định dùng GPU CUDA
nếu có, nếu không dùng CPU. Trọng số encoder được đóng băng, không fine-tune.

- Làm sạch nhẹ, giữ dấu/chữ hoa/số/dấu câu; tách từ bằng `pyvi.ViTokenizer`.
- Bỏ bài rỗng và trùng sau chuẩn hóa; báo lỗi nếu bài trùng khác nhãn.
- Chia phân tầng train/test 80/20 theo bài, seed 42, trước khi tạo chunk.
- Tokenize toàn bài, chia cửa sổ có overlap 32 token, không cắt bỏ đuôi bài.
- Giới hạn được lấy từ bảng vị trí RoBERTa và tokenizer, có chừa token đặc biệt.
- Mean pooling loại padding/token đặc biệt; gộp chunk theo số token mới.
  Đây là trọng số baseline: token overlap vẫn góp ngữ cảnh vào vector chunk,
  không phải phép loại trùng chính xác từng vector token.

Đầu ra trong `outputs/phobert/`:

| File | Nội dung |
| --- | --- |
| `features.npz` | `X_train`, `y_train`, `X_test`, `y_test`, ID và số chunk tương ứng |
| `metadata.json` | Cấu hình, revision, seed, ID từng split và thống kê dữ liệu/chunk |
| `articles.csv` | Văn bản sạch, nhãn và ID bản ghi CSV gốc (đánh số từ 0) |
| `encoder/` | PhoBERT và tokenizer dùng để trích vector |

Mỗi hàng `X` là một bài; số cột bằng `encoder.config.hidden_size` (base: 768).
Nhãn `0 = Real`, `1 = Fake`. Chưa chuẩn hóa vector, huấn luyện hay đánh giá
Logistic Regression/SVM. Khi triển khai tiếp, fit scaler trong từng fold train.
Chưa phát hiện tự động bài gần trùng/cùng sự kiện; cần rà soát riêng nếu có.

Tham khảo: [hướng dẫn PhoBERT chính thức](https://github.com/VinAIResearch/PhoBERT).

## Chia sẻ và chạy bằng Docker

Cần Docker Engine và Docker Compose (hoặc Docker Desktop trên Windows).
Image dùng CPU; người nhận không cần tự cài Python hay thư viện.
Mở terminal tại thư mục dự án trên ổ E:

```powershell
docker compose build
docker compose run --rm phobert --check-data
docker compose run --rm phobert
```

Lần chạy trích vector đầu tiên cần Internet để tải PhoBERT.
Kết quả lưu về `./outputs/phobert/`, cache và file tạm lưu về `./.cache/`
trên máy chủ; các thư mục này được giữ lại khi container bị xóa.
Thư mục `data` được gắn chỉ đọc. Có thể chỉnh tham số khi chạy:

```powershell
docker compose run --rm phobert --batch-size 4 --overlap 32
```

Kiểm thử trong container, không tải checkpoint:

```powershell
docker compose run --rm --entrypoint python phobert -B -m unittest -v test_phobert_features
```

Chia sẻ cả thư mục source (không cần `.venv`, `.cache`, `outputs`) để người nhận
build lại, hoặc xuất image đã build thành file:

```powershell
docker save -o phobert-features.tar phobert-features:local
```

Người nhận đặt file image và `compose.yaml` trong thư mục có dữ liệu CSV ở `data/`,
rồi chạy:

```powershell
docker load -i phobert-features.tar
docker compose run --rm phobert --check-data
docker compose run --rm phobert
```

Image đã kèm dữ liệu CSV hiện tại; nếu chạy bằng Compose, dữ liệu ở `./data/`
của người nhận sẽ thay thế dữ liệu trong image. File image không kèm cache mô hình
hay kết quả. Nếu chỉ nhận image, có thể kiểm tra dữ liệu kèm sẵn bằng
`docker run --rm phobert-features:local --check-data`.

Docker Desktop quản lý nơi lưu image/ổ đĩa máy ảo riêng. Để phần đó cũng nằm
trên ổ E, đặt **Settings → Resources → Advanced → Disk image location** sang ổ E
nếu bản Docker Desktop đang dùng hỗ trợ mục này. Các bind mount trong Compose
chỉ quyết định nơi lưu dữ liệu/cache/kết quả, không thay đổi nơi Docker lưu image.

Tài liệu Docker: [Compose run](https://docs.docker.com/reference/cli/docker/compose/run/),
[cấu hình nơi lưu image](https://docs.docker.com/desktop/settings-and-maintenance/settings/).
