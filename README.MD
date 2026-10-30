# Hướng dẫn: PhoBERT chunking + Logistic Regression/SVM

## 1. Mục tiêu

Dùng PhoBERT pretrained để biến mỗi bài báo thành vector đặc trưng, sau đó huấn luyện hai bộ phân loại:

1. PhoBERT + Logistic Regression
2. PhoBERT + SVM

Nhãn: `0 = Real`, `1 = Fake`. Mỗi bài tạo đúng một vector, dù có một hay nhiều chunk.

## 2. Quy trình

```text
Bài báo → làm sạch nhẹ/tách từ → tokenizer PhoBERT → chia chunk có overlap
         → PhoBERT đóng băng tạo vector chunk → gộp thành vector bài
         → Logistic Regression và SVM → đánh giá cùng tập test
```

## 3. Chuẩn bị dữ liệu

CSV cần có `text` và `label`. Xác nhận nhãn chỉ gồm 0 và 1; bỏ dòng thiếu/rỗng; chuẩn hóa khoảng trắng; loại văn bản trùng hoàn toàn trước khi chia tập. Ghi lại số dòng đã loại và phân bố nhãn.

Giữ dấu tiếng Việt, từ phủ định, tên riêng, số và dấu câu. Không bỏ stopwords, không xóa dấu, không chuyển thành không dấu. Nếu ô chứa cả tiêu đề và thân bài, giữ cả hai.

PhoBERT phù hợp với văn bản tiếng Việt đã tách từ. Dùng cùng một bộ tách từ như `pyvi.ViTokenizer` trong huấn luyện và dự đoán:

```text
Trước: Thủ tướng Abe gửi lời xin lỗi.
Sau:   Thủ_tướng Abe gửi lời xin_lỗi .
```

## 4. Chia dữ liệu để tránh rò rỉ

Chia dữ liệu theo **bài báo**, trước khi tạo chunk. Không chia chunk của cùng bài vào nhiều tập. Có thể giữ 15–20% làm test phân tầng theo nhãn và dùng cross-validation phân tầng trên phần còn lại để chọn cấu hình. Test chỉ dùng đánh giá cuối. Nếu xác định được bài gần trùng/cùng sự kiện, gom nhóm và giữ mỗi nhóm trong cùng fold/tập.

Đặt seed cố định và lưu các chỉ số bài thuộc mỗi split để tái lập kết quả.

## 5. Tokenize và chia chunk

### 5.1 Giới hạn đầu vào

Không suy số token từ số ký tự hay số từ ngăn bằng khoảng trắng. Đếm bằng tokenizer của đúng checkpoint PhoBERT.

Kiểm tra giới hạn từ cấu hình model, ví dụ `model.config.max_position_embeddings`, và tokenizer. Một số tokenizer khai báo `model_max_length` rất lớn như giá trị mặc định; cần đối chiếu với cấu hình model. PhoBERT base thường có giới hạn 256 vị trí, nhưng phải xác nhận checkpoint đang dùng.

Gọi giới hạn tổng là `L`, đã tính token đặc biệt. Nếu tokenizer thêm `S` token đặc biệt thì phần nội dung tối đa là `W = L - S`. Ví dụ `L=256`, `S=2` thì `W=254`. Tokenizer tự thêm token đầu/cuối cho mỗi chunk; đừng chia đủ 256 token nội dung rồi cộng thêm token đặc biệt.

### 5.2 Cửa sổ trượt có overlap

Đặt `W` là số token nội dung tối đa mỗi chunk, `O` là số token overlap (khởi đầu có thể thử 32), và `stride = W - O`.

Minh họa với `W=10`, `O=2`:

```text
Token toàn bài:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
Chunk 1:         1  2  3  4  5  6  7  8  9 10
Chunk 2:                           9 10 11 12 13 14 15
```

Hai token cuối chunk 1 xuất hiện lại ở đầu chunk 2. Phần lặp giữ một chút ngữ cảnh quanh ranh giới. Với `W=254`, `O=32`, mỗi chunk kế tiếp bắt đầu sau `254-32=222` token mới.

Pseudocode chia token nội dung:

```python
def make_chunks(content_ids, window_size, overlap):
    stride = window_size - overlap
    chunks = []
    start = 0
    while start < len(content_ids):
        chunks.append(content_ids[start : start + window_size])
        if start + window_size >= len(content_ids):
            break
        start += stride
    return chunks
```

Tokenize toàn văn bản với `truncation=False`; chia chuỗi token nội dung; sau đó thêm token đặc biệt riêng cho từng chunk bằng API tokenizer, chẳng hạn `prepare_for_model(..., add_special_tokens=True)`. Kiểm tra độ dài cuối cùng mỗi chunk không vượt `L`. Nếu văn bản ngắn hơn `W`, chỉ tạo một chunk.

## 6. Tạo vector cho mỗi chunk

Nạp PhoBERT pretrained làm encoder và đóng băng trọng số:

```python
for parameter in encoder.parameters():
    parameter.requires_grad = False
encoder.eval()
```

Đưa từng chunk qua encoder trong `torch.no_grad()`. PhoBERT trả hidden state cho mỗi token. Dùng masked mean pooling để lấy trung bình vector của token nội dung hợp lệ; loại padding và nên loại token đặc biệt khỏi phép trung bình.

Kích thước vector bằng chiều ẩn của checkpoint (thường 768 với base). Đọc từ cấu hình model thay vì hard-code.

```text
Chunk 1 → PhoBERT → vector chunk 1
Chunk 2 → PhoBERT → vector chunk 2
Chunk 3 → PhoBERT → vector chunk 3
```

## 7. Gộp vector chunk thành vector bài

Mỗi bài có thể có nhiều vector chunk. Gộp chúng thành một vector đại diện bài, ví dụ trung bình có trọng số:

```text
vector_bài = Σ(trọng_số_i × vector_chunk_i) / Σ(trọng_số_i)
```

Với overlap, trọng số nên phản ánh số token **mới** mà chunk đóng góp để phần lặp không bị tính hai lần. Quy tắc khởi đầu: chunk đầu tính toàn bộ token; chunk kế tiếp tính số token sau vùng overlap; chunk cuối tính số token nội dung thực có.

Trung bình vector chunk không giữ thứ tự các đoạn, nhưng đơn giản, tái lập được và phù hợp làm baseline. Kết quả là một vector cho mỗi bài, không phải một vector cho mỗi chunk.

## 8. Huấn luyện Logistic Regression và SVM

Tạo ma trận `X_train` gồm vector bài và `y_train` gồm nhãn tương ứng. Mỗi dòng là một bài báo. Hai bộ phân loại dùng cùng vector, nhãn và folds.

Chuẩn hóa vector nếu cần. `StandardScaler` phải chỉ fit trên phần train của từng fold, sau đó transform validation/test. Đặt scaler trong sklearn Pipeline để tránh leakage:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logreg = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, class_weight="balanced"),
)
svm = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", class_weight="balanced"),
)
```

Các giá trị trên là điểm bắt đầu. Chọn `C` và các tham số khác bằng validation/cross-validation, không dựa trên test. `class_weight="balanced"` cũng cần được kiểm chứng, không mặc định sẽ tốt hơn.

## 9. Đánh giá

Đánh giá cả hai mô hình trên đúng cùng các bài test. Báo cáo:

- Macro-F1 làm chỉ số chính.
- Precision, Recall, F1 riêng cho Real và Fake.
- Accuracy.
- Confusion matrix theo thứ tự `[Real (0), Fake (1)]`.
- Macro-F1 trung bình và độ lệch chuẩn qua các fold.
- Số chunk mỗi bài và tỷ lệ bài phải chia thành nhiều chunk.

Chọn cấu hình bằng cross-validation trên phần phát triển; fit lại trên toàn phần phát triển; dùng test độc lập một lần cuối.

## 10. Các cấu hình chunking nên so sánh

Giữ split và classifier giống nhau, so sánh:

1. Chỉ dùng 256 token đầu.
2. Sliding window không overlap.
3. Sliding window overlap 32 token.

Đây là cấu hình thí nghiệm ban đầu, không phải đáp án tối ưu đã biết trước. Chọn theo Macro-F1 cross-validation rồi mới đánh giá test. Với khoảng 225 văn bản duy nhất, cần diễn giải điểm số thận trọng.

## 11. Checklist

- [ ] Xác nhận nhãn `0=Real`, `1=Fake`.
- [ ] Bỏ mẫu rỗng và bản trùng trước khi chia.
- [ ] Chia theo bài, không theo chunk.
- [ ] Dùng cùng preprocessing/tokenizer/checkpoint ở mọi split.
- [ ] Xác nhận giới hạn token thực tế của checkpoint.
- [ ] Chia token nội dung trước; mỗi chunk sau khi thêm token đặc biệt vẫn trong giới hạn.
- [ ] Đóng băng PhoBERT khi trích vector.
- [ ] Gộp chunk để mỗi bài có một vector.
- [ ] Fit scaler và classifier chỉ trên train từng fold.
- [ ] So sánh hai classifier trên cùng folds.
- [ ] Lưu model, tokenizer, scaler, cấu hình chunking, seed và split.

## 12. Giới hạn của phương pháp

Chunking cho phép xử lý nhiều phần hơn của bài dài, nhưng không giải quyết được dữ liệu ít, nhãn nhiễu, trùng gần đúng hoặc khác biệt nguồn tin. PhoBERT đóng băng cũng không được điều chỉnh riêng cho nhãn Fake/Real. Đây là baseline có kiểm soát; fine-tuning có thể thử riêng nhưng dễ overfit với 226 mẫu.

## 13. Cấu trúc code hiện tại

Code đã triển khai đến mục 7, chưa viết hai bộ phân loại ở mục 8.
Chạy `run.ps1` tại thư mục dự án trên ổ E; hướng dẫn chi tiết ở `RUN.md`.

```text
phobert_features.py              # Điểm chạy chính
phobert_pipeline/
    config.py                    # Đường dẫn, checkpoint, cache trong dự án
    preprocessing.py             # Mục 3: đọc/làm sạch/tách từ
    splitting.py                 # Mục 4: chia theo bài
    chunking.py                  # Mục 5: giới hạn token, chia chunk
    encoder.py                   # Mục 6: PhoBERT, vector chunk
    aggregation.py               # Mục 7: gộp thành vector bài
    pipeline.py                  # Ghép các bước trích đặc trưng
    outputs.py                   # Lưu vector, nhãn, split và cấu hình
```
