# Task List: Router phân lớp tin giả
Xem `tasks/plan.md` để biết bối cảnh, quyết định kiến trúc và rủi ro.
Lệnh kiểm thử/build bên dưới là giả định (pytest + script Python); chỉnh lại theo repo thực tế.

---

## Task 1: Dựng project và nạp/chia dữ liệu

**Description:** Tạo cấu trúc project, file cấu hình và môi trường; viết module nạp dataset tin giả, làm sạch cơ bản và chia train/val/test phân tầng theo nhãn với seed cố định.

**Acceptance criteria:**
- [ ] Cấu trúc `src/`, `tests/`, `configs/`, `data/` và file requirements cài được
- [ ] Hàm `load_dataset()` trả về 3 tập train/val/test, tỉ lệ nhãn xấp xỉ nhau giữa các tập
- [ ] Không có mẫu trùng giữa các tập; chạy lại với cùng seed cho đúng cùng phép chia

**Verification:**
- [ ] Tests pass: `pytest tests/test_data.py`
- [ ] Build succeeds: `pip install -r requirements.txt` và `python -c "import src.data"`
- [ ] Manual check: in kích thước và phân bố nhãn từng tập

**Dependencies:** None

**Files likely touched:**
- `requirements.txt`
- `configs/default.yaml`
- `src/data.py`
- `tests/test_data.py`

**Estimated scope:** Small (1-2 file logic, kèm vài file cấu hình)

---

## Task 2: Tiền xử lý và cache embedding PhoBERT

**Description:** Chuẩn hóa văn bản (Unicode, URL, ký tự lạ), tách từ tiếng Việt, mã hóa bằng PhoBERT đóng băng (mean pooling và [CLS]), chia đoạn khi quá 256 token, rồi cache embedding của từng tập ra `.npy`.

**Acceptance criteria:**
- [ ] Mỗi tập có file embedding `(N, 768)` khớp thứ tự với nhãn
- [ ] Chạy lại dùng cache, không tính lại khi đã có
- [ ] Văn bản dài được chia đoạn thay vì bị cắt cụt âm thầm

**Verification:**
- [ ] Tests pass: `pytest tests/test_preprocess.py`
- [ ] Build succeeds: `python -m src.embed --config configs/default.yaml`
- [ ] Manual check: xem 10 mẫu trước/sau tách từ, kiểm tra shape và không có NaN

**Dependencies:** Task 1

**Files likely touched:**
- `src/preprocess.py`
- `src/embed.py`
- `tests/test_preprocess.py`
- `data/embeddings/` (artifact sinh ra, không commit)

**Estimated scope:** Medium (3-4 file)

---

## Task 3: Lane A (Logistic Regression) end-to-end

**Description:** Huấn luyện LR trên embedding đã cache, đánh giá trên validation, in macro-F1 và F1 lớp tin giả. Đây là lát cắt dọc đầu tiên chạy từ dữ liệu tới số liệu.

**Acceptance criteria:**
- [ ] `train_lane_a()` huấn luyện và trả về mô hình + xác suất dự đoán
- [ ] Báo cáo macro-F1, precision/recall/F1 lớp tin giả trên validation
- [ ] Xử lý mất cân bằng nhãn bằng `class_weight`

**Verification:**
- [ ] Tests pass: `pytest tests/test_lanes.py -k lane_a`
- [ ] Build succeeds: `python -m src.train --lane a`
- [ ] Manual check: số liệu cao hơn mức đoán ngẫu nhiên/đa số rõ rệt

**Dependencies:** Task 2

**Files likely touched:**
- `src/lanes.py`
- `src/evaluate.py`
- `tests/test_lanes.py`

**Estimated scope:** Small (2-3 file)

---

## Checkpoint: After Tasks 1-3
- [ ] Tất cả tests pass
- [ ] Pipeline chạy từ dữ liệu thô tới macro-F1 của LR
- [ ] Chạy lại cho kết quả giống nhau (seed cố định)
- [ ] Review với người dùng trước khi tiếp tục

---

## Task 4: Lane B (SVM + hiệu chỉnh xác suất) và bảng baseline

**Description:** Thêm SVM RBF bọc `CalibratedClassifierCV` để có xác suất; tinh chỉnh siêu tham số bằng CV trên tập train; tạo bảng so sánh LR đơn, SVM đơn và trung bình LR+SVM trên validation.

**Acceptance criteria:**
- [ ] `train_lane_b()` trả về mô hình có `predict_proba`
- [ ] Siêu tham số chọn bằng CV, không dùng validation để chọn
- [ ] Bảng so sánh LR / SVM / trung bình LR+SVM được lưu ra `reports/baselines.md`

**Verification:**
- [ ] Tests pass: `pytest tests/test_lanes.py -k lane_b`
- [ ] Build succeeds: `python -m src.train --lane b`
- [ ] Manual check: xác suất SVM nằm trong [0,1] và tổng hai lớp bằng 1

**Dependencies:** Task 3

**Files likely touched:**
- `src/lanes.py`
- `src/evaluate.py`
- `tests/test_lanes.py`
- `reports/baselines.md`

**Estimated scope:** Small (3-4 file, chủ yếu mở rộng file có sẵn)

---

## Task 5: Dự đoán out-of-fold và oracle router (cổng quyết định)

**Description:** Sinh dự đoán OOF của cả hai lane trên tập train (K-fold phân tầng), từ đó tính nhãn "lane nào đúng hơn" cho từng mẫu và tính giới hạn trên của oracle router. Đây là task rủi ro cao nên đặt sớm.

**Acceptance criteria:**
- [ ] Dự đoán OOF của LR và SVM cho toàn bộ tập train, mỗi mẫu được dự đoán bởi mô hình chưa thấy nó
- [ ] Nhãn router (A đúng / B đúng / cả hai đúng / cả hai sai) và tỉ lệ từng loại được báo cáo
- [ ] Macro-F1 của oracle router được so với baseline đơn lẻ trong `reports/oracle.md`

**Verification:**
- [ ] Tests pass: `pytest tests/test_oof.py`
- [ ] Build succeeds: `python -m src.oof --config configs/default.yaml`
- [ ] Manual check: kiểm tra tay vài mẫu thấy nhãn router đúng với dự đoán của hai lane

**Dependencies:** Task 4

**Files likely touched:**
- `src/oof.py`
- `tests/test_oof.py`
- `reports/oracle.md`

**Estimated scope:** Small (2-3 file)

---

## Checkpoint: Quyết định hướng đi (After Tasks 4-5)
- [ ] Tất cả tests pass
- [ ] Oracle router hơn baseline đơn lẻ đủ lớn để đáng làm router
- [ ] Nếu khoảng cách quá nhỏ: dừng và thảo luận với người dùng (đổi tiêu chí router, đổi lane, hoặc bỏ router)

---

## Task 6: Router luật + hard fusion + pipeline end-to-end

**Description:** Cài router luật (ví dụ ngưỡng độ tự tin của LR hoặc độ dài văn bản), hard fusion lấy dự đoán của lane được chọn, và ghép thành `predict(text) -> nhãn, xác suất, lane`.

**Acceptance criteria:**
- [ ] Router nhận embedding, trả về lane được chọn cho từng mẫu
- [ ] Hard fusion trả về nhãn cuối từ lane được chọn
- [ ] `predict()` chạy được từ văn bản thô tới nhãn, kèm thông tin lane đã dùng
- [ ] Báo cáo macro-F1 trên validation và tỉ lệ mẫu đi vào mỗi lane

**Verification:**
- [ ] Tests pass: `pytest tests/test_router.py tests/test_pipeline.py`
- [ ] Build succeeds: `python -m src.pipeline --demo`
- [ ] Manual check: thử 5 câu ví dụ, xem lane được chọn có hợp lý

**Dependencies:** Task 5

**Files likely touched:**
- `src/router.py`
- `src/fusion.py`
- `src/pipeline.py`
- `tests/test_router.py`
- `tests/test_pipeline.py`

**Estimated scope:** Medium (5 file)

---

## Task 7: Router học được

**Description:** Huấn luyện classifier nhỏ trên embedding để dự đoán lane tốt hơn, dùng nhãn OOF từ Task 5; so sánh với router luật và oracle.

**Acceptance criteria:**
- [ ] Router học được chỉ huấn luyện trên tập train với nhãn OOF
- [ ] Có thể thay router luật bằng router học được qua cấu hình, không đổi code pipeline
- [ ] Báo cáo độ chính xác chọn lane và macro-F1 cuối so với router luật và oracle

**Verification:**
- [ ] Tests pass: `pytest tests/test_router.py -k learned`
- [ ] Build succeeds: `python -m src.train --router learned`
- [ ] Manual check: phân bố lane được chọn không suy biến về một lane duy nhất

**Dependencies:** Task 6

**Files likely touched:**
- `src/router.py`
- `configs/default.yaml`
- `tests/test_router.py`

**Estimated scope:** Small (3 file)

---

## Task 8: Soft fusion (tùy chọn stacking)

**Description:** Thêm soft fusion `p = w·p_A + (1−w)·p_B` với `w` lấy từ xác suất của router; tùy chọn thêm meta-learner (stacking). So sánh với hard fusion.

**Acceptance criteria:**
- [ ] Kiểu fusion chọn được qua cấu hình (`hard` / `soft` / `stack`)
- [ ] Xác suất cuối hợp lệ trong [0,1]
- [ ] Bảng so sánh hard vs soft (vs stack nếu có) trên validation trong `reports/fusion.md`

**Verification:**
- [ ] Tests pass: `pytest tests/test_fusion.py`
- [ ] Build succeeds: `python -m src.pipeline --fusion soft --demo`
- [ ] Manual check: với w=1 và w=0 thì kết quả trùng đúng Lane A / Lane B

**Dependencies:** Task 7

**Files likely touched:**
- `src/fusion.py`
- `tests/test_fusion.py`
- `reports/fusion.md`

**Estimated scope:** Small (3 file)

---

## Checkpoint: After Tasks 6-8 (Core Features)
- [ ] Tất cả tests pass
- [ ] Pipeline chạy end-to-end với router luật/học được và fusion hard/soft
- [ ] Có bảng so sánh với LR, SVM, trung bình LR+SVM và oracle trên validation
- [ ] Review với người dùng trước khi sang Polish

---

## Task 9: Đánh giá cuối trên test và phân tích lỗi

**Description:** Chốt cấu hình tốt nhất dựa trên validation, chạy đúng một lần trên tập test, tạo báo cáo cuối và phân tích lỗi (mẫu router chọn sai lane, mẫu cả hai lane đều sai, phân bố lane theo độ dài/chủ đề).

**Acceptance criteria:**
- [ ] Cấu hình được chốt trước khi chạy test, không chỉnh lại sau khi thấy kết quả test
- [ ] Báo cáo macro-F1, F1 lớp tin giả và ma trận nhầm lẫn cho mọi baseline và cho router
- [ ] Có mục phân tích lỗi với ít nhất 3 nhóm lỗi và ví dụ

**Verification:**
- [ ] Tests pass: `pytest`
- [ ] Build succeeds: `python -m src.evaluate --split test`
- [ ] Manual check: đọc lại 20 mẫu lỗi, xác nhận nhóm lỗi hợp lý

**Dependencies:** Task 8

**Files likely touched:**
- `src/evaluate.py`
- `notebooks/error_analysis.ipynb`
- `reports/final.md`

**Estimated scope:** Small (3 file)

---

## Task 10: Ablation

**Description:** Chạy ablation có kiểm soát: mean pooling vs [CLS], chia đoạn vs cắt cụt, router luật vs học được, hard vs soft fusion; chỉ dùng validation/CV.

**Acceptance criteria:**
- [ ] Mỗi ablation thay đổi đúng một yếu tố, cùng seed và cùng phép chia
- [ ] Bảng kết quả ablation trong `reports/ablation.md`
- [ ] Nêu rõ yếu tố nào đóng góp nhiều nhất

**Verification:**
- [ ] Tests pass: `pytest tests/test_ablation.py`
- [ ] Build succeeds: `python -m src.ablation --config configs/ablation.yaml`
- [ ] Manual check: kết quả cấu hình gốc trong ablation trùng với kết quả ở Task 8

**Dependencies:** Task 8

**Files likely touched:**
- `src/ablation.py`
- `configs/ablation.yaml`
- `tests/test_ablation.py`
- `reports/ablation.md`

**Estimated scope:** Medium (4 file)

---

## Task 11: Lưu/nạp mô hình, CLI suy luận và README

**Description:** Lưu toàn bộ pipeline (lane, router, cấu hình fusion) bằng joblib, viết CLI `predict` nhận văn bản và in nhãn + lane + xác suất, và README hướng dẫn tái lập kết quả.

**Acceptance criteria:**
- [ ] Nạp lại mô hình đã lưu cho đúng dự đoán như trước khi lưu
- [ ] CLI nhận một đoạn văn bản hoặc file và in nhãn, xác suất, lane
- [ ] README mô tả cài đặt, huấn luyện, đánh giá, suy luận và seed

**Verification:**
- [ ] Tests pass: `pytest tests/test_cli.py`
- [ ] Build succeeds: `python -m src.cli predict --text "..."`
- [ ] Manual check: làm theo README trên máy sạch/venv mới tới khi ra kết quả

**Dependencies:** Task 9

**Files likely touched:**
- `src/pipeline.py`
- `src/cli.py`
- `tests/test_cli.py`
- `README.md`

**Estimated scope:** Small (4 file)

---

## Checkpoint: Complete
- [ ] Tất cả acceptance criteria đạt
- [ ] Router + fusion vượt các baseline đơn lẻ trên macro-F1, hoặc có giải thích rõ nếu không
- [ ] Kết quả tái lập được từ README
- [ ] Sẵn sàng review
