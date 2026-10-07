# Implementation Plan: Router phân lớp tin giả (PhoBERT + Lane A LR / Lane B SVM)

## Overview
Xây dựng hệ thống phân loại tin giả tiếng Việt theo kiến trúc **router**: văn bản được mã hóa bằng PhoBERT thành embedding, một **router** quyết định đưa mẫu vào **Lane A** (Logistic Regression) hoặc **Lane B** (SVM), rồi kết quả hai lane được **kết hợp (fusion)** ở cuối để ra nhãn tin thật / tin giả. Mục tiêu là chứng minh router + 2 lane tốt hơn từng mô hình đơn lẻ.

```
Văn bản ─► Tiền xử lý ─► PhoBERT embedding ─► Router ─┬─► Lane A: Logistic Regression ─┐
                                                      └─► Lane B: SVM ─────────────────┴─► Fusion ─► Nhãn
```

Tasks tracked in `tasks/todo.md`.

## Architecture Decisions
- **PhoBERT đóng băng, chỉ trích xuất đặc trưng**: dùng `vinai/phobert-base-v2`, tách từ trước khi mã hóa, cache embedding ra `.npy` để thử lane/router nhanh. Không fine-tune ở phiên bản này.
- **Embedding**: so sánh mean pooling và [CLS] trên validation; văn bản dài hơn 256 token thì chia đoạn rồi lấy trung bình.
- **Cắt lát dọc (vertical slice)**: Task 1-3 dựng sẵn một pipeline chạy được từ văn bản tới nhãn (chỉ LR), sau đó mỗi task thêm một thành phần (SVM, router, fusion) mà hệ thống vẫn chạy được.
- **Router học được dùng nhãn out-of-fold**: nhãn "lane nào đúng hơn" lấy từ dự đoán OOF của hai lane trên tập train để tránh rò rỉ. Có router luật đơn giản (độ dài, độ tự tin của LR) làm baseline.
- **Kiểm tra oracle router sớm (Task 5)**: nếu chọn lane hoàn hảo mà vẫn không hơn LR/SVM đơn lẻ đáng kể thì kiến trúc router không có lợi, nên làm sớm để dừng/đổi hướng sớm (fail fast).
- **Lane A = Logistic Regression** (nhanh, xác suất tốt, hợp mẫu dễ); **Lane B = SVM RBF + `CalibratedClassifierCV`** (biên phi tuyến, hợp mẫu sát biên).
- **Fusion**: thử hard routing (lấy lane được chọn) và soft fusion `p = w·p_A + (1−w)·p_B`, chọn theo validation. Stacking là tùy chọn.
- **Quy tắc đánh giá**: chỉ số chính là macro-F1 và F1 lớp tin giả. Tập test chỉ dùng một lần ở Task 9; mọi chọn lựa dùng validation/CV.
- **Baseline bắt buộc**: LR đơn, SVM đơn, trung bình LR+SVM, oracle router.

## Task List

### Phase 1: Foundation (vertical slice đầu: văn bản → embedding → LR → macro-F1)
- [ ] Task 1: Dựng project và nạp/chia dữ liệu (seed cố định, phân tầng nhãn)
- [ ] Task 2: Tiền xử lý + trích xuất và cache embedding PhoBERT
- [ ] Task 3: Lane A (Logistic Regression) end-to-end với đánh giá trên validation

### Checkpoint: Foundation
- [ ] Tests pass, chạy lại cho kết quả giống nhau, có macro-F1 baseline LR

### Phase 2: Core Features
- [ ] Task 4: Lane B (SVM + hiệu chỉnh xác suất) và bảng so sánh baseline
- [ ] Task 5: Dự đoán out-of-fold và oracle router (cổng quyết định)

### Checkpoint: Quyết định hướng đi
- [ ] Oracle router hơn baseline đơn lẻ đủ lớn để đáng làm router; nếu không, dừng và xem lại với người dùng

- [ ] Task 6: Router luật + hard fusion + pipeline end-to-end
- [ ] Task 7: Router học được (huấn luyện trên nhãn OOF)
- [ ] Task 8: Soft fusion (tùy chọn stacking)

### Checkpoint: Core Features
- [ ] Pipeline chạy end-to-end, có bảng so sánh router/fusion với các baseline trên validation

### Phase 3: Polish
- [ ] Task 9: Đánh giá cuối trên test và phân tích lỗi
- [ ] Task 10: Ablation (pooling, loại router, kiểu fusion)
- [ ] Task 11: Lưu/nạp mô hình, CLI suy luận và README

### Checkpoint: Complete
- [ ] Mọi acceptance criteria đạt, kết quả tái lập được
- [ ] Sẵn sàng review

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Rò rỉ dữ liệu khi tạo nhãn cho router | High | Dùng dự đoán out-of-fold; test chỉ dùng ở Task 9 |
| Router không hơn baseline / oracle gap nhỏ | High | Kiểm tra oracle ở Task 5 trước khi đầu tư router; thử đặc trưng router khác (độ dài, độ tự tin, cụm k-means) |
| Dữ liệu mất cân bằng nhãn | Med | `class_weight`, báo cáo macro-F1 và F1 lớp tin giả thay vì accuracy |
| Tách từ sai làm embedding kém | Med | Dùng bộ tách từ chuẩn tiếng Việt, kiểm tra mẫu thủ công ở Task 2 |
| Bài dài hơn 256 token bị cắt mất thông tin | Med | Chia đoạn và gộp embedding, so với cắt cụt ở Task 10 |
| SVM RBF chậm trên tập lớn | Low | Lưới tham số nhỏ, subsample khi tinh chỉnh, dự phòng `LinearSVC` |
| Embedding đóng băng giới hạn hiệu năng | Low | Ghi nhận giới hạn, cân nhắc fine-tune ở giai đoạn sau |

## Open Questions
- Dataset nào (ví dụ ReINTEL của VLSP 2020, ViFND hoặc dữ liệu tự thu thập)? Kích thước và độ cân bằng nhãn?
- Router dựa trên tiêu chí nào: "lane nào đúng hơn", độ khó/độ tự tin của mẫu, hay đặc điểm cụ thể? Kế hoạch mặc định chọn "lane nào đúng hơn".
- "Kết hợp lại ở sau" là hard (lấy kết quả lane được chọn) hay soft (trộn xác suất)? Kế hoạch thử cả hai.
- Bài toán nhị phân hay nhiều lớp?
- Có GPU để trích xuất embedding không?
- Dự án có dùng issue tracker riêng (GitHub Issues, Linear...) thay cho `tasks/todo.md` không? Mặc định dùng `tasks/todo.md`.
