"""README mục 5: xác định giới hạn đầu vào và chia token có overlap."""

from .preprocessing import segment_text


def tokenize_chunks(text, tokenizer, limit, overlap):
    """Tách từ → tokenize toàn bài → chia token nội dung, chưa thêm <s>/</s>."""
    # Ví dụ tổng L=256, hai token đặc biệt => cửa sổ nội dung W=254.
    window = limit - tokenizer.num_special_tokens_to_add(pair=False)
    # Không truncate: giữ toàn bài dài để chia chunk, không bỏ phần đuôi.
    ids = tokenizer.encode(segment_text(text), add_special_tokens=False, truncation=False)
    return make_chunks(ids, window, overlap)


def make_chunks(
    content_ids: list[int], window_size: int, overlap: int,
) -> list[tuple[list[int], int]]:
    """Chia token nội dung thành các cặp (chunk, trọng số = số token mới).

    Ví dụ W=10, overlap=2, bài 15 token: hai chunk dài 10 và 7,
    trọng số lần lượt 10 và 5. Tổng trọng số bằng số token toàn bài.
    """
    if not 0 <= overlap < window_size:
        raise ValueError("Cần 0 <= overlap < window_size.")
    if not content_ids:
        raise ValueError("Bài không có token nội dung.")
    chunks = []
    for start in range(0, len(content_ids), window_size - overlap):
        chunk = content_ids[start:start + window_size]
        # Chunk sau đã lặp overlap token từ chunk trước; chỉ đếm phần mới.
        new_tokens = len(chunk) if start == 0 else len(chunk) - overlap
        chunks.append((chunk, new_tokens))
        if start + window_size >= len(content_ids):
            # Đã phủ hết bài: dừng ngay, tránh tạo thêm chunk chỉ chứa phần lặp.
            break
    return chunks


def input_limit(encoder, tokenizer):
    """Lấy giới hạn tổng, bao gồm cả token đặc biệt, của checkpoint đang nạp."""
    # RoBERTa đánh số vị trí từ padding_idx + 1: 258 - 1 - 1 = 256.
    positions = encoder.embeddings.position_embeddings
    limit = positions.num_embeddings - positions.padding_idx - 1
    # Tokenizer có thể khai báo số cực lớn; giới hạn bảng vị trí vẫn được áp dụng.
    return min(limit, tokenizer.model_max_length)
