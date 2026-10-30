"""README mục 7: gộp vector chunk thành một vector đại diện bài."""


def aggregate_chunks(chunk_vectors, chunks):
    import numpy as np

    # Chunk đầu tính toàn bộ token; các chunk sau chỉ tính số token mới.
    # vector_bài = tổng(w_i * vector_chunk_i) / tổng(w_i).
    # Đây là trọng số baseline: overlap vẫn cung cấp ngữ cảnh trong vector chunk.
    weights = [new_tokens for _, new_tokens in chunks]
    return np.average(chunk_vectors, axis=0, weights=weights).astype(np.float32)
