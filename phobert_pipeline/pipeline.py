"""Ghép các bước mục 3–7; mỗi bài trả đúng một vector đặc trưng."""

from .aggregation import aggregate_chunks
from .chunking import input_limit, tokenize_chunks
from .encoder import encode_batch


def encode_articles(articles, encoder, tokenizer, overlap, batch_size, device):
    """Masked mean trên token nội dung, rồi weighted mean trên các chunk."""
    import numpy as np

    limit = input_limit(encoder, tokenizer)
    vectors, counts = [], []
    for article in articles:
        chunks = tokenize_chunks(article["text"], tokenizer, limit, overlap)
        # Xử lý từng batch nhỏ trên GPU/CPU; chỉ giữ vector chunk trên RAM.
        chunk_vectors = np.concatenate([
            encode_batch(chunks[start:start + batch_size], encoder, tokenizer, limit, device)
            for start in range(0, len(chunks), batch_size)
        ])
        vectors.append(aggregate_chunks(chunk_vectors, chunks))
        counts.append(len(chunks))
        if len(vectors) % 25 == 0:
            print(f"Đã trích vector {len(vectors)}/{len(articles)} bài", flush=True)
    features = np.stack(vectors)
    if not np.isfinite(features).all():
        raise ValueError("Vector đầu ra chứa NaN hoặc Inf.")
    return features, np.array(counts), limit
