"""Tiền xử lý và xuất CSV từng chunk, không tải trọng số encoder."""

import csv
import json
from pathlib import Path

from .chunking import position_limit, tokenize_chunks
from .config import CHECKPOINT, MODEL_CACHE, configure_runtime


def load_chunk_tokenizer():
    """Chỉ tải tokenizer và config để xác định đúng giới hạn PhoBERT."""
    configure_runtime()
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(CHECKPOINT, cache_dir=MODEL_CACHE)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, cache_dir=MODEL_CACHE, use_fast=False)
    if config.model_type != "roberta" or config.pad_token_id is None:
        raise ValueError("Cần config PhoBERT/RoBERTa có pad_token_id để tính giới hạn vị trí.")
    limit = position_limit(config.max_position_embeddings, config.pad_token_id, tokenizer)
    return tokenizer, limit


def export_chunks(articles, train, tokenizer, limit, overlap, path: Path):
    """Mỗi dòng CSV là một chunk; giữ ID bài và split để tránh lệch nhãn/rò rỉ."""
    train_indices = set(train)
    columns = [
        "article_id", "split", "label", "chunk_id", "start_token", "end_token",
        "content_token_count", "new_token_count", "input_length", "chunk_text", "input_ids",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    # UTF-8 BOM để mở CSV tiếng Việt bằng Excel thuận tiện.
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index, article in enumerate(articles):
            chunks = tokenize_chunks(article["text"], tokenizer, limit, overlap)
            start = 0
            for chunk_id, (tokens, new_tokens) in enumerate(chunks):
                # Thêm token đặc biệt giống bước encoder; input_ids dùng được cho PhoBERT.
                input_ids = tokenizer.build_inputs_with_special_tokens(tokens)
                if len(input_ids) > limit:
                    raise ValueError("Chunk vượt giới hạn vị trí của PhoBERT.")
                writer.writerow({
                    "article_id": article["article_id"],
                    "split": "train" if index in train_indices else "test",
                    "label": article["label"], "chunk_id": chunk_id,
                    "start_token": start, "end_token": start + len(tokens),
                    "content_token_count": len(tokens), "new_token_count": new_tokens,
                    "input_length": len(input_ids),
                    "chunk_text": tokenizer.decode(tokens, clean_up_tokenization_spaces=False),
                    "input_ids": json.dumps(input_ids),
                })
                start += len(tokens) - overlap
                count += 1
    return count
