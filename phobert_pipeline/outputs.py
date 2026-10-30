"""Lưu đặc trưng, split và encoder để tái lập thí nghiệm."""

import csv
import json

from .config import CHECKPOINT


def save_embeddings_csv(path, articles, features, counts, train):
    """Xuất vector bài để review: mỗi hàng là một bài, mỗi feature là một số thực."""
    train_indices = set(train)
    # Đọc số chiều thực tế, không hard-code 768 để tránh lệch checkpoint.
    columns = ["article_id", "split", "label", "chunk_count"]
    columns += [f"feature_{i}" for i in range(features.shape[1])]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for index, (article, vector, count) in enumerate(zip(articles, features, counts, strict=True)):
            # Vector được gộp từ các chunk bằng trọng số số token mới như README mục 7.
            writer.writerow([
                article["article_id"], "train" if index in train_indices else "test",
                article["label"], int(count), *vector.tolist(),
            ])


def save_outputs(args, articles, stats, features, counts, limit, train, test, encoder, tokenizer):
    """Lưu vector, ID split và cấu hình để bước phân loại dùng lại đúng dữ liệu."""
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=True)
    article_ids = np.array([a["article_id"] for a in articles])
    labels = np.array([a["label"] for a in articles], dtype=np.int64)
    # Cùng thứ tự chỉ số cho X, y, ID và số chunk để không lệch nhãn.
    arrays = {}
    for name, indices in (("train", train), ("test", test)):
        arrays.update({
            f"X_{name}": features[indices], f"y_{name}": labels[indices],
            f"{name}_ids": article_ids[indices], f"{name}_chunk_counts": counts[indices],
        })
    np.savez_compressed(args.output / "features.npz", **arrays)
    save_embeddings_csv(args.output / "embeddings.csv", articles, features, counts, train)
    metadata = {
        "checkpoint": CHECKPOINT, "revision": encoder.config._commit_hash,
        "seed": args.seed, "test_size": args.test_size, "overlap": args.overlap,
        "max_length": limit, "hidden_size": encoder.config.hidden_size,
        "preprocessing": "NFC + whitespace + pyvi.ViTokenizer",
        "pooling": "content masked mean; chunk mean weighted by new token count",
        "labels": {"0": "Real", "1": "Fake"}, "data_stats": stats,
        "train_ids": article_ids[train].tolist(), "test_ids": article_ids[test].tolist(),
        "chunk_counts": dict(zip(map(str, article_ids), counts.tolist())),
        "multi_chunk_ratio": float((counts > 1).mean()),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    with (args.output / "articles.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["article_id", "text", "label"])
        writer.writeheader()
        writer.writerows(articles)
    # Giữ encoder/tokenizer để lần dự đoán sau dùng đúng bộ trích đặc trưng này.
    encoder.save_pretrained(args.output / "encoder")
    tokenizer.save_pretrained(args.output / "encoder")
