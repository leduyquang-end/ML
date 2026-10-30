"""Chạy các bước README mục 3–7; dừng trước Logistic Regression và SVM."""

import argparse
import json
from pathlib import Path

from phobert_pipeline.config import ROOT, configure_runtime
from phobert_pipeline.encoder import load_encoder
from phobert_pipeline.outputs import save_outputs
from phobert_pipeline.pipeline import encode_articles
from phobert_pipeline.preprocessing import load_articles
from phobert_pipeline.splitting import split_articles


def main():
    """Điều phối: đọc dữ liệu → chia split → nạp encoder → trích vector → lưu."""
    configure_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/vn_news_226_tlfr.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phobert")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--export-chunks", type=Path, metavar="CSV", help="Xuất dữ liệu sau chunk rồi dừng, không chạy encoder.")
    args = parser.parse_args()
    articles, stats = load_articles(args.data)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.check_data:
        return
    if args.batch_size < 1 or not 0 < args.test_size < 1 or args.overlap < 0:
        parser.error("Cần batch-size > 0, test-size trong (0, 1), overlap >= 0.")

    # Mục 4: chia theo bài trước khi tạo chunk để tránh rò rỉ train/test.
    train, test = split_articles(articles, args.test_size, args.seed)
    if args.export_chunks:
        from phobert_pipeline.export_chunks import export_chunks, load_chunk_tokenizer

        tokenizer, limit = load_chunk_tokenizer()
        count = export_chunks(articles, train, tokenizer, limit, args.overlap, args.export_chunks)
        print(f"Đã xuất {count} chunk từ {len(articles)} bài: {args.export_chunks.resolve()}")
        return
    encoder, tokenizer, device = load_encoder(args.seed)
    print(f"Trích vector bằng {device}; train={len(train)}, test={len(test)}", flush=True)
    features, counts, limit = encode_articles(
        articles, encoder, tokenizer, args.overlap, args.batch_size, device,
    )
    save_outputs(args, articles, stats, features, counts, limit, train, test, encoder, tokenizer)
    print(f"Đã lưu {features.shape} vector vào {args.output}")


if __name__ == "__main__":
    main()
