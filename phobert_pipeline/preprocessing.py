"""README mục 3: đọc, làm sạch, kiểm tra nhãn và tách từ tiếng Việt."""

import csv
import os
import unicodedata
from collections import Counter
from pathlib import Path
from typing import TypedDict


class Article(TypedDict):
    article_id: int
    text: str
    label: int


class DataStats(TypedDict):
    total: int
    empty: int
    duplicate: int
    kept: int
    labels: dict[int, int]


def segment_text(text: str) -> str:
    """Dùng chung PyVi cho bước trích đặc trưng và dự đoán sau này."""
    from pyvi import ViTokenizer

    # CRFsuite trên Windows không mở được tên file chứa dấu (ví dụ "học máy").
    # Python đọc được file này: nạp bytes vào RAM một lần, không đổi cache sang ổ C.
    model = ViTokenizer.ViTokenizer.model
    if os.name == "nt" and model._tagger is None:
        import pycrfsuite

        # CRFsuite giữ con trỏ đến bytes, nên phải giữ buffer cùng vòng đời model.
        model._phobert_crf_bytes = Path(model.modelfile.name).read_bytes()
        tagger = getattr(pycrfsuite, "Tagger")()
        tagger.open_inmemory(model._phobert_crf_bytes)
        model._tagger = tagger
    return ViTokenizer.tokenize(text)


def load_articles(path: str | Path) -> tuple[list[Article], DataStats]:
    """Trả danh sách bài sạch và thống kê; ID là chỉ số bản ghi gốc từ 0."""
    articles: list[Article] = []
    seen: dict[str, int] = {}
    stats = Counter(total=0, empty=0, duplicate=0)
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV phải có hai cột text và label.")
        for article_id, row in enumerate(reader):
            stats["total"] += 1
            # NFC thống nhất cách mã hóa dấu; split/join gộp khoảng trắng/xuống dòng.
            # Giữ nguyên nội dung tiêu đề + thân bài, chữ hoa, dấu câu và phủ định.
            text = " ".join(unicodedata.normalize("NFC", row["text"] or "").split())
            raw_label = (row["label"] or "").strip()
            if not text or not raw_label:
                stats["empty"] += 1
                continue
            if raw_label not in {"0", "1"}:
                raise ValueError(f"Bản ghi {article_id}: nhãn không hợp lệ {raw_label!r}.")
            label = int(raw_label)
            # Loại trùng trước khi chia tập để cùng bài không lọt vào cả train/test.
            # Khác nhãn thì dừng để kiểm tra dữ liệu, không tự chọn một nhãn.
            if text in seen:
                if seen[text] != label:
                    raise ValueError(f"Bản ghi {article_id}: bài trùng nhưng khác nhãn.")
                stats["duplicate"] += 1
                continue
            seen[text] = label
            articles.append({"article_id": article_id, "text": text, "label": label})
    if not articles:
        raise ValueError("Không còn bài hợp lệ sau khi làm sạch.")
    return articles, {
        "total": stats["total"], "empty": stats["empty"],
        "duplicate": stats["duplicate"], "kept": len(articles),
        "labels": dict(Counter(a["label"] for a in articles)),
    }
