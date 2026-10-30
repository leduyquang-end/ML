"""README mục 4: chia train/test theo bài trước khi tạo chunk."""

from .preprocessing import Article


def split_articles(articles: list[Article], test_size: float, seed: int):
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Phân tầng giữ tỷ lệ nhãn; seed cố định cho phép tái lập cùng một split.
    # Trả chỉ số bài, không chia các chunk của cùng bài sang hai tập khác nhau.
    labels = [article["label"] for article in articles]
    return train_test_split(
        np.arange(len(articles)), test_size=test_size,
        random_state=seed, stratify=labels,
    )
