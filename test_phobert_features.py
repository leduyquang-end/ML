"""Kiểm tra logic dữ liệu/chunking, không cần tải PhoBERT."""

import io
import csv
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phobert_pipeline.chunking import input_limit, make_chunks
from phobert_pipeline.preprocessing import load_articles
from phobert_pipeline.aggregation import aggregate_chunks
from phobert_pipeline.splitting import split_articles
from phobert_pipeline.export_chunks import export_chunks
from phobert_pipeline.config import configure_runtime
from phobert_pipeline.preprocessing import segment_text
from phobert_pipeline.outputs import save_embeddings_csv


class DataAndChunkTests(unittest.TestCase):
    def test_embedding_csv_preserves_vectors_labels_and_splits(self):
        import numpy as np

        articles = [{"article_id": 7, "label": 1}, {"article_id": 2, "label": 0}]
        features = np.array([[1.25, -2.5], [0., 3.75]], dtype=np.float32)
        buffer = io.StringIO()
        with patch.object(Path, "open") as opener:
            opener.return_value.__enter__.return_value = buffer
            save_embeddings_csv(Path("embeddings.csv"), articles, features, [3, 1], [1])
        rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))
        self.assertEqual(list(rows[0]),
                         ["article_id", "split", "label", "chunk_count", "feature_0", "feature_1"])
        self.assertEqual([(r["article_id"], r["split"], r["label"]) for r in rows],
                         [("7", "test", "1"), ("2", "train", "0")])
        np.testing.assert_array_equal(
            [[float(r["feature_0"]), float(r["feature_1"])] for r in rows], features,
        )

    def test_segmenter_supports_vietnamese_project_path(self):
        configure_runtime()
        self.assertEqual(segment_text("Thủ tướng gửi lời xin lỗi."),
                         "Thủ_tướng gửi lời xin_lỗi .")

    def test_export_keeps_article_split_and_exact_token_ids(self):
        buffer = io.StringIO()
        tokenizer = SimpleNamespace(
            build_inputs_with_special_tokens=lambda ids: [0] + ids + [2],
            decode=lambda ids, **kwargs: " ".join(map(str, ids)),
        )
        articles = [{"article_id": 5, "text": "A", "label": 1},
                    {"article_id": 8, "text": "B", "label": 0}]
        chunks = [[([4, 5, 6, 7], 4), ([6, 7, 8], 1)], [([9], 1)]]
        with patch.object(Path, "mkdir"), patch.object(Path, "open") as opener, patch(
            "phobert_pipeline.export_chunks.tokenize_chunks", side_effect=chunks,
        ):
            opener.return_value.__enter__.return_value = buffer
            count = export_chunks(articles, [0], tokenizer, 6, 2, Path("chunks.csv"))
        rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))
        self.assertEqual(count, 3)
        self.assertEqual([row["split"] for row in rows], ["train", "train", "test"])
        self.assertEqual([row["start_token"] for row in rows], ["0", "2", "0"])
        self.assertEqual([row["end_token"] for row in rows], ["4", "5", "1"])
        self.assertEqual([json.loads(row["input_ids"]) for row in rows],
                         [[0, 4, 5, 6, 7, 2], [0, 6, 7, 8, 2], [0, 9, 2]])

    def test_weighted_article_vector(self):
        import numpy as np

        chunks = make_chunks(list(range(15)), 10, 2)
        vector = aggregate_chunks(np.array([[1., 0.], [0., 1.]]), chunks)
        np.testing.assert_allclose(vector, [2 / 3, 1 / 3])
        self.assertEqual(vector.shape, (2,))

    def test_split_is_reproducible_and_disjoint(self):
        import numpy as np

        articles, _ = load_articles(Path(__file__).parent / "data/vn_news_226_tlfr.csv")
        train, test = split_articles(articles, 0.2, 42)
        again_train, again_test = split_articles(articles, 0.2, 42)
        np.testing.assert_array_equal(train, again_train)
        np.testing.assert_array_equal(test, again_test)
        self.assertEqual((len(train), len(test)), (180, 45))
        self.assertFalse(set(train) & set(test))

    def load_csv(self, text):
        with patch.object(Path, "open", return_value=io.StringIO(text)):
            return load_articles("sample.csv")

    def test_clean_and_preserve_original_ids(self):
        articles, stats = self.load_csv(
            'text,label\n" A  B ",0\nA B,0\n,1\nC,1\n'
        )
        self.assertEqual([a["article_id"] for a in articles], [0, 3])
        self.assertEqual(articles[0]["text"], "A B")
        self.assertEqual((stats["duplicate"], stats["empty"]), (1, 1))

    def test_reject_invalid_data(self):
        for text in ["text,label\nA,0\nA,1\n", "text,label\nA,2\n",
                     "text,label\n,\n", "wrong,label\nA,0\n"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load_csv(text)

    def test_chunks_cover_all_tokens_and_weight_each_new_token_once(self):
        for length in [1, 10, 11, 18, 19, 254, 255, 476, 477, 1000]:
            for overlap in [0, 2, 9]:
                with self.subTest(length=length, overlap=overlap):
                    tokens = list(range(length))
                    chunks = make_chunks(tokens, 10, overlap)
                    recovered = chunks[0][0][:]
                    for chunk, weight in chunks[1:]:
                        recovered.extend(chunk[overlap:])
                        self.assertEqual(weight, len(chunk) - overlap)
                    self.assertEqual(recovered, tokens)
                    self.assertEqual(sum(w for _, w in chunks), length)
                    self.assertTrue(all(0 < len(c) <= 10 and w > 0 for c, w in chunks))

    def test_exact_boundary_has_no_extra_tail(self):
        self.assertEqual(len(make_chunks(list(range(18)), 10, 2)), 2)
        self.assertEqual(make_chunks([1, 2], 10, 2), [([1, 2], 2)])

    def test_reject_invalid_chunks(self):
        for tokens, window, overlap in [([], 10, 2), ([1], 10, -1), ([1], 10, 10)]:
            with self.subTest(overlap=overlap), self.assertRaises(ValueError):
                make_chunks(tokens, window, overlap)

    def test_roberta_position_offset_and_tokenizer_limit(self):
        encoder = SimpleNamespace(embeddings=SimpleNamespace(
            position_embeddings=SimpleNamespace(num_embeddings=258, padding_idx=1)))
        self.assertEqual(input_limit(encoder, SimpleNamespace(model_max_length=10**30)), 256)
        self.assertEqual(input_limit(encoder, SimpleNamespace(model_max_length=128)), 128)


if __name__ == "__main__":
    unittest.main()
