"""Kiểm tra logic dữ liệu/chunking, không cần tải PhoBERT."""

import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phobert_pipeline.chunking import input_limit, make_chunks
from phobert_pipeline.preprocessing import load_articles
from phobert_pipeline.aggregation import aggregate_chunks
from phobert_pipeline.splitting import split_articles


class DataAndChunkTests(unittest.TestCase):
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
