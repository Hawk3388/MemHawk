import math
import types
import unittest

from memhawk import MemHawk


class RetrievalEmbeddingTests(unittest.TestCase):
    def make_memhawk(self, prompt_weight=0.8, history_decay=0.7):
        memhawk = MemHawk.__new__(MemHawk)
        memhawk.current_prompt_weight = prompt_weight
        memhawk.history_decay = history_decay
        return memhawk

    def test_prompt_dominates_opposing_history(self):
        memhawk = self.make_memhawk()

        result = memhawk.create_retrieval_embedding(
            prompt_embedding=[1.0, 0.0],
            history_embeddings=[[0.0, 1.0], [0.0, 1.0]],
        )

        self.assertGreater(result[0], result[1] * 3)

    def test_recent_history_guides_more_than_old_history(self):
        memhawk = self.make_memhawk(prompt_weight=0.8, history_decay=0.5)

        result = memhawk.create_retrieval_embedding(
            prompt_embedding=[1.0, 0.0, 0.0],
            history_embeddings=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )

        self.assertGreater(result[2], result[1])
        self.assertGreater(result[0], result[1] + result[2])

    def test_blending_preserves_embedding_magnitude(self):
        memhawk = self.make_memhawk(prompt_weight=0.8)

        result = memhawk.create_retrieval_embedding(
            prompt_embedding=[1.0, 0.0],
            history_embeddings=[[0.0, 1.0]],
        )

        magnitude = math.sqrt(sum(value * value for value in result))
        self.assertAlmostEqual(magnitude, 1.0)

    def test_no_history_returns_prompt_unchanged(self):
        memhawk = self.make_memhawk()
        prompt = [0.25, -0.75]

        self.assertEqual(memhawk.create_retrieval_embedding(prompt), prompt)

    def test_retrieve_context_uses_prompt_first_embedding(self):
        memhawk = self.make_memhawk(prompt_weight=0.8, history_decay=1.0)
        memhawk.embed_model = "test-model"
        memhawk.retrieval_per_query_k = 5
        memhawk.top_k_retrieval = 3
        memhawk.max_retrieval_distance = 1.2

        class EmbeddingItem:
            def __init__(self, embedding):
                self.embedding = embedding

        class EmbeddingsClient:
            def create(self, model, input):
                self.input = input
                return types.SimpleNamespace(
                    data=[EmbeddingItem([0.0, 1.0]), EmbeddingItem([1.0, 0.0])]
                )

        class Collection:
            def count(self):
                return 1

            def query(self, **kwargs):
                self.query_arguments = kwargs
                return {"documents": [["memory"]], "distances": [[0.2]]}

        embeddings_client = EmbeddingsClient()
        memhawk.api_client = types.SimpleNamespace(embeddings=embeddings_client)
        collection = Collection()

        result = memhawk.retrieve_context(
            "current request",
            history=[{"role": "assistant", "content": "older context"}],
            collection=collection,
        )

        query_vector = collection.query_arguments["query_embeddings"][0]
        self.assertEqual(
            embeddings_client.input,
            ["Assistant: older context", "User: current request"],
        )
        self.assertGreater(query_vector[0], query_vector[1] * 3)
        self.assertEqual(result, ["memory"])


if __name__ == "__main__":
    unittest.main()
