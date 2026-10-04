import types
import unittest
import uuid
from unittest.mock import Mock

import chromadb

from memhawk import MemHawk
from memhawk.v2 import MemHawkV2


class SearchContextTests(unittest.TestCase):
    def make_engine(self, engine_class, max_distance=1.2):
        collection = chromadb.Client().create_collection(f"search-{uuid.uuid4().hex}")
        embeddings = Mock()
        embeddings.create.return_value = types.SimpleNamespace(
            data=[types.SimpleNamespace(embedding=[1.0, 0.0])]
        )
        client = types.SimpleNamespace(embeddings=embeddings)
        if engine_class is MemHawkV2:
            engine = MemHawkV2(
                api_client=client,
                collection=collection,
                embed_model="test-model",
                top_k_retrieval=2,
                retrieval_per_query_k=3,
                max_retrieval_distance=max_distance,
                recency_tiebreak_weight=0.0,
            )
        else:
            engine = MemHawk.__new__(MemHawk)
            engine.api_client = client
            engine.collection = collection
            engine.embed_model = "test-model"
            engine.top_k_retrieval = 2
            engine.retrieval_per_query_k = 3
            engine.max_retrieval_distance = max_distance
        collection.add(
            ids=["near", "next", "far"],
            documents=["Primary database: PostgreSQL.", "Database backups: daily.", "Unrelated fact."],
            embeddings=[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
            metadatas=[{"namespace": "default", "status": "active"} for _ in range(3)],
        )
        return engine, embeddings, collection

    def test_query_only_search_ranks_filters_and_limits_without_writing(self):
        for engine_class in (MemHawk, MemHawkV2):
            with self.subTest(engine=engine_class.__name__):
                engine, embeddings, collection = self.make_engine(engine_class)
                query = "Which database do we use?"
                self.assertEqual(
                    engine.search_context(query),
                    ["Primary database: PostgreSQL.", "Database backups: daily."],
                )
                embeddings.create.assert_called_once_with(model="test-model", input=query)
                self.assertEqual(collection.count(), 3)
                self.assertEqual(engine.search_context(query, top_k=1), ["Primary database: PostgreSQL."])
                self.assertEqual(len(engine.search_context(query, top_k=3)), 2)
                engine.max_retrieval_distance = -1.0
                self.assertEqual(engine.search_context(query), [])

    def test_disabled_distance_threshold_keeps_all_candidates(self):
        for engine_class in (MemHawk, MemHawkV2):
            with self.subTest(engine=engine_class.__name__):
                engine, _, _ = self.make_engine(engine_class, max_distance=None)
                self.assertEqual(len(engine.search_context("database", top_k=3)), 3)

    def test_empty_custom_collection_skips_embedding(self):
        for engine_class in (MemHawk, MemHawkV2):
            with self.subTest(engine=engine_class.__name__):
                engine, embeddings, _ = self.make_engine(engine_class)
                empty = chromadb.Client().create_collection(f"empty-{uuid.uuid4().hex}")
                self.assertEqual(engine.search_context("database", collection=empty), [])
                embeddings.create.assert_not_called()

    def test_blank_query_and_nonpositive_limit_skip_embedding(self):
        for engine_class in (MemHawk, MemHawkV2):
            with self.subTest(engine=engine_class.__name__):
                engine, embeddings, _ = self.make_engine(engine_class)
                for query in ("", " \n\t"):
                    with self.assertRaises(ValueError):
                        engine.search_context(query)
                for top_k in (0, -1):
                    self.assertEqual(engine.search_context("database", top_k=top_k), [])
                embeddings.create.assert_not_called()

    def test_v2_search_excludes_superseded_and_other_namespaces(self):
        engine, _, collection = self.make_engine(MemHawkV2)
        collection.update(ids=["near"], metadatas=[{"status": "superseded"}])
        collection.update(ids=["next"], metadatas=[{"namespace": "other-user"}])
        self.assertEqual(engine.search_context("database"), [])


if __name__ == "__main__":
    unittest.main()
