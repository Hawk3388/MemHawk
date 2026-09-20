import math
import types
import unittest
import uuid
from datetime import datetime, timedelta, timezone

import chromadb

from memhawk.v2 import MemHawkV2, MemoryCandidate


class KeywordEmbeddings:
    def __init__(self):
        self.calls = 0
        self.last_input = None

    def create(self, model, input):
        self.calls += 1
        self.last_input = input
        texts = [input] if isinstance(input, str) else input
        return types.SimpleNamespace(
            data=[
                types.SimpleNamespace(embedding=self._embed(text))
                for text in texts
            ]
        )

    def _embed(self, text):
        lowered = text.lower()
        if any(word in lowered for word in ("database", "mysql", "postgresql")):
            vector = [1.0, 0.0, 0.0]
        elif "python" in lowered:
            vector = [0.0, 1.0, 0.0]
        else:
            vector = [0.0, 0.0, 1.0]
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector]


class MemHawkV2Tests(unittest.TestCase):
    def make_engine(self, collection=None, namespace="test", **kwargs):
        collection = collection or chromadb.Client().get_or_create_collection(
            f"test-{uuid.uuid4().hex}"
        )
        embeddings = KeywordEmbeddings()
        kwargs.setdefault("max_retrieval_distance", 1.2)
        engine = MemHawkV2(
            api_client=types.SimpleNamespace(embeddings=embeddings),
            collection=collection,
            namespace=namespace,
            embed_model="test-model",
            **kwargs,
        )
        return engine, embeddings, collection

    def test_new_version_supersedes_old_version(self):
        engine, _, collection = self.make_engine()
        old_id = engine.remember(
            "The primary database is MySQL.",
            memory_key="infrastructure.primary_database",
        )
        new_id = engine.remember(
            "The primary database is PostgreSQL.",
            memory_key="infrastructure.primary_database",
        )

        stored = collection.get(ids=[old_id, new_id], include=["metadatas"])
        metadata_by_id = dict(zip(stored["ids"], stored["metadatas"]))
        self.assertEqual(metadata_by_id[old_id]["status"], "superseded")
        self.assertEqual(metadata_by_id[new_id]["status"], "active")
        self.assertEqual(metadata_by_id[new_id]["version"], 2)

        retrieved = engine.retrieve_context("Which database is primary?")
        self.assertEqual(retrieved, ["The primary database is PostgreSQL."])

    def test_duplicate_is_skipped_before_second_embedding_call(self):
        engine, embeddings, collection = self.make_engine()
        document = "Use Black to format Python code."

        first_id = engine.remember(document, memory_key="python.formatter")
        second_id = engine.remember(document, memory_key="python.formatter")

        self.assertEqual(first_id, second_id)
        self.assertEqual(collection.count(), 1)
        self.assertEqual(embeddings.calls, 1)

    def test_namespaces_do_not_leak(self):
        collection = chromadb.Client().get_or_create_collection(
            f"test-{uuid.uuid4().hex}"
        )
        engine_a, _, _ = self.make_engine(collection=collection, namespace="alice")
        engine_b, _, _ = self.make_engine(collection=collection, namespace="bob")
        engine_a.remember("Alice uses PostgreSQL.", memory_key="database")
        engine_b.remember("Bob uses MySQL.", memory_key="database")

        self.assertEqual(
            engine_a.retrieve_context("Which database does Alice use?"),
            ["Alice uses PostgreSQL."],
        )
        self.assertEqual(
            engine_b.retrieve_context("Which database does Bob use?"),
            ["Bob uses MySQL."],
        )

    def test_repeated_history_archival_is_idempotent(self):
        engine, embeddings, collection = self.make_engine(max_live_user_turns=0)
        history = [
            {
                "role": "user",
                "content": "Which database should we use?",
                "metadata": {"memory_key": "database"},
            },
            {"role": "assistant", "content": "Use PostgreSQL."},
        ]

        engine.archive_oldest_pair_if_needed(history)
        engine.archive_oldest_pair_if_needed(history)

        self.assertEqual(collection.count(), 1)
        self.assertEqual(embeddings.calls, 1)

    def test_expired_and_future_candidates_are_filtered(self):
        engine, _, _ = self.make_engine(max_retrieval_distance=None)
        now = datetime.now(timezone.utc)
        base = {
            "id": "memory",
            "document": "document",
            "distance": 0.1,
            "metadata": {},
            "status": "active",
            "version": 1,
            "ranking_score": 0.1,
        }
        active = MemoryCandidate(
            **base,
            memory_key="active",
            valid_from=now - timedelta(days=1),
        )
        expired = MemoryCandidate(
            **{**base, "id": "expired"},
            memory_key="expired",
            valid_until=now - timedelta(seconds=1),
        )
        future = MemoryCandidate(
            **{**base, "id": "future"},
            memory_key="future",
            valid_from=now + timedelta(days=1),
        )

        selected = engine.select_valid_candidates(
            [expired, future, active],
            now=now,
        )

        self.assertEqual([candidate.id for candidate in selected], ["memory"])

    def test_query_embedding_limits_history_messages(self):
        engine, embeddings, _ = self.make_engine(
            max_history_embedding_messages=3
        )
        history = [
            {"role": "user", "content": f"message {index}"}
            for index in range(8)
        ]

        engine._create_query_vector("database question", history)

        self.assertEqual(
            embeddings.last_input,
            [
                "User: message 5",
                "User: message 6",
                "User: message 7",
                "User: database question",
            ],
        )


if __name__ == "__main__":
    unittest.main()
