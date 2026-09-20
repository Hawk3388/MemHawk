import unittest

from benchmarks.run_benchmark import DEFAULT_DATASET, evaluate_profile, load_dataset


class BenchmarkTests(unittest.TestCase):
    def test_dataset_references_known_memories(self):
        dataset = load_dataset(DEFAULT_DATASET)

        self.assertEqual(dataset["name"], "memhawk-boundary-retrieval-v2")
        self.assertGreaterEqual(len(dataset["memories"]), 25)
        self.assertGreaterEqual(len(dataset["cases"]), 25)
        self.assertTrue(any(case.get("expect_no_result") for case in dataset["cases"]))
        self.assertTrue(
            any(
                len(case["relevant_memory_ids"]) > 1
                for case in dataset["cases"]
            )
        )

    def test_metric_aggregation_with_perfect_retrieval(self):
        dataset = load_dataset(DEFAULT_DATASET)
        document_by_id = {
            memory["id"]: memory["document"] for memory in dataset["memories"]
        }
        relevant_by_query = {
            case["query"]: case["relevant_memory_ids"] for case in dataset["cases"]
        }

        class PerfectEngine:
            def retrieve_context(self, prompt, history, collection, top_k):
                return [
                    document_by_id[memory_id]
                    for memory_id in relevant_by_query[prompt]
                ]

        result = evaluate_profile(
            "perfect",
            PerfectEngine(),
            collection=None,
            dataset=dataset,
            top_k=3,
        )

        self.assertEqual(result["hit_rate_at_k"], 1.0)
        self.assertEqual(result["top1_accuracy"], 1.0)
        self.assertEqual(result["mrr"], 1.0)
        self.assertEqual(result["recall_at_k"], 1.0)
        self.assertEqual(result["rejection_rate"], 1.0)
        self.assertEqual(result["contamination_rate"], 0.0)
        self.assertEqual(result["overall_accuracy"], 1.0)
        self.assertGreater(result["average_context_reduction"], 0.5)


if __name__ == "__main__":
    unittest.main()
