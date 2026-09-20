"""Benchmark MemHawk retrieval with a real embedding endpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chromadb
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memhawk import MemHawk
from memhawk.v2 import MemHawkV2


DEFAULT_DATASET = Path(__file__).with_name("dataset.json")


def load_dataset(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    memories = data.get("memories", [])
    cases = data.get("cases", [])
    memory_ids = [memory["id"] for memory in memories]
    documents = [memory["document"] for memory in memories]
    case_ids = [case["id"] for case in cases]

    if not memories or not cases:
        raise ValueError("The benchmark dataset needs memories and cases")
    if len(memory_ids) != len(set(memory_ids)):
        raise ValueError("Memory IDs must be unique")
    if len(documents) != len(set(documents)):
        raise ValueError("Memory documents must be unique")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Case IDs must be unique")

    known_ids = set(memory_ids)
    for case in cases:
        expected = set(case.get("relevant_memory_ids", []))
        expects_no_result = case.get("expect_no_result", False)
        if not expected and not expects_no_result:
            raise ValueError(
                f"Case {case.get('id')} needs relevant memories or expect_no_result"
            )
        if expected and expects_no_result:
            raise ValueError(
                f"Case {case.get('id')} cannot expect memories and no result"
            )
        if not case.get("category") or not case.get("difficulty"):
            raise ValueError(
                f"Case {case.get('id')} needs category and difficulty labels"
            )
        if case["difficulty"] not in {"easy", "medium", "hard"}:
            raise ValueError(
                f"Case {case.get('id')} has an invalid difficulty label"
            )
        unknown = expected - known_ids
        if unknown:
            raise ValueError(
                f"Case {case.get('id')} references unknown memories: {sorted(unknown)}"
            )
        forbidden = set(case.get("forbidden_memory_ids", []))
        unknown_forbidden = forbidden - known_ids
        if unknown_forbidden:
            raise ValueError(
                f"Case {case.get('id')} forbids unknown memories: "
                f"{sorted(unknown_forbidden)}"
            )
        overlap = expected & forbidden
        if overlap:
            raise ValueError(
                f"Case {case.get('id')} expects and forbids: {sorted(overlap)}"
            )
    return data


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentage * len(ordered)) - 1)
    return ordered[index]


def create_engine(
    embedding_client: Any,
    collection: Any,
    prompt_weight: float,
    history_decay: float,
    top_k: int,
    candidates: int,
    max_distance: float,
    embed_model: str,
    engine_version: str,
) -> Any:
    if engine_version == "v2":
        return MemHawkV2(
            api_client=SimpleNamespace(embeddings=embedding_client),
            collection=collection,
            namespace="benchmark",
            embed_model=embed_model,
            top_k_retrieval=top_k,
            retrieval_per_query_k=candidates,
            max_retrieval_distance=max_distance,
            current_prompt_weight=prompt_weight,
            history_decay=history_decay,
            recency_tiebreak_weight=0.0,
        )

    engine = MemHawk.__new__(MemHawk)
    engine.api_client = SimpleNamespace(embeddings=embedding_client)
    engine.collection = collection
    engine.embed_model = embed_model
    engine.current_prompt_weight = prompt_weight
    engine.history_decay = history_decay
    engine.top_k_retrieval = top_k
    engine.retrieval_per_query_k = candidates
    engine.max_retrieval_distance = max_distance
    return engine


def populate_collection(
    collection: Any,
    embedding_client: Any,
    embed_model: str,
    memories: list[dict[str, Any]],
    engine_version: str,
) -> None:
    documents = [memory["document"] for memory in memories]
    response = embedding_client.create(model=embed_model, input=documents)
    metadatas = []
    for memory in memories:
        source_metadata = memory.get("metadata", {})
        if engine_version == "v2":
            status = source_metadata.get("status", "current")
            metadatas.append(
                {
                    "benchmark_id": memory["id"],
                    "namespace": "benchmark",
                    "status": (
                        "superseded" if status == "superseded" else "active"
                    ),
                    "version": int(source_metadata.get("version", 1)),
                    "memory_key": source_metadata.get("memory_key", memory["id"]),
                    "content_hash": MemHawkV2._content_hash(memory["document"]),
                    "created_at": source_metadata.get(
                        "timestamp", "2026-01-01T00:00:00+00:00"
                    ),
                }
            )
        else:
            metadatas.append({"benchmark_id": memory["id"], **source_metadata})

    collection.add(
        ids=[memory["id"] for memory in memories],
        documents=documents,
        embeddings=[item.embedding for item in response.data],
        metadatas=metadatas,
    )


def mean_or_zero(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize_cases(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [result for result in case_results if not result["is_negative"]]
    negatives = [result for result in case_results if result["is_negative"]]
    contamination_cases = [
        result for result in case_results if result["forbidden"]
    ]
    rejection_rate = (
        mean_or_zero([float(result["rejected"]) for result in negatives])
        if negatives
        else None
    )
    return {
        "case_count": len(case_results),
        "positive_case_count": len(positives),
        "negative_case_count": len(negatives),
        "overall_accuracy": mean_or_zero(
            [float(result["correct"]) for result in case_results]
        ),
        "hit_rate_at_k": mean_or_zero(
            [float(result["hit"]) for result in positives]
        ),
        "top1_accuracy": mean_or_zero(
            [float(result["top1"]) for result in positives]
        ),
        "mrr": mean_or_zero(
            [result["reciprocal_rank"] for result in positives]
        ),
        "precision_at_k": mean_or_zero(
            [result["precision_at_k"] for result in positives]
        ),
        "recall_at_k": mean_or_zero(
            [result["recall_at_k"] for result in positives]
        ),
        "rejection_rate": rejection_rate,
        "false_positive_rate": 1.0 - rejection_rate if negatives else None,
        "contamination_rate": (
            mean_or_zero(
                [float(result["contaminated"]) for result in contamination_cases]
            )
            if contamination_cases
            else None
        ),
    }


def build_breakdown(
    case_results: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    values = sorted({result[label] for result in case_results})
    return {
        value: summarize_cases(
            [result for result in case_results if result[label] == value]
        )
        for value in values
    }


def evaluate_profile(
    name: str,
    engine: Any,
    collection: Any,
    dataset: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    document_to_id = {
        memory["document"]: memory["id"] for memory in dataset["memories"]
    }
    corpus_characters = sum(
        len(memory["document"]) for memory in dataset["memories"]
    )
    case_results = []
    latencies_ms = []

    for case in dataset["cases"]:
        started = time.perf_counter()
        documents = engine.retrieve_context(
            case["query"],
            history=case.get("history", []),
            collection=collection,
            top_k=top_k,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000)

        retrieved_ids = [document_to_id[document] for document in documents]
        relevant = set(case.get("relevant_memory_ids", []))
        forbidden = set(case.get("forbidden_memory_ids", []))
        is_negative = case.get("expect_no_result", False)
        relevant_ranks = [
            rank
            for rank, memory_id in enumerate(retrieved_ids, start=1)
            if memory_id in relevant
        ]
        relevant_found = len(set(retrieved_ids) & relevant)
        retrieved_characters = sum(len(document) for document in documents)
        hit = bool(relevant_ranks) if not is_negative else False
        complete_recall = (
            relevant_found == len(relevant) if not is_negative else False
        )
        forbidden_retrieved = [
            memory_id for memory_id in retrieved_ids if memory_id in forbidden
        ]
        contaminated = bool(forbidden_retrieved)
        rejected = not retrieved_ids if is_negative else False
        case_results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "difficulty": case["difficulty"],
                "expected": sorted(relevant),
                "forbidden": sorted(forbidden),
                "forbidden_retrieved": forbidden_retrieved,
                "retrieved": retrieved_ids,
                "is_negative": is_negative,
                "correct": (
                    rejected
                    if is_negative
                    else complete_recall and not contaminated
                ),
                "hit": hit,
                "complete_recall": complete_recall,
                "contaminated": contaminated,
                "top1": bool(relevant_ranks and relevant_ranks[0] == 1),
                "rejected": rejected,
                "reciprocal_rank": (
                    1.0 / relevant_ranks[0] if relevant_ranks else 0.0
                ),
                "precision_at_k": (
                    relevant_found / top_k if not is_negative else None
                ),
                "recall_at_k": (
                    relevant_found / len(relevant) if not is_negative else None
                ),
                "context_reduction": 1.0
                - (retrieved_characters / corpus_characters),
            }
        )

    summary = summarize_cases(case_results)
    return {
        "name": name,
        **summary,
        "average_returned": statistics.fmean(
            len(result["retrieved"]) for result in case_results
        ),
        "average_context_reduction": statistics.fmean(
            result["context_reduction"] for result in case_results
        ),
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms),
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
        },
        "category_breakdown": build_breakdown(case_results, "category"),
        "difficulty_breakdown": build_breakdown(case_results, "difficulty"),
        "cases": case_results,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset(args.dataset)
    api_client = OpenAI(base_url=args.api_url, api_key=args.api_key)
    embedding_client = api_client.embeddings

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name=f"memhawk-benchmark-{uuid.uuid4().hex}"
    )
    populate_collection(
        collection,
        embedding_client,
        args.embed_model,
        dataset["memories"],
        args.engine,
    )

    candidates = min(args.candidates, len(dataset["memories"]))
    profile_weights = (
        [("prompt_only", 1.0), ("prompt_first_history", args.prompt_weight)]
        if args.profile == "both"
        else [
            (
                args.profile,
                1.0 if args.profile == "prompt_only" else args.prompt_weight,
            )
        ]
    )
    warmup_engine = create_engine(
        embedding_client=embedding_client,
        collection=collection,
        prompt_weight=args.prompt_weight,
        history_decay=args.history_decay,
        top_k=args.top_k,
        candidates=candidates,
        max_distance=args.max_distance,
        embed_model=args.embed_model,
        engine_version=args.engine,
    )
    warmup_case = dataset["cases"][0]
    for _ in range(args.warmup_runs):
        warmup_engine.retrieve_context(
            warmup_case["query"],
            history=warmup_case.get("history", []),
            collection=collection,
            top_k=args.top_k,
        )

    profiles = []
    for name, prompt_weight in profile_weights:
        engine = create_engine(
            embedding_client=embedding_client,
            collection=collection,
            prompt_weight=prompt_weight,
            history_decay=args.history_decay,
            top_k=args.top_k,
            candidates=candidates,
            max_distance=args.max_distance,
            embed_model=args.embed_model,
            engine_version=args.engine,
        )
        profiles.append(
            evaluate_profile(name, engine, collection, dataset, args.top_k)
        )

    return {
        "benchmark": dataset.get("name", args.dataset.stem),
        "api_url": args.api_url,
        "configuration": {
            "engine": args.engine,
            "embed_model": args.embed_model,
            "top_k": args.top_k,
            "candidates": candidates,
            "max_distance": args.max_distance,
            "prompt_weight": args.prompt_weight,
            "history_decay": args.history_decay,
            "warmup_runs": args.warmup_runs,
        },
        "profiles": profiles,
    }


def print_report(report: dict[str, Any]) -> None:
    config = report["configuration"]
    print(f"Benchmark: {report['benchmark']}")
    print(
        f"Engine: {config['engine']} | Embedding endpoint: {report['api_url']} | "
        f"top_k={config['top_k']} | max_distance={config['max_distance']}"
    )
    print()
    print(
        f"{'Profile':<24} {'Pos':>4} {'Neg':>4} {'Hit@K':>7} {'Top1':>7} "
        f"{'MRR':>7} {'P@K':>7} {'R@K':>7} {'Reject':>7} {'Contam':>7} "
        f"{'Overall':>8}"
    )
    print("-" * 102)
    for profile in report["profiles"]:
        rejection = profile["rejection_rate"]
        rejection_text = f"{rejection:.1%}" if rejection is not None else "-"
        contamination = profile["contamination_rate"]
        contamination_text = (
            f"{contamination:.1%}" if contamination is not None else "-"
        )
        print(
            f"{profile['name']:<24} "
            f"{profile['positive_case_count']:>4} "
            f"{profile['negative_case_count']:>4} "
            f"{profile['hit_rate_at_k']:>7.1%} "
            f"{profile['top1_accuracy']:>7.1%} "
            f"{profile['mrr']:>7.3f} "
            f"{profile['precision_at_k']:>7.3f} "
            f"{profile['recall_at_k']:>7.3f} "
            f"{rejection_text:>7} "
            f"{contamination_text:>7} "
            f"{profile['overall_accuracy']:>8.1%}"
        )

    print()
    print(
        f"{'Profile':<24} {'Docs':>7} {'Mean ms':>9} {'P50 ms':>9} "
        f"{'P95 ms':>9} {'Reduction':>10}"
    )
    print("-" * 72)
    for profile in report["profiles"]:
        latency = profile["latency_ms"]
        print(
            f"{profile['name']:<24} "
            f"{profile['average_returned']:>7.2f} "
            f"{latency['mean']:>9.2f} "
            f"{latency['p50']:>9.2f} "
            f"{latency['p95']:>9.2f} "
            f"{profile['average_context_reduction']:>9.1%}"
        )

    for profile in report["profiles"]:
        print(f"\nBoundary breakdown for {profile['name']}:")
        print(
            f"{'Category':<24} {'Cases':>5} {'Accuracy':>9} "
            f"{'Hit@K':>7} {'MRR':>7} {'Reject':>7} {'Contam':>7}"
        )
        print("-" * 74)
        for category, metrics in profile["category_breakdown"].items():
            rejection = metrics["rejection_rate"]
            rejection_text = f"{rejection:.1%}" if rejection is not None else "-"
            contamination = metrics["contamination_rate"]
            contamination_text = (
                f"{contamination:.1%}" if contamination is not None else "-"
            )
            print(
                f"{category:<24} "
                f"{metrics['case_count']:>5} "
                f"{metrics['overall_accuracy']:>9.1%} "
                f"{metrics['hit_rate_at_k']:>7.1%} "
                f"{metrics['mrr']:>7.3f} "
                f"{rejection_text:>7} "
                f"{contamination_text:>7}"
            )

        print(f"\nDifficulty breakdown for {profile['name']}:")
        print(f"{'Difficulty':<24} {'Cases':>5} {'Accuracy':>9} {'Top1':>7}")
        print("-" * 50)
        for difficulty, metrics in profile["difficulty_breakdown"].items():
            print(
                f"{difficulty:<24} "
                f"{metrics['case_count']:>5} "
                f"{metrics['overall_accuracy']:>9.1%} "
                f"{metrics['top1_accuracy']:>7.1%}"
            )

        issues = [
            case
            for case in profile["cases"]
            if not case["correct"] or (not case["is_negative"] and not case["top1"])
        ]
        if issues:
            print(f"\nBoundary cases for {profile['name']}:")
            for case in issues:
                if case["is_negative"]:
                    issue = "false positive"
                elif not case["hit"]:
                    issue = "miss"
                elif not case["complete_recall"]:
                    issue = "partial recall"
                elif case["contaminated"]:
                    issue = "conflicting memory retrieved"
                else:
                    issue = "relevant result below rank 1"
                print(
                    f"  - {case['id']} ({issue}): expected={case['expected']}, "
                    f"retrieved={case['retrieved']}, "
                    f"forbidden={case['forbidden_retrieved']}"
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure MemHawk retrieval quality, latency, and context reduction."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--engine", choices=("v1", "v2"), default="v1")
    parser.add_argument(
        "--profile",
        choices=("both", "prompt_only", "prompt_first_history"),
        default="both",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("MEMHAWK_API_URL", "http://localhost:11434/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("MEMHAWK_API_KEY", "test"))
    parser.add_argument(
        "--embed-model",
        default=os.getenv("MEMHAWK_EMBED_MODEL", "nomic-embed-text-v2-moe"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--max-distance", type=float, default=1.2)
    parser.add_argument("--prompt-weight", type=float, default=0.8)
    parser.add_argument("--history-decay", type=float, default=0.7)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    parser.add_argument("--min-mrr", type=float, default=0.0)
    parser.add_argument("--min-rejection-rate", type=float, default=0.0)
    parser.add_argument("--min-overall-accuracy", type=float, default=0.0)
    parser.add_argument("--max-contamination-rate", type=float, default=1.0)
    parser.add_argument("--json-output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k < 1 or args.candidates < args.top_k:
        raise SystemExit("candidates must be greater than or equal to top-k >= 1")
    if not args.max_distance >= 0.0:
        raise SystemExit("max-distance must be zero or greater")
    if args.warmup_runs < 0:
        raise SystemExit("warmup-runs must be zero or greater")
    if not 0.5 < args.prompt_weight <= 1.0:
        raise SystemExit("prompt-weight must be greater than 0.5 and at most 1.0")
    if not 0.0 < args.history_decay <= 1.0:
        raise SystemExit("history-decay must be greater than 0.0 and at most 1.0")
    quality_thresholds = {
        "min-hit-rate": args.min_hit_rate,
        "min-mrr": args.min_mrr,
        "min-rejection-rate": args.min_rejection_rate,
        "min-overall-accuracy": args.min_overall_accuracy,
        "max-contamination-rate": args.max_contamination_rate,
    }
    for label, value in quality_thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{label} must be between 0.0 and 1.0")

    report = run_benchmark(args)
    print_report(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON report: {args.json_output}")

    measured_profile = next(
        (
            profile
            for profile in report["profiles"]
            if profile["name"] == "prompt_first_history"
        ),
        report["profiles"][0],
    )
    if measured_profile["hit_rate_at_k"] < args.min_hit_rate:
        print(
            f"\nFAILED: hit rate {measured_profile['hit_rate_at_k']:.3f} "
            f"is below {args.min_hit_rate:.3f}",
            file=sys.stderr,
        )
        return 1
    if measured_profile["mrr"] < args.min_mrr:
        print(
            f"\nFAILED: MRR {measured_profile['mrr']:.3f} "
            f"is below {args.min_mrr:.3f}",
            file=sys.stderr,
        )
        return 1
    if (
        measured_profile["rejection_rate"] is not None
        and measured_profile["rejection_rate"] < args.min_rejection_rate
    ):
        print(
            f"\nFAILED: rejection rate {measured_profile['rejection_rate']:.3f} "
            f"is below {args.min_rejection_rate:.3f}",
            file=sys.stderr,
        )
        return 1
    if measured_profile["overall_accuracy"] < args.min_overall_accuracy:
        print(
            f"\nFAILED: overall accuracy {measured_profile['overall_accuracy']:.3f} "
            f"is below {args.min_overall_accuracy:.3f}",
            file=sys.stderr,
        )
        return 1
    if (
        measured_profile["contamination_rate"] is not None
        and measured_profile["contamination_rate"]
        > args.max_contamination_rate
    ):
        print(
            f"\nFAILED: contamination rate "
            f"{measured_profile['contamination_rate']:.3f} is above "
            f"{args.max_contamination_rate:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
