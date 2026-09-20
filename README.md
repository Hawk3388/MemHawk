# MemHawk

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/memhawk.svg)](https://pypi.org/project/memhawk/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENCE)

**Persistent, retrieval-augmented memory for Python chat applications.**

MemHawk moves older chat turns into a local ChromaDB vector store and retrieves
only the memories that are relevant to the current request. This keeps the live
context compact while preserving useful information across longer conversations
and application restarts.

MemHawk is model-agnostic: it creates an OpenAI-compatible message list but does
not perform the final chat completion itself. Embeddings are generated through an
OpenAI-compatible API, which makes the library suitable for local servers such as
Ollama as well as other compatible providers.

## Features

- Persistent long-term memory backed by ChromaDB
- Automatic archival of complete user/assistant exchanges
- Prompt-first semantic retrieval with recency-weighted history
- Configurable retrieval depth, distance threshold, and live context size
- OpenAI-compatible embedding API
- OpenAI-style output messages for easy integration with existing chat clients
- Optional custom ChromaDB collections
- Local-first defaults with no hosted service required

## How it works

For every new request, MemHawk follows this pipeline:

1. Older completed exchanges are embedded and stored when the configured live
   history limit is exceeded.
2. The current prompt and live history are embedded separately.
3. A prompt-first retrieval vector is created. By default, the current prompt
   contributes 80% and the complete history shares the remaining 20%.
4. ChromaDB returns the nearest stored exchanges.
5. Results above the configured distance threshold are discarded.
6. Relevant memories are added as a system message before the live conversation.

Newer history messages receive more of the history budget than older messages.
The blended vector is also rescaled to preserve the expected embedding magnitude,
which avoids artificially worsening L2 distances through vector averaging.

## Requirements

- Python 3.10 or newer
- An OpenAI-compatible embeddings endpoint
- An embedding model available through that endpoint

The default configuration expects an API at `http://localhost:11434/v1` and an
embedding model named `nomic-embed-text-v2-moe`.

## Installation

Install the latest release from PyPI:

```bash
python -m pip install memhawk
```

Using a dedicated Conda environment:

```bash
conda create -n memhawk python=3.10
conda activate memhawk
python -m pip install memhawk
```

To work on MemHawk itself, clone the repository and install it in editable mode:

```bash
git clone https://github.com/Hawk3388/MemHawk.git
cd MemHawk
python -m pip install -e .
```

The release currently pins these core dependencies:

- `chromadb==1.5.9`
- `openai==3.1.0`

## Quick start

The following example uses one OpenAI-compatible endpoint for both embeddings and
chat completions. Replace `your-chat-model` with a model served by your endpoint.

```python
from openai import OpenAI

from memhawk import MemHawk


api_url = "http://localhost:11434/v1"
api_key = "test"

memory = MemHawk(
    api_url=api_url,
    api_key=api_key,
    embed_model="nomic-embed-text-v2-moe",
    db_path="MemHawk_db",
)
chat_client = OpenAI(base_url=api_url, api_key=api_key)

history = []

try:
    prompt = "What did we decide about the database architecture?"
    messages = memory.run(prompt, history)

    response = chat_client.chat.completions.create(
        model="your-chat-model",
        messages=messages,
    )
    answer = response.choices[0].message.content or ""

    history.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    )

    # Keep the caller-owned history synchronized with MemHawk's live-history limit.
    history = memory.archive_oldest_pair_if_needed(history)
finally:
    # Persist all remaining complete exchanges before shutdown.
    memory.save_history(history)
```

`run()` returns the final message list and does not call a language model. It also
does not mutate the supplied `history` list. Your application remains responsible
for generating the answer, appending the new exchange, and retaining the returned
trimmed history when calling `archive_oldest_pair_if_needed()` directly.

## Experimental version-aware engine

The original `MemHawk` implementation remains available unchanged. An experimental
version-aware implementation can be imported separately:

```python
from memhawk.v2 import MemHawkV2

memory = MemHawkV2(
    namespace="user-123",
    collection_name="memory_v2",
)
```

`MemHawkV2` uses a separate `memory_v2` collection by default, so it does not
modify the original engine's `memory` collection. It adds:

- exact-content deduplication before embedding,
- active and superseded memory versions,
- namespace isolation,
- validity windows with `valid_from` and `valid_until`,
- lightweight recency tie-breaking,
- a configurable cap on history messages embedded for each query,
- caller-visible trimmed history through `run_with_history()`,
- and retrieved-memory instructions that explicitly treat stored text as
  untrusted factual context.

Store versioned facts with a stable semantic key:

```python
memory.remember(
    "The primary database is MySQL 8.",
    memory_key="infrastructure.primary_database",
)

memory.remember(
    "The primary database is PostgreSQL 16 with pgvector.",
    memory_key="infrastructure.primary_database",
)
```

The second call marks the MySQL memory as `superseded`. Retrieval queries only
active memories, so obsolete versions do not consume candidate slots or enter the
model context.

Archived chat pairs can provide the key through message metadata:

```python
history = [
    {
        "role": "user",
        "content": "Which database should we use?",
        "metadata": {
            "memory_key": "infrastructure.primary_database",
        },
    },
    {
        "role": "assistant",
        "content": "Use PostgreSQL 16 with pgvector.",
    },
]
```

Applications that already know their domain can alternatively provide a cheap
local resolver callback:

```python
def resolve_memory_key(pair):
    text = " ".join(message["content"] for message in pair).lower()
    if "database" in text:
        return "infrastructure.primary_database"
    return None


memory = MemHawkV2(memory_key_resolver=resolve_memory_key)
```

The resolver is deliberately not implemented as another model call. A retrieval
still uses one embedding batch and one ChromaDB query; validity filtering is
linear in the small candidate pool and happens in memory. Writes perform metadata
lookups for deduplication and version replacement but avoid embedding exact
duplicates.

By default, only the six newest history messages are embedded for retrieval. Set
`max_history_embedding_messages` to another limit, or use `0` to retrieve from the
current prompt alone:

```python
memory = MemHawkV2(max_history_embedding_messages=6)
```

Use `run_with_history()` to keep the caller-owned history synchronized:

```python
messages, history = memory.run_with_history(prompt, history)
```

Without a stable `memory_key`, exact duplicates are still removed, but semantic
contradictions cannot be identified reliably. This is intentional: silently
guessing keys from arbitrary conversation text could invalidate unrelated facts.

## Message format

History uses the standard OpenAI chat format:

```python
history = [
    {"role": "user", "content": "My preferred database is PostgreSQL."},
    {"role": "assistant", "content": "Understood."},
]
```

Only complete exchanges containing a user message followed by an assistant
message can be archived. Other messages between them are stored as part of the
same exchange.

## Configuration

```python
memory = MemHawk(
    api_url="http://localhost:11434/v1",
    api_key="test",
    embed_model="nomic-embed-text-v2-moe",
    db_path="MemHawk_db",
    max_live_user_turns=6,
    top_k_retrieval=3,
    retrieval_per_query_k=5,
    max_retrieval_distance=1.2,
    current_prompt_weight=0.8,
    history_decay=0.7,
)
```

| Parameter | Default | Description |
| --- | --- | --- |
| `api_url` | `http://localhost:11434/v1` | Base URL of the OpenAI-compatible embeddings API. |
| `api_key` | `test` | API key passed to the client. Local servers may accept a placeholder. |
| `embed_model` | `nomic-embed-text-v2-moe` | Model used for stored documents and retrieval queries. |
| `db_path` | `MemHawk_db` | Directory containing the persistent ChromaDB database. |
| `max_live_user_turns` | `6` | Number of user turns retained in the live conversation before older exchanges are archived. |
| `top_k_retrieval` | `3` | Maximum number of stored memories injected into the final messages. |
| `retrieval_per_query_k` | `5` | Number of candidates requested from ChromaDB before filtering. |
| `max_retrieval_distance` | `1.2` | Maximum accepted ChromaDB distance. Lower values are stricter. |
| `current_prompt_weight` | `0.8` | Share of the retrieval vector assigned to the current prompt. Must be greater than `0.5` and at most `1.0`. |
| `history_decay` | `0.7` | Exponential recency factor for history. Must be greater than `0.0` and at most `1.0`. |

Distance values depend on the embedding model and ChromaDB distance space. Tune
`max_retrieval_distance` against representative conversations rather than treating
the default as universal.

## Retrieval weighting

Given a prompt embedding `p` and history embeddings `h`, MemHawk builds the query
direction from two budgets:

```text
query = prompt_weight × p + history_budget × recency_weighted_history
history_budget = 1 - prompt_weight
```

With the defaults, the prompt receives `0.8` of the total input weight. All history
messages together receive `0.2`; within that budget, each step backwards in the
conversation multiplies a message's influence by `0.7`.

Useful presets:

```python
# Strong prompt isolation; history only provides a small hint.
focused = MemHawk(current_prompt_weight=0.9, history_decay=0.5)

# More conversational continuity while the prompt remains dominant.
contextual = MemHawk(current_prompt_weight=0.65, history_decay=0.85)

# Equal weighting among history messages, still subordinate to the prompt.
uniform_history = MemHawk(current_prompt_weight=0.8, history_decay=1.0)
```

## Public API

### `run(prompt, history, collection=None)`

Archives excess live turns, retrieves relevant memories, and returns the complete
OpenAI-style message list for the next model call.

### `retrieve_context(prompt, history=None, collection=None, top_k=None)`

Returns the relevant stored conversation documents after distance filtering.

### `archive_oldest_pair_if_needed(history, collection=None, save=False)`

Archives complete exchanges until the live-history limit is met and returns the
remaining history. With `save=True`, all complete exchanges are archived.

### `save_history(history, collection=None)`

Archives every remaining complete exchange. Call this during a graceful shutdown
if the current session should be available as memory later.

### `build_chat_messages(prompt, history, retrieved_docs)`

Builds the final messages without performing retrieval. Retrieved memories are
placed in a system message before the live history and current prompt.

## Persistence and collections

The default database is created in `MemHawk_db/` with a collection named
`memory`. Each archived exchange receives:

- a generated UUID,
- its embedded conversation text,
- an ISO-formatted timestamp.

You can provide another ChromaDB collection to `run()`, `retrieve_context()`,
`archive_oldest_pair_if_needed()`, or `save_history()`:

```python
custom_collection = memory.client_db.get_or_create_collection("project_alpha")
messages = memory.run(prompt, history, collection=custom_collection)
```

Always use the same embedding model and vector dimensions for documents already
stored in a collection.

## Optional interactive demo

The built-in `demo()` method uses the separate `ollama` Python client for chat
generation. Install it and ensure both the configured embedding model and chosen
chat model are available on your Ollama server:

```bash
python -m pip install ollama
python -c "from memhawk import MemHawk; MemHawk().demo(model='qwen3.5')"
```

Enter `exit` or `quit` to stop. Remaining history is saved during shutdown.

## Retrieval benchmark

MemHawk includes a boundary-oriented benchmark with 26 stored memories and 29
retrieval cases. In addition to direct questions, it deliberately tests difficult
conditions:

- semantically similar memories and near-duplicate entity names,
- current facts competing with explicitly superseded information,
- vague follow-up questions that depend on conversation history,
- history that conflicts with an explicit topic change,
- English memories queried in German,
- questions requiring multiple memories,
- long, distracting conversation history,
- and questions whose answers were never stored.

From a source checkout, run the benchmark against the default embedding endpoint
at `http://localhost:11434/v1`:

```bash
python benchmarks/run_benchmark.py
```

With the repository's Conda environment:

```bash
conda run -n memhawk python benchmarks/run_benchmark.py
```

The report compares prompt-only retrieval with MemHawk's prompt-first history
strategy and measures:

- Hit Rate at K (`Hit@K`)
- Top-1 accuracy
- Mean Reciprocal Rank (`MRR`)
- Precision and Recall at K (`P@K`, `R@K`)
- rejection and false-positive rates for unknown information
- contamination rate for obsolete or explicitly conflicting memories
- strict overall accuracy, requiring complete recall and no forbidden memory
- median and 95th-percentile retrieval latency
- average number of returned documents
- context reduction relative to injecting the entire memory corpus

Results are also broken down by scenario category and by `easy`, `medium`, and
`hard` difficulty. Misses, partial multi-memory results, false positives, and
relevant memories ranked below position one are printed individually.

Specify another OpenAI-compatible endpoint or embedding model when needed:

```bash
python benchmarks/run_benchmark.py \
  --api-url http://localhost:11434/v1 \
  --embed-model nomic-embed-text-v2-moe
```

The benchmark uses the original engine by default. Test the version-aware engine
against the same cases with:

```bash
python benchmarks/run_benchmark.py --engine v2
```

The API key can be supplied with `--api-key` or the `MEMHAWK_API_KEY` environment
variable. `MEMHAWK_API_URL` and `MEMHAWK_EMBED_MODEL` are also supported.

Write a machine-readable report and enforce quality gates in CI:

```bash
python benchmarks/run_benchmark.py \
  --json-output benchmark-report.json \
  --min-hit-rate 0.95 \
  --min-mrr 0.70 \
  --min-rejection-rate 0.50 \
  --min-overall-accuracy 0.20 \
  --max-contamination-rate 0.95
```

The maximum retrieval distance defaults to the engine value of `1.2`. Because
distance distributions differ between embedding models, use `--max-distance` to
evaluate and tune the threshold for the selected model.

One warm-up retrieval is performed before latency measurement. Change it with
`--warmup-runs` or set it to `0` when cold-start performance is the target.

The dataset is stored in `benchmarks/dataset.json` and can be extended with
project-specific memories and queries. Every case declares one or more
`relevant_memory_ids` as ground truth. Negative cases use
`"expect_no_result": true` and an empty ground-truth list. Cases can also declare
`forbidden_memory_ids` to detect obsolete or confusable results that must not be
injected alongside the correct memory.

## Development and tests

Run the test suite with the project environment active:

```bash
python -m unittest discover -s tests -v
```

Or run it directly in the `memhawk` Conda environment:

```bash
conda run -n memhawk python -m unittest discover -s tests -v
```

The tests cover prompt dominance, history recency, embedding magnitude, no-history
behavior, the query vector passed into retrieval, dataset integrity, and the
benchmark metric calculations. Version-aware tests additionally cover
supersession, deduplication, namespace isolation, repeated archival, and validity
windows.

## Project structure

```text
MemHawk/
├── benchmarks/
│   ├── dataset.json             # Retrieval ground truth
│   └── run_benchmark.py         # Real-model benchmark CLI
├── memhawk/
│   ├── __init__.py              # Original implementation and optional demo
│   └── v2.py                    # Experimental version-aware implementation
├── tests/
│   └── test_retrieval_embedding.py
├── requirements.txt
├── setup.py
├── LICENCE
└── README.md
```

## Roadmap

- [ ] Add memory namespaces for users, sessions, and projects
- [ ] Support metadata filters during retrieval
- [ ] Add explicit APIs for updating, deleting, and forgetting memories
- [ ] Introduce duplicate and contradiction detection
- [ ] Evaluate multi-stage retrieval, MMR, and optional reranking
- [ ] Support importance-aware memories in addition to relevance and recency
- [ ] Provide asynchronous and batch-oriented APIs
- [x] Build a reproducible evaluation suite for retrieval quality, latency, and
  context reduction

## Operational notes

- MemHawk stores conversation content unencrypted in the configured local database.
  Do not archive secrets or regulated data without appropriate safeguards.
- Changing embedding models for an existing collection may cause dimension errors
  or inconsistent retrieval quality. Use a new collection or database when needed.
- Retrieval adds memory as a system message. Make sure this matches the trust and
  instruction hierarchy of the model serving your application.
- `save_history()` stores complete exchanges only; an unanswered final user message
  remains unarchived.

## License

MemHawk is licensed under the [Apache License 2.0](LICENCE).
