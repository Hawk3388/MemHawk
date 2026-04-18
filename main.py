import time
import uuid

import chromadb
import ollama


EMBED_MODEL = "nomic-embed-text-v2-moe"
CHAT_MODEL = "qwen3.5"

DB_PATH = "vector_db2"
COLLECTION_NAME = "docs_v2"

ARCHIVE_TRIGGER_USER_TURNS = 6
MAX_LIVE_USER_TURNS = 4
MAX_RECENT_TURNS_IN_PROMPT = 4
TOP_K_RETRIEVAL = 3
RETRIEVAL_CONTEXT_USER_TURNS = 2
RETRIEVAL_PER_QUERY_K = 4
# Lower distance usually means better match. Set to None to disable filtering.
MAX_RETRIEVAL_DISTANCE = 1.2


def count_user_turns(messages):
    return sum(1 for msg in messages if msg.get("role") == "user")


def extract_oldest_turn_pair(messages):
    user_idx = next(
        (i for i, m in enumerate(messages) if m.get("role") == "user"),
        None,
    )
    if user_idx is None:
        return None, messages

    assistant_idx = next(
        (
            i
            for i in range(user_idx + 1, len(messages))
            if messages[i].get("role") == "assistant"
        ),
        None,
    )
    if assistant_idx is None:
        return None, messages

    pair = messages[user_idx : assistant_idx + 1]
    remaining = messages[:user_idx] + messages[assistant_idx + 1 :]
    return pair, remaining


def format_pair_for_embedding(pair):
    lines = []
    for msg in pair:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def archive_oldest_pair_if_needed(history, collection):
    if count_user_turns(history) < ARCHIVE_TRIGGER_USER_TURNS:
        return history

    while count_user_turns(history) > MAX_LIVE_USER_TURNS:
        pair, history = extract_oldest_turn_pair(history)
        if not pair:
            break

        doc_text = format_pair_for_embedding(pair)
        embed = ollama.embed(model=EMBED_MODEL, input=doc_text)
        vector = embed["embeddings"][0]

        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[vector],
            documents=[doc_text],
            metadatas=[{"source": "chat_history"}],
        )

    return history


def format_messages_for_retrieval(messages):
    if not messages:
        return ""

    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_retrieval_queries(prompt, history):
    recent = recent_messages(history, max_user_turns=RETRIEVAL_CONTEXT_USER_TURNS)
    recent_context = format_messages_for_retrieval(recent)

    queries = [prompt]
    if recent_context:
        queries.append(
            "Aktuelle Frage:\n"
            f"{prompt}\n\n"
            "Letzter relevanter Dialogkontext:\n"
            f"{recent_context}"
        )
    return queries


def retrieve_context(prompt, history, collection, top_k=TOP_K_RETRIEVAL):
    if collection.count() == 0:
        return []

    queries = build_retrieval_queries(prompt, history)
    embed_result = ollama.embed(model=EMBED_MODEL, input=queries)
    query_vectors = embed_result.get("embeddings", [])

    if not query_vectors:
        return []

    best_distance_by_doc = {}
    for vector in query_vectors:
        result = collection.query(
            query_embeddings=[vector],
            n_results=RETRIEVAL_PER_QUERY_K,
            include=["documents", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for doc, distance in zip(docs, distances):
            if not doc:
                continue
            prev = best_distance_by_doc.get(doc)
            if prev is None or distance < prev:
                best_distance_by_doc[doc] = distance

    if not best_distance_by_doc:
        return []

    ranked = sorted(best_distance_by_doc.items(), key=lambda item: item[1])

    if MAX_RETRIEVAL_DISTANCE is not None:
        filtered = [doc for doc, dist in ranked if dist <= MAX_RETRIEVAL_DISTANCE]
        if filtered:
            return filtered[:top_k]

    return [doc for doc, _ in ranked[:top_k]]


def recent_messages(history, max_user_turns=MAX_RECENT_TURNS_IN_PROMPT):
    recent = []
    user_turns_seen = 0
    for msg in reversed(history):
        recent.append(msg)
        if msg.get("role") == "user":
            user_turns_seen += 1
            if user_turns_seen >= max_user_turns:
                break
    return list(reversed(recent))


def build_chat_messages(prompt, history, retrieved_docs):
    messages = []

    if retrieved_docs:
        context_blob = "\n\n---\n\n".join(retrieved_docs)
        messages.append(
            {
                "role": "system",
                "content": (
                    "Nutze den folgenden frueheren Kontext nur dann, wenn er relevant zur aktuellen "
                    "Frage ist. Wenn unklar, priorisiere die aktuelle Unterhaltung.\n\n"
                    f"{context_blob}"
                ),
            }
        )

    messages.extend(recent_messages(history))
    messages.append({"role": "user", "content": prompt})
    return messages


def main():
    history = []

    client_db = chromadb.PersistentClient(path=DB_PATH)
    collection = client_db.get_or_create_collection(name=COLLECTION_NAME)

    while True:
        prompt = input(">>> ").strip()
        if not prompt:
            continue

        if prompt.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        start_time = time.time()

        try:
            history = archive_oldest_pair_if_needed(history, collection)
            retrieved_docs = retrieve_context(prompt, history, collection)
            messages = build_chat_messages(prompt, history, retrieved_docs)

            answer = ollama.chat(model=CHAT_MODEL, messages=messages, think=False)
            answer_text = answer.message.content

            print(answer_text)

            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": answer_text})
            print(f"Time taken: {time.time() - start_time:.2f} seconds")
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()