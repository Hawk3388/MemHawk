import time
import uuid

import chromadb
import ollama


class MemHawk:
    def __init__(self):
        self.embed_model = "nomic-embed-text-v2-moe"
        self.chat_model = "qwen3.5"
        self.db_path = "vector_db2"
        self.collection_name = "docs_v2"
        self.archive_trigger_user_turns = 6
        self.max_live_user_turns = 4
        self.max_recent_turns_in_prompt = 4
        self.top_k_retrieval = 3
        self.retrieval_context_user_turns = 2
        self.retrieval_per_query_k = 4
        self.max_retrieval_distance = 1.2
        self.history = []

    def count_user_turns(self, messages):
        return sum(1 for msg in messages if msg.get("role") == "user")

    def extract_oldest_turn_pair(self, messages):
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

    def format_pair_for_embedding(self, pair):
        lines = []
        for msg in pair:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def archive_oldest_pair_if_needed(self, history, collection):
        if self.count_user_turns(history) < self.archive_trigger_user_turns:
            return history

        while self.count_user_turns(history) > self.max_live_user_turns:
            pair, history = self.extract_oldest_turn_pair(history)
            if not pair:
                break

            doc_text = self.format_pair_for_embedding(pair)
            embed = ollama.embed(model=self.embed_model, input=doc_text)
            vector = embed["embeddings"][0]

            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[vector],
                documents=[doc_text],
                metadatas=[{"source": "chat_history"}],
            )

        return history

    def format_messages_for_retrieval(self, messages):
        if not messages:
            return ""

        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def recent_messages(self, history, max_user_turns=None):
        if max_user_turns is None:
            max_user_turns = self.max_recent_turns_in_prompt

        recent = []
        user_turns_seen = 0
        for msg in reversed(history):
            recent.append(msg)
            if msg.get("role") == "user":
                user_turns_seen += 1
                if user_turns_seen >= max_user_turns:
                    break
        return list(reversed(recent))

    def build_retrieval_queries(self, prompt, history):
        recent = self.recent_messages(
            history,
            max_user_turns=self.retrieval_context_user_turns,
        )
        recent_context = self.format_messages_for_retrieval(recent)

        queries = [prompt]
        if recent_context:
            queries.append(
                "Aktuelle Frage:\n"
                f"{prompt}\n\n"
                "Letzter relevanter Dialogkontext:\n"
                f"{recent_context}"
            )
        return queries

    def retrieve_context(self, prompt, history, collection, top_k=None):
        if top_k is None:
            top_k = self.top_k_retrieval

        if collection.count() == 0:
            return []

        queries = self.build_retrieval_queries(prompt, history)
        embed_result = ollama.embed(model=self.embed_model, input=queries)
        query_vectors = embed_result.get("embeddings", [])

        if not query_vectors:
            return []

        best_distance_by_doc = {}
        for vector in query_vectors:
            result = collection.query(
                query_embeddings=[vector],
                n_results=self.retrieval_per_query_k,
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

        if self.max_retrieval_distance is not None:
            filtered = [doc for doc, dist in ranked if dist <= self.max_retrieval_distance]
            if filtered:
                return filtered[:top_k]

        return [doc for doc, _ in ranked[:top_k]]

    def build_chat_messages(self, prompt, history, retrieved_docs):
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

        messages.extend(self.recent_messages(history))
        messages.append({"role": "user", "content": prompt})
        return messages

    def run(self):
        client_db = chromadb.PersistentClient(path=self.db_path)
        collection = client_db.get_or_create_collection(name=self.collection_name)

        while True:
            prompt = input(">>> ").strip()
            if not prompt:
                continue

            if prompt.lower() in {"exit", "quit"}:
                print("Bye.")
                break

            start_time = time.time()

            try:
                self.history = self.archive_oldest_pair_if_needed(self.history, collection)
                retrieved_docs = self.retrieve_context(prompt, self.history, collection)
                messages = self.build_chat_messages(prompt, self.history, retrieved_docs)

                answer = ollama.chat(model=self.chat_model, messages=messages, think=False)
                answer_text = answer.message.content

                print(answer_text)

                self.history.append({"role": "user", "content": prompt})
                self.history.append({"role": "assistant", "content": answer_text})
                print(f"Time taken: {time.time() - start_time:.2f} seconds")
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as exc:
                print(f"Error: {exc}")


if __name__ == "__main__":
    MemHawk().run()