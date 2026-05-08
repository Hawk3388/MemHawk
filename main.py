# ToDo: 
# * add more customization (support more api's, more options)
# * Support for retrieval with history context


import time
from datetime import datetime
import uuid
import os
import json

import chromadb
from openai import OpenAI


class MemHawk:
    def __init__(self, api_url="http://localhost:11434/v1", api_key="test", embed_model="nomic-embed-text-v2-moe", history_folder="history", history_file="history.json", db_path="vector_db", collection_name="docs", max_live_user_turns=4, top_k_retrieval=3, retrieval_per_query_k=5, max_retrieval_distance=1.2, context=4096):
        self.api_url = api_url
        self.api_key = api_key
        self.embed_model = embed_model
        self.history_folder = history_folder
        self.history_file = history_file
        self.db_path = db_path
        self.collection_name = collection_name
        self.max_live_user_turns = max_live_user_turns
        self.top_k_retrieval = top_k_retrieval
        self.retrieval_per_query_k = retrieval_per_query_k
        self.max_retrieval_distance = max_retrieval_distance
        self.context = context
        self.api_client = OpenAI(base_url=self.api_url, api_key=self.api_key)
        self.client_db = chromadb.PersistentClient(path=os.path.join(self.history_folder, self.db_path))
        self.collection = self.client_db.get_or_create_collection(name=self.collection_name)

    def load_history(self, history_path=None):
        if history_path is None:
            history_path = os.path.join(self.history_folder, self.history_file)
        if not os.path.exists(history_path):
            return []
        with open(history_path, "r") as f:
            return json.load(f)

    def save_history(self, history, history_path=None):
        if history_path is None:
            history_path = os.path.join(self.history_folder, self.history_file)
        with open(history_path, "w") as f:
            json.dump(history, f)

    def count_user_turns(self, messages):
        return sum(1 for msg in messages if msg.get("role") == "user")
    
    def history_to_embedding_input(self, history):
        return [f"{msg['role'].capitalize()}: {msg['content']}" for msg in history]

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

    def archive_oldest_pair_if_needed(self, history, collection=None):
        if collection is None:
            collection = self.collection

        user_turns = self.count_user_turns(history)

        while user_turns > self.max_live_user_turns:
            pair, history = self.extract_oldest_turn_pair(history)
            if not pair:
                break

            doc_text = self.format_pair_for_embedding(pair)
            embed = self.api_client.embeddings.create(model=self.embed_model, input=doc_text)
            vector = embed["embeddings"][0]

            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[vector],
                documents=[doc_text],
                metadatas=[{"source": "chat_history", "timestamp": datetime.now().isoformat(timespec="seconds")}],
            )

            user_turns = self.count_user_turns(history)

        return history

    def create_average_embedding(self, embeddings):
        return [sum(col) / len(col) for col in zip(*embeddings)]

    def retrieve_context(self, prompt, history=None, collection=None, top_k=None):
        if collection is None:
            collection = self.collection

        if top_k is None:
            top_k = self.top_k_retrieval

        if collection.count() == 0:
            return []

        if history is None:
            embed_result = self.api_client.embeddings.create(model=self.embed_model, input=prompt)
            query_vector = embed_result["embeddings"][0]
        else:
            history.append({"role": "user", "content": prompt})
            embed_result = self.api_client.embeddings.create(model=self.embed_model, input=self.history_to_embedding_input(history))
            embeddings = embed_result["embeddings"]
            embeddings.append(embeddings[0])
            embeddings.append(embeddings[0])
            query_vector = self.create_average_embedding(embeddings)

        if not query_vector:
            return []

        best_distance_by_doc = {}

        result = collection.query(
            query_embeddings=[query_vector],
            n_results=self.retrieval_per_query_k,
            include=["documents", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for doc, distance in zip(docs, distances):
            if not doc:
                continue
            
            best_distance_by_doc[doc] = distance

        if not best_distance_by_doc:
            return []

        ranked = sorted(best_distance_by_doc.items(), key=lambda item: item[1])

        if self.max_retrieval_distance is not None:
            filtered = [doc for doc, dist in ranked if dist <= self.max_retrieval_distance]
            if filtered:
                return filtered[:top_k]

        return []

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

        messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def simple_run(self, prompt, history):
        history = self.archive_oldest_pair_if_needed(history)
        retrieved_docs = self.retrieve_context(prompt, history)
        messages = self.build_chat_messages(prompt, history, retrieved_docs)
        return messages

    def run(self, model="qwen3.5"):
        history = self.load_history(os.path.join(self.history_folder, self.history_file))

        try:
            while True:
                prompt = input(">>> ").strip()
                if not prompt:
                    continue

                if prompt.lower() in {"exit", "quit"}:
                    print("Bye.")
                    break

                start_time = time.time()

                history = self.archive_oldest_pair_if_needed(history, self.collection)
                retrieved_docs = self.retrieve_context(prompt, history, self.collection)
                messages = self.build_chat_messages(prompt, history, retrieved_docs)

                answer = self.api_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    think=False,
                    stream=True,
                    options={"num_ctx": self.context}
                )

                answer_text = ""
                for chunk in answer:
                    answer_text += chunk.message.content

                    print(chunk.message.content, end="", flush=True)

                history.append({"role": "user", "content": prompt})
                history.append({"role": "assistant", "content": answer_text})
                print(f"\nTime taken: {time.time() - start_time:.2f} seconds")
        except KeyboardInterrupt:
            print("\nStopped.")
        except Exception as exc:
            print(f"Error: {exc}")
        finally:
            self.save_history(os.path.join(self.history_folder, self.history_file), history)


if __name__ == "__main__":
    MemHawk().run()