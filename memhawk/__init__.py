# ToDo: 
# * Better support for retrieval with history context


import time
from datetime import datetime
import uuid
import os

import chromadb
from openai import OpenAI


class MemHawk:
    def __init__(self, api_url="http://localhost:11434/v1", api_key="test", embed_model="nomic-embed-text-v2-moe", history_folder="history", history_file="history.json", db_path="vector_db", collection_name="docs", max_live_user_turns=6, top_k_retrieval=3, retrieval_per_query_k=5, max_retrieval_distance=1.2):
        self.api_url = api_url
        self.api_key = api_key
        self.embed_model = embed_model
        self.history_folder = os.path.abspath(history_folder)
        self.history_file = history_file
        self.db_path = db_path
        self.collection_name = collection_name
        self.max_live_user_turns = max_live_user_turns
        self.top_k_retrieval = top_k_retrieval
        self.retrieval_per_query_k = retrieval_per_query_k
        self.max_retrieval_distance = max_retrieval_distance

        os.makedirs(self.history_folder, exist_ok=True)

        self.history_path = os.path.join(self.history_folder, self.history_file)
        self.db_dir = os.path.join(self.history_folder, self.db_path)
        os.makedirs(self.db_dir, exist_ok=True)

        self.api_client = OpenAI(base_url=self.api_url, api_key=self.api_key)
        self.client_db = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client_db.get_or_create_collection(name=self.collection_name)

    # def load_history(self, history_path=None):
    #     if history_path is None:
    #         history_path = self.history_path

    #     if not os.path.exists(history_path):
    #         return []

    #     try:
    #         with open(history_path, "r", encoding="utf-8") as f:
    #             history = json.load(f)
    #         if isinstance(history, list):
    #             return history
    #     except (json.JSONDecodeError, OSError, TypeError, ValueError):
    #         pass

    #     return []

    # def save_history(self, history, history_path=None):
    #     if history_path is None:
    #         history_path = self.history_path

    #     os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)

    #     with open(history_path, "w", encoding="utf-8") as f:
    #         json.dump(history, f, ensure_ascii=False, indent=2)

    def save_history(self, history, collection=None):
        if collection is None:
            collection = self.collection

        user_turns = self.count_user_turns(history)

        while user_turns > 0:
            pair, history = self.extract_oldest_turn_pair(history)
            if not pair:
                break

            doc_text = self.format_pair_for_embedding(pair)
            embed = self.api_client.embeddings.create(model=self.embed_model, input=doc_text)
            vector = embed.data[0].embedding

            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[vector],
                documents=[doc_text],
                metadatas=[{"source": "chat_history", "timestamp": datetime.now().isoformat(timespec="seconds")}],
            )

            user_turns = self.count_user_turns(history)

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
            vector = embed.data[0].embedding

            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[vector],
                documents=[doc_text],
                metadatas=[{"source": "chat_history", "timestamp": datetime.now().isoformat(timespec="seconds")}],
            )

            user_turns = self.count_user_turns(history)

        return history

    def create_linear_weighted_embedding(self, embeddings):
        if not embeddings:
            return []
        weights = list(range(1, len(embeddings) + 1))
        weight_sum = sum(weights)
        if weight_sum == 0:
            return []
        weighted = []
        for col in zip(*embeddings):
            weighted_sum = sum(val * w for val, w in zip(col, weights))
            weighted.append(weighted_sum / weight_sum)
        return weighted

    def retrieve_context(self, prompt, history=None, collection=None, top_k=None):
        if collection is None:
            collection = self.collection

        if top_k is None:
            top_k = self.top_k_retrieval

        if collection.count() == 0:
            return []

        if history is None:
            embed_result = self.api_client.embeddings.create(model=self.embed_model, input=prompt)
            query_vector = embed_result.data[0].embedding
        else:
            query_history = list(history) + [{"role": "user", "content": prompt}]
            embed_result = self.api_client.embeddings.create(
                model=self.embed_model,
                input=self.history_to_embedding_input(query_history),
            )
            embeddings = [item.embedding for item in embed_result.data]
            query_vector = self.create_linear_weighted_embedding(embeddings)

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
                        "Treat the following information as part of your memory and use it naturally when it helps "
                        "answer the current question. Do not mention that you are using memory or past context. "
                        "If it is not relevant, ignore it and focus on the current conversation.\n\n"
                        f"{context_blob}"
                    ),
                }
            )

        messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def run(self, prompt, history):
        history = self.archive_oldest_pair_if_needed(list(history))
        retrieved_docs = self.retrieve_context(prompt, history)
        messages = self.build_chat_messages(prompt, history, retrieved_docs)
        return messages

    def demo(self, model="qwen3.5"):
        import ollama

        # history = self.load_history()
        history = []

        try:
            while True:
                prompt = input(">>> ").strip()
                if not prompt:
                    continue

                if prompt.lower() in {"exit", "quit"}:
                    print("Bye.")
                    break

                start_time = time.time()

                messages = self.run(prompt, history)

                answer = ollama.chat(
                    model=model,
                    messages=messages,
                    stream=True,
                    think=False,
                    options={"num_ctx": 4096},
                )

                answer_text = ""
                for chunk in answer:
                    response_chunk = chunk.message.content
                    answer_text += response_chunk
                    print(response_chunk, end="", flush=True)
                
                history.append({"role": "user", "content": prompt})
                history.append({"role": "assistant", "content": answer_text})
                print(f"\nTime taken: {time.time() - start_time:.2f} seconds")
        except KeyboardInterrupt:
            print("\nStopped.")
        except Exception as exc:
            print(f"Error: {exc}")
        finally:
            self.save_history(history)


if __name__ == "__main__":
    MemHawk().demo()
