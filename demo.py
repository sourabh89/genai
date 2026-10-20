"""Compare vector RAG and graph RAG with LlamaIndex.

Install:
    python3 -m venv .venv
    .venv/bin/python -m pip install llama-index llama-index-llms-openai llama-index-embeddings-openai llama-index-readers-web ddgs

Set OPENAI_API_KEY, then run:
    .venv/bin/python demo.py --mode vector
    .venv/bin/python demo.py --mode graph
    .venv/bin/python demo.py --mode graph --refresh

Exact repeats use the answer cache; similar questions reuse answers above a
similarity threshold. This comparison makes an embedding API request.
Add --refresh to search the web and generate a fresh answer.
The local vector store and answer cache are saved under .rag_cache/.
Here, "graph" means LlamaIndex's PropertyGraphIndex (graph RAG), not a
separate package named LlamaGraph.
"""

import argparse
import json
import math
import os
import sqlite3
from pathlib import Path

from ddgs import DDGS
from llama_index.core import (
    Document,
    PropertyGraphIndex,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import SimpleWebPageReader

CACHE_DIR = Path(__file__).resolve().parent / ".rag_cache"
VECTOR_STORE_DIR = CACHE_DIR / "vector_store"
ANSWER_CACHE_PATH = CACHE_DIR / "answers.sqlite3"
SIMILARITY_THRESHOLD = 0.86


def _answer_cache_connection() -> sqlite3.Connection:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(ANSWER_CACHE_PATH)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS answers (
            cache_key TEXT PRIMARY KEY,
            answer TEXT NOT NULL,
            source_urls TEXT NOT NULL,
            question_embedding TEXT
        )"""
    )
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(answers)")
    }
    if "question_embedding" not in columns:
        connection.execute("ALTER TABLE answers ADD COLUMN question_embedding TEXT")
    return connection


def _cache_key(question: str, mode: str) -> str:
    normalized_question = " ".join(question.casefold().split())
    return f"{mode}:{normalized_question}"


def get_cached_answer(question: str, mode: str) -> tuple[str, list[str]] | None:
    connection = _answer_cache_connection()
    try:
        row = connection.execute(
            "SELECT answer, source_urls FROM answers WHERE cache_key = ?",
            (_cache_key(question, mode),),
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        return None
    return row[0], json.loads(row[1])


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        return 0.0

    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if not first_norm or not second_norm:
        return 0.0

    dot_product = sum(left * right for left, right in zip(first, second))
    return dot_product / (first_norm * second_norm)


def get_similar_cached_answer(
    mode: str,
    question_embedding: list[float],
    embed_model: OpenAIEmbedding,
) -> tuple[str, list[str], float] | None:
    connection = _answer_cache_connection()
    try:
        rows = connection.execute(
            """SELECT cache_key, answer, source_urls, question_embedding
            FROM answers WHERE cache_key LIKE ? ORDER BY rowid DESC""",
            (f"{mode}:%",),
        ).fetchall()

        rows_to_backfill = [row for row in rows if row[3] is None]
        if rows_to_backfill:
            questions = [row[0].partition(":")[2] for row in rows_to_backfill]
            try:
                embeddings = embed_model.get_text_embedding_batch(questions)
            except Exception:
                embeddings = []

            for row, embedding in zip(rows_to_backfill, embeddings):
                encoded_embedding = json.dumps(embedding)
                connection.execute(
                    "UPDATE answers SET question_embedding = ? WHERE cache_key = ?",
                    (encoded_embedding, row[0]),
                )
                row_index = next(
                    index for index, stored_row in enumerate(rows)
                    if stored_row[0] == row[0]
                )
                rows[row_index] = (*row[:3], encoded_embedding)
            connection.commit()
    finally:
        connection.close()

    best_match = None
    best_score = SIMILARITY_THRESHOLD
    for _, answer, source_urls, stored_embedding in rows:
        if stored_embedding is None:
            continue
        try:
            candidate_embedding = json.loads(stored_embedding)
        except json.JSONDecodeError:
            continue

        score = _cosine_similarity(question_embedding, candidate_embedding)
        if score >= best_score:
            best_score = score
            best_match = (answer, json.loads(source_urls), score)

    return best_match


def save_cached_answer(
    question: str,
    mode: str,
    answer: str,
    source_urls: list[str],
    question_embedding: list[float],
) -> None:
    connection = _answer_cache_connection()
    try:
        connection.execute(
            """INSERT OR REPLACE INTO answers
            (cache_key, answer, source_urls, question_embedding)
            VALUES (?, ?, ?, ?)""",
            (
                _cache_key(question, mode),
                answer,
                json.dumps(source_urls),
                json.dumps(question_embedding),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def load_vector_index() -> VectorStoreIndex | None:
    if not (VECTOR_STORE_DIR / "docstore.json").exists():
        return None

    storage_context = StorageContext.from_defaults(
        persist_dir=str(VECTOR_STORE_DIR)
    )
    return load_index_from_storage(storage_context)


def get_documents(question: str) -> tuple[list[Document], list[str]]:
    """Search the web, then fetch a few result pages for RAG context."""
    results = list(DDGS().text(question, max_results=3))
    reader = SimpleWebPageReader(html_to_text=True)
    documents = []
    source_urls = []

    for result in results:
        url = result.get("href", "")
        if not url.startswith(("http://", "https://")):
            continue

        title = result.get("title", "Web result")
        snippet = result.get("body", "")
        source_urls.append(url)
        documents.append(
            Document(
                text=f"Title: {title}\nSearch snippet: {snippet}\nURL: {url}",
                metadata={"source_url": url, "title": title},
            )
        )

        try:
            page_documents = reader.load_data(urls=[url])
        except Exception:
            continue

        for document in page_documents:
            document.metadata["source_url"] = url
            document.metadata.setdefault("title", title)
            documents.append(document)

    if not documents:
        raise RuntimeError("Web search returned no usable results.")

    return documents, source_urls


def ask(
    question: str,
    mode: str,
    refresh: bool = False,
) -> tuple[str, list[str], str]:
    if not refresh:
        cached_answer = get_cached_answer(question, mode)
        if cached_answer is not None:
            return cached_answer[0], cached_answer[1], "exact cache"

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set the OPENAI_API_KEY environment variable first.")

    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    question_embedding = Settings.embed_model.get_text_embedding(question)
    if not refresh:
        similar_answer = get_similar_cached_answer(
            mode,
            question_embedding,
            Settings.embed_model,
        )
        if similar_answer is not None:
            answer, source_urls, score = similar_answer
            return answer, source_urls, f"similar-question cache ({score:.3f})"

    documents, source_urls = get_documents(question)
    index = load_vector_index()

    if index is None:
        index = VectorStoreIndex.from_documents(documents)
    else:
        for document in documents:
            index.insert(document)
    index.storage_context.persist(persist_dir=str(VECTOR_STORE_DIR))

    if mode == "vector":
        query_engine = index.as_query_engine(similarity_top_k=2)
    else:
        index = PropertyGraphIndex.from_documents(documents)
        query_engine = index.as_query_engine(
            include_text=True,
            similarity_top_k=2,
        )

    response = query_engine.query(question)
    answer = str(response)
    used_urls = sorted(
        {
            node.node.metadata.get("source_url")
            for node in response.source_nodes
            if node.node.metadata.get("source_url")
        }
    )
    if not used_urls:
        used_urls = source_urls

    save_cached_answer(question, mode, answer, used_urls, question_embedding)
    return answer, used_urls, "web search"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("vector", "graph"),
        default="vector",
        help="Choose ordinary vector retrieval or property-graph retrieval.",
    )
    parser.add_argument(
        "--question",
        help="Ask one question and exit; omit this option for interactive mode.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Skip the saved answer and search the web again.",
    )
    args = parser.parse_args()
    print(f"Mode: {args.mode}")

    if args.question:
        questions = [args.question]
    else:
        questions = None
        print("Ask a question. Type 'quit' or 'exit' to stop.")

    while True:
        if questions is None:
            try:
                question = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.casefold() in {"quit", "exit"}:
                break
            if not question:
                continue
        else:
            question = questions.pop()

        try:
            answer, source_urls, cache_hit = ask(
                question,
                args.mode,
                refresh=args.refresh,
            )
        except Exception as error:
            print(f"Error: {error}")
        else:
            print(f"Answer source: {cache_hit}")
            print(f"Answer: {answer}")
            print("Sources:")
            for url in source_urls:
                print(f"- {url}")

        if questions is not None:
            break


if __name__ == "__main__":
    main()
