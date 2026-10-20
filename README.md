# Web RAG Demo

A small command-line RAG demo using LlamaIndex. It searches the web for each new question, fetches up to three result pages, and answers from retrieved context. Choose vector retrieval or graph retrieval.

## Requirements

- Python 3.10 or newer
- An OpenAI API key with API access and billing enabled
- An internet connection for web search, page fetching, and uncached questions

## Setup

From the project directory, create a virtual environment and install dependencies:

```zsh
python3 -m venv .venv
.venv/bin/python -m pip install llama-index llama-index-llms-openai llama-index-embeddings-openai llama-index-readers-web ddgs
```

Set `OPENAI_API_KEY` in your shell environment. If you already store it in `~/.zshrc`, load it into the current terminal with:

```zsh
source ~/.zshrc
```

Do not commit or share your API key.

## Run

Start an interactive session using vector retrieval (the default):

```zsh
.venv/bin/python demo.py
```

Choose graph retrieval:

```zsh
.venv/bin/python demo.py --mode graph
```

Type questions at the prompt. Enter `quit` or `exit` to stop. You can also ask one question and exit:

```zsh
.venv/bin/python demo.py --mode graph --question "Who is the Prime Minister of India?"
```

Force a fresh web search and answer instead of using the answer cache:

```zsh
.venv/bin/python demo.py --mode graph --refresh
```

## How It Works

1. The program checks for an exact cached question and mode.
2. If there is no exact match, it embeds the question and compares it with saved questions from the same mode. A match at or above the `0.86` cosine-similarity threshold reuses the saved answer and sources.
3. If no cache entry matches, DuckDuckGo search finds up to three pages. `SimpleWebPageReader` fetches their content; search snippets are also kept as fallback context.
4. LlamaIndex retrieves context and uses the OpenAI model to answer. The answer, sources, and question embedding are saved for future lookups.

Exact cache hits do not call OpenAI. Semantic-cache checks make an embedding API request, but skip web search and answer generation when they find a match. `--refresh` bypasses both answer-cache checks.

## Storage

Generated local data is kept in `.rag_cache/` and excluded from Git:

- `answers.sqlite3` stores answers, source URLs, and question embeddings.
- `vector_store/` stores the persistent LlamaIndex vector index used in vector mode.
- Graph mode builds a property graph from the current question's fetched pages; that graph is not persisted between runs.

The cache is separated by mode, so a vector-mode answer is not reused in graph mode. Older saved answers without question embeddings can be backfilled during semantic lookup.

## Troubleshooting

- `OPENAI_API_KEY` errors: load the variable into this terminal, for example with `source ~/.zshrc`, and verify without printing the secret: `[[ -n "$OPENAI_API_KEY" ]] && echo "API key is set"`.
- Missing `llama_index` imports: run the demo with `.venv/bin/python` and install packages using `.venv/bin/python -m pip install ...` so they go into the same environment.
- A question gets a new web-search answer instead of a cached answer: verify that the mode matches. Semantic matching also depends on the embedding threshold; use `--refresh` when you need current information.
- A page cannot be fetched: some sites block automated readers. The search snippet may still be used, and the source URL is shown in the output.
