# Intelligent Search Engine

NLP term project. An LLM-backed search system that recognises query intent, retrieves from a local knowledge base, calls domain plugins, and accepts multimodal input.

## Features

- **Intent recognition** — classifies the domain and intent of a query, then routes it
- **Retrieval** — chunked vector store over a local knowledge base, with reranking
- **Domain plugins** — weather, finance, transport, web search, calculator, knowledge lookup
- **Multimodal input** — images (OCR), PDF, Word, and plain text
- **Evaluation harness** — measures search time, total latency, and success rate; emits JSON, text, and CSV reports

## Layout

```
main.py            Entry point (interactive / evaluate / batch / query modes)
server.py          HTTP server for the web frontend
config.py          Configuration, reads secrets from the environment
api.py             Thin HKGAI chat client
core/              LLM client, intent recogniser, RAG engine, vector store, workflow engine
retrieval/         Document processing and reranking
plugins/           Domain plugins, all subclassing BasePlugin
multimodal/        Image, PDF, DOCX, and text processors
evaluation/        Performance evaluator
utils/             Cache, logging, metrics
data/              Knowledge base (sample included)
```

## Setup

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in your own keys
```

Configuration is read from environment variables, so nothing secret lives in the source. The relevant variables are listed in `.env.example`; leaving one blank disables the plugin that needs it.

## Usage

```bash
python main.py --interactive              # interactive session
python main.py --evaluate                 # run the evaluation harness
python main.py --query "your question"    # one-shot query
python main.py --batch in.txt out.json    # batch mode
```

Multimodal queries reference a file inline, for example `--query "describe this 文件:hkust.png"`.

In interactive mode, `files` lists available data files and `answer` runs batch question answering over a document.

## Notes

- A Redis connection error at startup is expected; the system falls back to an in-memory cache.
- Runtime output (`logs/`, `evaluation_results/`, the vector database, and the embedding model cache) is excluded from version control. The sentence-transformers model downloads on first run.
- Sample test documents and evaluation runs from the original submission are not published.
