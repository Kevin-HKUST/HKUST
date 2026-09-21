# Intelligent Search Engine

NLP term project (MAIE 5221). An LLM-backed question answering system that classifies query intent, routes to a domain plugin or a local vector store, and asks the LLM to synthesise an answer from whatever was retrieved. Accepts images, PDFs, and Word documents as additional input.

## Pipeline

```
query ──▶ parse multimodal prefix ──▶ OCR (if an image was attached)
      ──▶ IntentRecognizer ──▶ retrieve()  ──▶ plugin | vector store | web fallback
      ──▶ synthesize() ──▶ LLM ──▶ answer
```

`WorkflowEngine.retrieve()` and `.synthesize()` are separate entry points so callers can time retrieval and generation independently; `.run()` chains them.

## What is implemented

| Area | Detail |
|---|---|
| Intent routing | Keyword scoring across English, Simplified, and Traditional Chinese. Labels live in `core/intents.py` and are shared by the recognizer and the engine. |
| Plugins | `weather`, `finance`, `transport`, `web_search`, `math`. Registry is asserted against `PLUGIN_INTENTS` at construction. |
| Retrieval | FAISS via LangChain (`core/vector_store.py`), persisted under `data/faiss_index`. Empty index falls back to web search. |
| Multimodal | Google Vision OCR for images; PyMuPDF for PDF; python-docx for Word. |
| Evaluation | `--evaluate` runs every `.docx` in `data/test_questions/` and writes JSON, text, and CSV reports to `evaluation_results/`. |
| Batch QA | `--batch in.docx out.json` answers every question in a document and reports success rate and timings. |

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in your own keys
```

Configuration is read from environment variables, so no secret lives in the source. `python-dotenv` is optional — exporting the variables directly works the same way. A blank key disables the plugin that needs it; `HKGAI_API_KEY` is the only one required to start.

## Usage

```bash
python main.py --interactive              # interactive session
python main.py --query "your question"    # one-shot query
python main.py --batch in.docx out.json   # batch question answering
python main.py --evaluate                 # timing and success-rate report
python main.py --rebuild_rag              # rebuild the standalone RAG index
```

Multimodal queries name a file inline, either `[Input: hkust.png] "what is this"` or `hkust.png | what is this`.

In interactive mode, entering a bare file path parses that file and answers from its contents.

## Tests

```bash
python -m unittest discover -s tests -t .
```

The suite runs offline with no dependencies installed and no API key: it covers intent routing, the two regressions described below, and the contract that every intent the recognizer emits is one the engine dispatches. The single network test skips itself unless `HKGAI_API_KEY` is set.

## Known limitations

- **Two retrieval implementations coexist.** `core/vector_store.py` (LangChain + FAISS) serves the query path; `core/rag_engine.py` (raw FAISS + SentenceTransformer) backs only `--rebuild_rag` and writes a separate index. They are not interchangeable and should be consolidated.
- **`VectorStore` seeds from `data/knowledge.json`**, which is not in the repository. Without it, and without a prebuilt `data/faiss_index`, local retrieval returns nothing and every knowledge query falls through to web search.
- Answer quality scoring in `main.py` is a length-and-keyword heuristic, not a real relevance metric.

## Fixed after submission

This project was reviewed after the fact and the following defects were corrected. They are listed because the original submission shipped with them.

- `requirements.txt` did not describe the project: FAISS, pandas, loguru, pypdf, and matplotlib were used but unlisted, while `openai` and `chromadb` were pinned but never imported. Installing from it could not produce a working environment.
- `tests/test_basic.py` failed at import — it pulled `HKGAI_API_KEY` from the wrong module and `VectorStore` from the wrong package, then called a method that did not exist. The suite had never run.
- The recognizer emitted `general_knowledge` while the engine only dispatched `knowledge`, so that branch was dead. Intent labels now have a single definition in `core/intents.py`, and a test enforces the contract.
- `math` was dispatched by the engine but never emitted by the recognizer, leaving `CalculatorPlugin` unreachable. The recognizer now scores arithmetic, and the plugin registry is asserted against the intent list.
- `recognize()` discarded `confidence`, `domains`, and `requires_web_search`, throwing away most of the scoring work.
- Bare `今天` / `明天` were weather keywords at the highest weight, so `今天股价怎么样` classified as weather. Only time-qualified weather phrases remain.
- The transport branch was force-triggered by hardcoded `kfc` / `肯德基` / `麦当劳` strings tuned to the demo set. Removed in favour of the scored keywords.
- The system prompt hardcoded `当前日期: 2025-12-14`. Now uses the current date.
- `config.py` disabled TLS verification process-wide (`ssl._create_default_https_context = _create_unverified_context`) and created five directories as an import side effect. Both removed; directory creation is now an explicit `ensure_runtime_dirs()` call from the entry point.
- `--evaluate` was declared in the argument parser but never handled, and `SearchEvaluator` called two `WorkflowEngine` methods that did not exist. The flag now works against the `retrieve()` / `synthesize()` seam.
- `Config.BAIDU_MAP_API_KEY` was read by the API checker but never defined.
- Deleted as superseded and unreferenced: `api.py` (duplicate LLM client), `core/response_generator.py`, `retrieval/document_processor.py`, `retrieval/reranker.py`, `utils/cache.py`, `utils/metrics.py`, `utils/logger.py`, `plugins/knowledge_plugin.py` (a no-op stub the engine could not reach). Earlier READMEs advertised reranking, caching, and metrics as features; none of them were wired in.
- Ad-hoc scripts named `test_*.py` at the project root would have been collected as tests and would have hit the network. Moved to `scripts/` with descriptive names.
