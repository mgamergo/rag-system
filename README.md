# Contextual RAG

Ask questions about local Markdown and text documents with a TypeScript retrieval-augmented generation pipeline. Experiment with contextual indexing, semantic and lexical retrieval, rank fusion, reranking, and full-corpus prompts. The companion Next.js app uses Convex and OpenRouter.

## Get started

Install Node.js 22.13 or later, then run from the repository root:

```powershell
npm ci
npm run ingest -- examples/notes.md --output-dir artifacts/demo
ollama pull llama3.2:3b
npm run ask -- "What are the two stages of RAG?" --artifact-dir artifacts/demo --provider ollama --model llama3.2:3b
```

Start the Ollama server before asking. Alternatively, set `OPENROUTER_API_KEY` in your process environment and use `--provider openrouter --model YOUR_PROVIDER/MODEL_ID`. Clients read environment variables directly and do not load `.env` files. Hosted requests send document text and metadata to providers and may incur charges.

The CLI prints answers and numbered sources with paths, chunk indexes, character offsets, and scores. Prompts ask for citations and an unknown answer when evidence is insufficient. Those instructions do not guarantee factual accuracy or valid citations.

## Repository boundaries

```text
src/
  domain/          Shared records, runtime schemas, and inference ports
  infrastructure/  HTTP transport, timeouts, and credential lookup
  storage/         Validated JSONL and atomic file replacement
  ingestion/       Discovery, decoding, chunking, and contextualization
  embeddings/      Local ONNX and hosted embedding adapters
  retrieval/       Semantic, lexical, fusion, full corpus, and reranking
  generation/      Grounded prompts and chat provider adapter
  evaluation/      Label validation and retrieval metrics
  pipelines/       Ingestion and answering orchestration
  cli/             Arguments, exit codes, and terminal output
  index.ts         Reusable library entry point
web/               Independent Next.js and Convex application
examples/          Sample notes and embedding smoke test
tests/             Behavioral tests and local HTTP integration
docs/              Interactive architecture and code-flow guides
artifacts/         Local indexes and reports, ignored by Git
```

Domain types have no I/O dependencies. Processing modules do not depend on orchestration or CLI code. Pipelines compose those modules. CLI modules translate arguments and present results. `Encoder`, `PairScorer`, `Contextualize`, and `HttpClient` separate processing from inference and external services. External JSON is validated with Zod before processing. The CLI and web app retain separate dependencies and storage.

List source files with `rg --files src tests examples web`. See the [architecture guide](docs/architecture-overview.html) and [code flow](docs/code-flow.html).

## Ingest documents

```powershell
npm run ingest -- "C:\path\to\documents" --output-dir artifacts/my-run --chunk-size 500 --overlap 50 --batch-size 32 --device cpu
```

Input is one UTF-8 `.txt` or `.md` file, or a directory scanned recursively. Discovery skips dependencies, caches, and Git metadata. Empty or unreadable documents are reported while valid documents continue. BOMs are removed and newlines normalized. Other file formats are unsupported.

Sizes and offsets count Unicode code points, not tokens. Offsets refer to decoded, normalized text; the end is exclusive. Overlap must be nonnegative and less than chunk size. `--chunk-boundary` accepts `character`, `sentence`, or `paragraph`. Natural boundaries prefer the latter half of a window, with a hard size fallback. Models can truncate inputs at their token limit.

Runs write three artifacts to `artifacts/run-<id>` or `--output-dir`:

- `chunks.jsonl`: `chunk_id`, `document_id`, `chunk_index`, absolute `source`, original `text`, `start_char`, `end_char`, and generated `context`.
- `embeddings.jsonl`: `chunk_id`, `model`, and `embedding`.
- `failures.jsonl`: skipped document `source` and `error`.

IDs are regenerated each run. Records without `context` load with an empty string. Embeddings join by `chunk_id`, independent of line order. Context generation and embedding finish before any replacement. Each artifact is replaced atomically; the three files are separate operations. Unrelated output files are retained. Empty input loads no model.

The default embedding model ID is `sentence-transformers/all-MiniLM-L6-v2`, resolved to its `Xenova/all-MiniLM-L6-v2` ONNX counterpart. It returns normalized 384-dimensional vectors. The default reranker resolves `cross-encoder/ms-marco-MiniLM-L-6-v2` to `Xenova/ms-marco-MiniLM-L-6-v2`. Transformers.js runs full-precision local inference. First use downloads weights from Hugging Face; later calls use the cache. Custom IDs and directories must contain compatible ONNX models and tokenizers. CPU is supported. `--device cuda` or `webgpu` requires compatible hardware and runtime support. Local inference has no HTTP timeout.

Hosted embeddings use `--model gemini:MODEL_ID` with `GEMINI_API_KEY`, or `--model voyage:MODEL_ID` with `VOYAGE_API_KEY`. The saved prefix selects the same model for queries. Requests distinguish query and document tasks, cap batches at 100 texts, and have a 120-second timeout. Provider input limits still apply.

```powershell
npm run embed-chunks -- artifacts/demo/chunks.jsonl --output artifacts/embeddings.jsonl
npm run smoke:embeddings
```

## Add document context

Set `ANTHROPIC_API_KEY`, then select an Anthropic model:

```powershell
npm run ingest -- examples/notes.md --output-dir artifacts/contextual --context-model YOUR_ANTHROPIC_MODEL_ID --chunk-boundary paragraph --chunk-size 1200 --overlap 100
```

Each request includes the whole document and one chunk. The instruction targets 50-100 tokens; the default output cap is 160. Context is stored separately from original text. Embedding and lexical search use context plus text. Generation receives separate context and text fields; original offsets remain exact.

The shared document prefix has an Anthropic prompt-cache breakpoint. `--no-context-cache` disables it. Usage totals print after ingestion. Cache eligibility and expiry depend on the provider. `--context-prompt path/to/instruction.txt` replaces the instruction and requires `--context-model`. `--context-max-tokens` and `--context-timeout` control output and timeout. Context failure aborts before artifact replacement. Documents are not truncated to fit the model window.

## Select retrieval and reranking

```powershell
npm run ask -- "Your question" --artifact-dir artifacts/contextual --model llama3.2:3b --retrieval-method hybrid --candidate-k 150 --top-k 20 --rerank-model
```

`--retrieval-method` accepts `semantic` by default, `bm25`, `tfidf`, or `hybrid`. Semantic search normalizes vectors and scores cosine similarity. It rejects missing or duplicate vectors, mixed models, invalid values, and inconsistent dimensions before query encoding. Re-ingest when changing models and keep the JSONL files from one run together.

BM25 uses `k1=1.5`, `b=0.75`, and positive Robertson IDF. TF-IDF uses sublinear term frequency, smoothed IDF, and cosine similarity. The tokenizer keeps identifiers such as `TS-999` intact and ignores case. Lexical methods need no vector artifacts and omit zero-score matches. No matches returns an unknown answer without calling generation.

Hybrid retrieval takes up to `candidate-k` results per branch and fuses ranks with constant 60. Its lexical branch defaults to BM25; `--lexical-method tfidf` changes it. `candidate-k` defaults to 150 and must be at least `top-k`. Answering defaults to `top-k=20`.

`--rerank-model` enables the default local cross encoder. An optional ID selects another model. Retrieval overfetches candidates, then reranking selects final sources. For hosted reranking, set `COHERE_API_KEY` or `VOYAGE_API_KEY` and pass `--rerank-provider cohere` or `voyage` plus `--rerank-model MODEL_ID`. Provider errors fail the query.

`--timeout` defaults to 120 seconds and applies separately to reranking and generation requests. `--max-tokens` defaults to 512; truncated answers are errors. `--ollama-base-url` defaults to `http://localhost:11434/v1` and must include the compatible API prefix. There are no automatic retries, conversation history, relevance thresholds, or token budgeting. Keep source sizes within model windows. Ranking scores are not correctness probabilities. Repeated queries reload indexes and models.

## Send a small corpus in full

```powershell
npm run ask -- "Your question" --artifact-dir artifacts/demo --model llama3.2:3b --full-context --full-context-max-chars 100000
```

This reconstructs documents, removes overlap, and bypasses retrieval and query embeddings. It rejects invalid offsets, gaps, conflicting overlaps, and inconsistent sources. The default budget is 600,000 characters, which does not guarantee the model's token window is sufficient. Full context ignores `top-k` for source selection, cannot use reranking, and does not add generation prompt caching.

## Evaluate retrieval

Create UTF-8 JSONL with one question per line and exactly one nonempty relevance field:

```json
{"question":"What are the stages of RAG?","relevant_chunk_ids":["COPY_CHUNK_ID_FROM_INDEX"]}
```

Alternatively, `relevant_sources` accepts exact source strings from the index. Unknown labels fail evaluation. IDs change on re-ingestion; source labels can compare chunk sizes.

```powershell
npm run evaluate -- questions.jsonl --artifact-dir artifacts/contextual --methods semantic bm25 tfidf hybrid --top-k 5 10 20 --output artifacts/retrieval-eval.json
```

Add `--rerank-model MODEL_ID` and optionally `--rerank-provider cohere` or `voyage` to evaluate reranking. Reports contain macro recall@k, failure rate, hit rate, MRR, per-query retrieved IDs, and latency. Repeated chunks from a source count once for document recall. Evaluation never calls generation. Latency includes model loading and is not a warmed-service benchmark.

## Call the library

```typescript
import { runRag } from './src/index.js';

const result = await runRag('What are the two stages of RAG?', 'artifacts/demo', {
  provider: 'ollama',
  model: 'llama3.2:3b',
});
process.stdout.write(result.answer + '\n');
```

`runIngestion(path, options)` returns the output directory, counts, and document failures. `runRag(question, artifactDir, options)` returns the question, answer, and sources. `npm run build` emits JavaScript and declarations to `dist/`.

## Run the web app

The independent `web/` app stores documents, 2,048-dimensional vectors, and successful question traces in Convex. It uses semantic vector retrieval and OpenRouter answers. The CLI provides the additional retrieval experiments.

From `web/`:

```powershell
npm ci
npx convex dev --configure new --once
npx convex env set OPENROUTER_API_KEY
npm run dev
```

Convex setup writes `NEXT_PUBLIC_CONVEX_URL` to `web/.env.local` and generates bindings. Embeddings default to `nvidia/nemotron-3-embed-1b:free`. Answers default to `google/gemma-4-26b-a4b-it`; override with the Convex variable `GENERATION_MODEL`. Reindex when changing embedding models or dimensions. Uploads accept UTF-8 `.md` and `.txt` files up to 500,000 characters.

## Verify changes

```powershell
npm run check
npm --prefix web run lint
npm --prefix web exec tsc -- --noEmit -p web/tsconfig.json
npm --prefix web run build
```

Tests use temporary files, controlled inference adapters, provider responses, and a real local HTTP server. They need no API keys, model downloads, or Ollama. `npm run smoke:embeddings` is a separate live local-model check.

All commands support `--help` without loading models. Exit status is `0` for success, `1` for processing errors or partial ingestion, and `2` for invalid CLI syntax.
