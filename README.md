# Field Notes RAG

A Next.js + Convex workbench for contextual retrieval experiments. The browser is the only interface. Convex Node actions call one shared engine in `src/`; there is no ingestion, answering or evaluation CLI.

## Run locally

Use Node.js 22.13 or later. Install the two dependency sets:

```powershell
npm ci
npm --prefix web ci
cd web
npx convex dev
```

On first use, select or create a Convex deployment. Keep `convex dev` running to synchronize backend edits. In a second terminal, set only the server keys needed for the selected providers, then start Next.js:

```powershell
cd web
npx convex env set OPENROUTER_API_KEY
# Optional providers:
npx convex env set ANTHROPIC_API_KEY
npx convex env set VOYAGE_API_KEY
npx convex env set GEMINI_API_KEY
npx convex env set COHERE_API_KEY
npm run dev
```

Convex setup writes `NEXT_PUBLIC_CONVEX_URL` into `web/.env.local`. Provider keys belong on Convex, never in public browser variables. Uploads, questions and evaluation runs may incur provider charges. Tests and builds do not require inference keys or call paid APIs.

## Workspace

Upload UTF-8 `.md` or `.txt` files. Choose chunk size, overlap and character/sentence/paragraph boundaries. Optional contextual retrieval sends each whole document and chunk to an explicit Anthropic model. Customize the instruction, cache breakpoint, token cap and timeout. Original text, code-point offsets and generated context are stored separately.

Embeddings use OpenRouter, Voyage or Gemini. Local ONNX models, CPU/GPU controls and localhost Ollama are replaced with hosted providers. Defaults preserve the old web embedding model; generation and retrieval controls preserve the old CLI numerical defaults. See [every option and its replacement](docs/option-mapping.md).

Ask with semantic, BM25, TF-IDF or hybrid retrieval. Hybrid uses equal-weight RRF, constant 60 by default. Optional hosted Cohere/Voyage reranking overfetches candidateK then selects topK. Full-context mode reconstructs selected documents, removes overlap and bypasses query embeddings; it cannot use reranking. Select documents and the compatible embedding space. Sources show filename, zero-based chunk index, offsets, score, chunk ID, original text and separate generated context.

Use a document's re-embed button to preserve its chunks/context while creating a new embedding run with the selected embedding settings. Downloads export chunks and active vectors as JSONL. Identical normalized content plus ingestion config reuses an existing ready/in-progress document; a changed config creates a new experiment. SHA-256 replaces the old collision-prone contentHash implementation.

## Evaluations

Open `/evals`, upload a JSONL gold set, select documents, methods, k values and optional reranking. Each row requires a question and exactly one nonempty relevance field:

```json
{"question":"What are the stages of RAG?","relevant_sources":["notes.md"]}
```

Alternatively use `relevant_chunk_ids` copied from exported chunks. Unknown labels fail. Filenames must match exactly; repeated filenames are ambiguous, so prefer chunk IDs. Runs save the gold set, config, immutable corpus run IDs, per-query results and aggregate recall@k, hit rate, MRR, 1-recall miss rate and latency. Per-query evaluation latency includes query embedding, scoring and reranking, but excludes the initial Convex snapshot load. Question trace total latency includes that load; its retrieval stage measures the shared engine only. This evaluates retrieval only, with no generated answers. Semantic query embeddings and reranking can still incur charges.

## Architecture

- `src/domain`: shared records, schemas, defaults and inference ports; no Convex/I/O imports.
- `src/ingestion`, `embeddings`, `retrieval`, `generation`, `evaluation`: reusable processing and hosted adapters.
- `src/pipelines`: compose those modules over corpus records, independent of persistence.
- `web/convex/rag.ts`: Node actions, provider calls and SHA-256; no database writes directly.
- `web/convex/documents.ts`: transactional mutations and paginated internal queries.
- `web/convex/schema.ts`: documents, immutable chunks/vectors, indexing runs, success/failure traces and evaluation runs.
- `web/features/rag`: shared form controls, workspace and evaluation view.

Semantic search is exact cosine over a model-filtered Convex snapshot, matching the former CLI rather than the old approximate web vector search. BM25/TF-IDF rebuild statistics from stored context plus original text using the original formulas. There is no Convex full-text substitution. This is intended for bounded experiment corpora, not large-scale search.

Each indexing/re-embedding operation stages an immutable run and activates it in one transaction. Failed runs never replace active evidence. Failed question traces are recorded too. The original full file is reconstructible from chunks, not saved as a blob. Legacy pre-migration documents remain stored but must be uploaded again to obtain manifests/new vectors; no destructive automatic migration runs.

## Defaults and limits

Shared source of truth: `src/domain/config.ts`.

| Setting | Default |
|---|---|
| Chunk size / overlap / boundary | 500 / 50 code points / character |
| Context / prompt caching / output | Off / on when enabled / 160 tokens |
| Embedding provider/model | OpenRouter / nvidia/nemotron-3-embed-1b:free |
| Embedding batch | 32; hosted cap 100 |
| Retrieval / hybrid lexical | semantic / BM25 |
| topK / candidateK / RRF constant | 20 / 150 / 60 |
| Reranking | Off; Cohere selected; explicit model required |
| Full context / character budget | Off / 600,000 |
| Generation model / max output | google/gemma-4-26b-a4b-it / 512 tokens |
| All request timeouts | 120 seconds, configurable up to 600 |
| Evaluation methods / k | All four / 5, 10, 20 |

Uploads are capped at 8 files/request, 500,000 code points/file and 10,000 chunks/document. Exact snapshots are capped at 10,000 chunks. Evals are capped at 200 questions. Convex runtime/storage limits still apply; long serial context/eval runs can exceed an action's deadline. These are synchronous actions, not resumable workers. Historical/failed staging data is retained; cleanup and interrupted-run recovery remain future work.

Output truncation is an error. There are no automatic retries, input token-window fitting, citation verification, answer-quality graders or auth/tenant isolation. System prompts treat documents as untrusted data but are not an injection-proof boundary. Add access control before exposing the deployment to untrusted users.

## Verify offline

```powershell
npm run check
npm --prefix web run check
# Or both:
npm run check:all
```

The root runs typecheck, lint, 44 engine/provider tests and compilation. Web runs typecheck, lint, Convex integration tests and a production build. Convex tests replace fetch with controlled offline responses. `npm run smoke:embeddings` is now an offline adapter-contract check, not a model download.

CI runs the same root/web checks on Node 22. The web build uses a non-network placeholder `NEXT_PUBLIC_CONVEX_URL` when no real deployment is configured. Builds do not deploy Convex. Deployment/codegen needs your selected project and is separate from these offline checks.

See the [architecture guide](docs/architecture-overview.html), [flow guide](docs/code-flow.html), [change report](docs/change-report.html) and [logbook](docs/logbook.md).
