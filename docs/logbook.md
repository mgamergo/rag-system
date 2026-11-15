# Development logbook

## 2026-11-15: one engine, web-only interface

Replaced the two implementations with Next.js controls and Convex Node actions over the shared `src/` domain/pipeline modules. Removed CLI commands, Commander and local ONNX inference. The full option mapping and deletion scope are recorded in [option-mapping.md](option-mapping.md).

Preserved code-point chunking, natural-boundary heuristics, Anthropic whole-document contextualization/cache breakpoint, separate original text/context, contextual embeddings and lexical ranking, BM25/TF-IDF math, RRF, candidateK-to-topK reranking, full-corpus reconstruction, grounded generation prompt and retrieval evaluator metrics.

Chose exact cosine instead of fixed-dimension Convex approximate search to retain CLI ranking and support provider-selected dimensions. Model/document filtering happens before scoring. Store text/context and rebuild lexical statistics for the selected corpus rather than substitute Convex full-text semantics. This deliberately trades scale for experiment parity.

Added SHA-256 content/config deduplication, immutable staging runs with atomic activation, persisted manifests/context usage, re-embedding preserving chunk IDs, chunk/vector downloads, failed question traces, explicit request deadlines, output-truncation checks and saved evaluation datasets/configs/reports/corpus versions. Legacy data remains untouched and requires re-upload to use the new engine.

Hosted inference replaces machine-local model/device/server choices. There is no claim of numerical parity between different embedding/reranking/generation models. Filesystem paths become browser uploads, Convex document selection and downloads. Existing chunk JSONL input is replaced by stored-document re-embedding.

Tests are offline: shared engine/provider behavior plus Convex persistence/action integration with intercepted fetch. No paid inference calls or deployment are part of verification. CI includes the web tests. See change-report.html for final check results and known gaps.

Remaining work: auth/tenant ownership, durable resumable jobs and interrupted-run recovery, historical-run cleanup, input token budgets, richer failure usage accounting, answer-quality evaluation, stable passage-level gold data and large-corpus indexing. Synchronous action deadlines and exact-snapshot limits are explicit limitations.

Final local verification: npm run check:all passed root/web typecheck and lint, 44 engine/provider tests, 10 Convex integration tests, TypeScript compilation and Next production routes / and /evals. Root format:check, offline embedding contract smoke and Git whitespace checks passed. CI was mirrored locally on Windows, not executed in an Ubuntu runner. No paid API calls, deployment or live browser/backend test were performed.
