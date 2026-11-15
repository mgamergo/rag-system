# Option-to-control mapping

The web is the only user interface. Every old command option is accounted for below. Config objects are checked by Convex validators and shared Zod schemas. Removed local/path settings are explicit deployment substitutions, not silently ignored flags.

| Old command | Flag / input | UI control | Convex argument | Default / replacement | Validation |
|---|---|---|---|---|---|
| ingest | input document-path | Multi-file upload (.md/.txt) | files[].filename/content | Required; filesystem discovery replaced | Strict UTF-8; nonblank; 8 files/request; 500,000 code points/file |
| ingest | --output-dir | Run-name text | runName | Generated timestamp name | Convex run replaces directory; no server filesystem path |
| ingest | --chunk-size | Number | config.chunkSize | 500 | Positive integer |
| ingest | --overlap | Number | config.overlap | 50 | Integer >=0 and < chunkSize |
| ingest | --chunk-boundary | Select | config.boundary | character | character / sentence / paragraph |
| ingest | --model | Provider select + model text | config.embedding.provider/model | openrouter / nvidia/nemotron-3-embed-1b:free | openrouter / voyage / gemini; nonblank model; local MiniLM removed |
| ingest | --batch-size | Number | config.embedding.batchSize | 32 | Positive integer; HTTP batches capped at 100 |
| ingest | --device | Hosted-inference explanation | none | Removed | Local CPU/CUDA/WebGPU replaced with hosted inference |
| ingest | --context-model | Enable checkbox + model text | config.context.enabled/model | Disabled; no model default | Nonblank model when enabled |
| ingest | --context-prompt | Optional textarea + instruction-file import | config.context.instruction | Existing DEFAULT_INSTRUCTION | Custom instruction requires enabled context |
| ingest | --context-max-tokens | Number | config.context.maxTokens | 160 | Positive integer |
| ingest | --context-timeout | Number (seconds) | config.context.timeout | 120 | Positive finite; maximum 600 seconds |
| ingest | --no-context-cache | Cache-document checkbox | config.context.cacheDocument | true | Boolean; unchecked disables prefix caching |
| ask | input question | Textarea | question | Required | Trimmed nonblank string |
| ask | --artifact-dir | Document checkboxes + indexed embedding-space select | documentIds / config.embedding | All compatible ready documents | Explicit IDs must be ready; semantic/hybrid reject mismatched models |
| ask | --provider | Generation-provider select | config.generation.provider | openrouter | Hosted only; Ollama removed |
| ask | --model | Model text | config.generation.model | google/gemma-4-26b-a4b-it | Nonblank; no live availability check |
| ask | --top-k | Number | config.topK | 20 | Positive integer |
| ask | --device | Hosted-inference explanation | none | Removed | No local execution devices |
| ask | --retrieval-method | Select | config.method | semantic | semantic / bm25 / tfidf / hybrid |
| ask | --lexical-method | Select | config.lexicalMethod | bm25 | bm25 / tfidf |
| ask | --candidate-k | Number | config.candidateK | 150 | Integer >= topK |
| ask | --rerank-model [id] | Enable checkbox + model text | config.rerank.enabled/model | Disabled; explicit hosted model required | Nonblank when enabled; local default removed |
| ask | --rerank-provider | Select | config.rerank.provider | cohere | cohere / voyage |
| ask | --full-context | Checkbox | config.fullContext | false | Cannot combine with reranking |
| ask | --full-context-max-chars | Number | config.fullContextMaxChars | 600000 | Positive integer, Unicode code points |
| ask | --ollama-base-url | Provider explanation | none | Removed | No localhost or arbitrary generation URL |
| ask | --max-tokens | Number | config.generation.maxTokens | 512 | Positive integer; finish_reason length fails |
| ask | --timeout | Number (seconds) | config.timeout | 120 | Positive finite; maximum 600; per generation/rerank request |
| embed-chunks | input chunks-path | Document row + re-embed button | documentId / embedding | Selected ready document | Convex document replaces JSONL path; text/context/chunk IDs preserved |
| embed-chunks | --output | Saved embedding run + JSONL download | runName; exportDocument(documentId) | Re-embedding run / filename-based downloads | No filesystem output path; previous active run retained until success |
| evaluate | input dataset | JSONL upload | datasetName / datasetJsonl | Required | Nonempty cases; exactly one nonempty relevance field; known labels; max 200 questions |
| evaluate | --artifact-dir | Document checkboxes + embedding controls | documentIds / config.retrieval.embedding | All compatible ready documents | Semantic configs require same model; lexical-only configs can span models |
| evaluate | --methods | Method checkboxes | config.methods | semantic,bm25,tfidf,hybrid | Nonempty enum array |
| evaluate | --top-k | Add/remove number inputs | config.ks | 5,10,20 | Nonempty positive integers; max <= candidateK |
| evaluate | --candidate-k | Number | config.retrieval.candidateK | 150 | Integer >= every k |
| evaluate | --rerank-model | Enable checkbox + model text | config.retrieval.rerank.enabled/model | Disabled | Nonblank model when enabled |
| evaluate | --rerank-provider | Select | config.retrieval.rerank.provider | cohere | cohere / voyage |
| evaluate | --device | Hosted-inference explanation | none | Removed | No local execution |
| evaluate | --output | Saved run + report download | Automatically saved; documents.evaluations query | Report stored in Convex | Download config, corpus versions, reports and original gold set |
| all | --help / -h | Inline help, README and reference | none | Always visible | CLI syntax/exit statuses removed |
| extra | Library rrfConstant | Number | config.rrfConstant | 60 | Positive integer |
| extra | Embedding request timeout | Number | config.embedding.timeout | 120 | Positive finite; maximum 600 |

## Removal scope

The mapping covers removal of src/cli, CLI npm scripts, Commander, local ONNX adapter/Transformers dependency, obsolete dist/cli and local-adapter outputs, and CLI-specific subprocess/filesystem tests. Existing domain/ranking/provider tests are ported. Filesystem discovery is replaced by browser uploads; artifact path destinations become Convex runs and downloads. No unrelated local corpus or environment file is deleted.

## Parity and intentional changes

Exact cosine, BM25/TF-IDF formulas, tokenizer, RRF and candidateK-to-topK behavior are retained. Native Convex approximate vector search and its fixed 2048-dimensional index are replaced by exact model-filtered ranking so arbitrary hosted model dimensions work without schema changes. Lexical statistics are rebuilt over selected documents from stored context plus original text. Convex full-text search is not substituted because its tokenizer/ranking would change results.

Local MiniLM, local cross-encoder and local Ollama are removed as deployment backends. Their hosted replacements are configurable and are not claimed to produce identical vectors/scores/answers. The old custom Ollama URL is removed. GENERATION_MODEL server configuration is replaced by the explicit generation-model control and saved question config. Context models remain explicit, with no guessed default.

New safety limits: request timeouts at most 600 seconds; 8 upload files per action; 500,000 code points per file; 10,000 chunks per document/snapshot; 200 evaluation questions. Context windows still require user judgment. Chunk batches are persisted with count/byte bounds. Sources use uploaded filenames rather than absolute machine paths; use chunk-ID labels if filenames repeat.
