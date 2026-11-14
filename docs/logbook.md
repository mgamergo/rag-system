# Development logbook

## Local RAG pipeline

The CLI supports UTF-8 document ingestion, character and natural-boundary chunks, local and hosted embeddings, optional Anthropic context, semantic and lexical search, hybrid rank fusion, local and hosted reranking, full-corpus prompts, and labeled retrieval evaluation.

Runtime schemas protect artifacts and provider responses. Processing completes before artifact replacement. Each JSONL file is replaced atomically. Public pipeline functions retain question and source metadata for inspection.

## Web application

The Next.js interface uses Convex for document storage, vector search, and successful question traces. OpenRouter supplies embeddings and answers. Its semantic retrieval flow remains independent of CLI experiments.

## Validation

Run `npm run check` from the root for type checking, lint, behavioral tests, and compilation. The embedding smoke test downloads and runs the default local ONNX model. Build and lint `web/` separately.
