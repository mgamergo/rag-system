import { randomUUID } from 'node:crypto';
import { join } from 'node:path';
import { boundarySchema, nonBlank, positiveInteger } from '../domain/models.js';
import type { Chunk, ChunkBoundary, IngestionFailure } from '../domain/models.js';
import type { Contextualize } from '../domain/ports.js';
import { embedChunks } from '../embeddings/encoder.js';
import type { EmbeddingOptions } from '../embeddings/encoder.js';
import { chunkText, validateChunkConfig } from '../ingestion/chunking.js';
import { ingestDocuments } from '../ingestion/documents.js';
import { expandPath, saveChunks, saveEmbeddings, saveFailures } from '../storage/jsonl.js';

export interface IngestionOptions extends EmbeddingOptions {
  outputDir?: string;
  chunkSize?: number;
  overlap?: number;
  boundary?: ChunkBoundary;
  contextualize?: Contextualize;
}
export interface IngestionResult {
  outputDir: string;
  chunkCount: number;
  embeddingCount: number;
  failures: IngestionFailure[];
}
export async function runIngestion(
  documentPath: string,
  options: IngestionOptions = {},
): Promise<IngestionResult> {
  const { chunkSize = 500, overlap = 50, boundary = 'character' } = options;
  validateChunkConfig(chunkSize, overlap);
  boundarySchema.parse(boundary);
  positiveInteger.parse(options.batchSize ?? 32);
  const outputDir = expandPath(
    options.outputDir ?? join('artifacts', `run-${randomUUID().replaceAll('-', '')}`),
  );
  const { documents, failures } = await ingestDocuments(documentPath);
  const chunks: Chunk[] = [];
  for (const document of documents) {
    for (const [index, raw] of chunkText(document.text, {
      chunkSize,
      overlap,
      boundary,
    }).entries()) {
      const context = options.contextualize
        ? nonBlank.parse(await options.contextualize(document.text, raw.text)).trim()
        : '';
      chunks.push({
        chunk_id: randomUUID(),
        document_id: document.document_id,
        source: document.source,
        chunk_index: index,
        text: raw.text,
        start_char: raw.start_idx,
        end_char: raw.end_idx,
        context,
      });
    }
  }
  const embeddings = await embedChunks(chunks, options);
  await saveChunks(chunks, join(outputDir, 'chunks.jsonl'));
  await saveEmbeddings(embeddings, join(outputDir, 'embeddings.jsonl'));
  await saveFailures(failures, join(outputDir, 'failures.jsonl'));
  return { outputDir, chunkCount: chunks.length, embeddingCount: embeddings.length, failures };
}
