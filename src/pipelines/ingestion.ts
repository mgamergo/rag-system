import { randomUUID } from 'node:crypto';
import { boundarySchema, nonBlank, positiveInteger } from '../domain/models.ts';
import type { Chunk, ChunkBoundary, Corpus, Document } from '../domain/models.ts';
import type { Contextualize } from '../domain/ports.ts';
import { embedChunks } from '../embeddings/encoder.ts';
import type { EmbeddingOptions } from '../embeddings/encoder.ts';
import { chunkText, validateChunkConfig } from '../ingestion/chunking.ts';
export interface IngestionOptions extends EmbeddingOptions {
  chunkSize?: number;
  overlap?: number;
  boundary?: ChunkBoundary;
  contextualize?: Contextualize;
}
export async function runIngestion(
  documents: Document[],
  options: IngestionOptions = {},
): Promise<Corpus> {
  const { chunkSize = 500, overlap = 50, boundary = 'character' } = options;
  validateChunkConfig(chunkSize, overlap);
  boundarySchema.parse(boundary);
  positiveInteger.parse(options.batchSize ?? 32);
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
  return { chunks, embeddings };
}
