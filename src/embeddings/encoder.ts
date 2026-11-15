import { nonBlank, positiveInteger, retrievalText, vectorSchema } from '../domain/models.ts';
import type { Chunk, Embedding } from '../domain/models.ts';
import type { Encoder } from '../domain/ports.ts';
export const DEFAULT_EMBEDDING_MODEL = 'openrouter:nvidia/nemotron-3-embed-1b:free';
import { remoteEncoder } from './remote.ts';

export interface EmbeddingOptions {
  model?: string;
  batchSize?: number;
  encoder?: Encoder;
}
function encoderFor(model: string): Encoder {
  nonBlank.parse(model);
  return remoteEncoder();
}

export async function embedChunks(
  chunks: Chunk[],
  options: EmbeddingOptions = {},
): Promise<Embedding[]> {
  const model = nonBlank.parse(options.model ?? DEFAULT_EMBEDDING_MODEL);
  const batchSize = positiveInteger.parse(options.batchSize ?? 32);
  if (!chunks.length) return [];
  const vectors = await (options.encoder ?? encoderFor(model)).encode(chunks.map(retrievalText), {
    model,
    batchSize,
    query: false,
  });
  if (vectors.length !== chunks.length)
    throw new Error('Embedding model returned a different number of vectors than chunks');
  let dimensions: number | undefined;
  return chunks.map((chunk, index) => {
    const vector = vectorSchema.parse(vectors[index]);
    if (dimensions !== undefined && dimensions !== vector.length)
      throw new Error('Inconsistent embedding dimensions');
    dimensions = vector.length;
    return { chunk_id: chunk.chunk_id, model, embedding: vector };
  });
}

export async function embedQuery(
  question: string,
  options: EmbeddingOptions = {},
): Promise<number[]> {
  const model = nonBlank.parse(options.model ?? DEFAULT_EMBEDDING_MODEL);
  const vectors = await (options.encoder ?? encoderFor(model)).encode([question], {
    model,
    batchSize: 1,
    query: true,
  });
  if (vectors.length !== 1) throw new Error('Expected exactly one query embedding');
  return vectorSchema.parse(vectors[0]);
}
