import { nonBlank, positiveInteger, retrievalText, vectorSchema } from '../domain/models.js';
import type { Chunk, Embedding } from '../domain/models.js';
import type { Encoder } from '../domain/ports.js';
import { DEFAULT_EMBEDDING_MODEL, localEncoder } from './local.js';
import { remoteEncoder } from './remote.js';

export interface EmbeddingOptions {
  model?: string;
  batchSize?: number;
  device?: string | undefined;
  encoder?: Encoder;
}
function encoderFor(model: string): Encoder {
  return /^(gemini|voyage):/u.test(model) ? remoteEncoder() : localEncoder;
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
    device: options.device,
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
    device: options.device,
    query: true,
  });
  if (vectors.length !== 1) throw new Error('Expected exactly one query embedding');
  return vectorSchema.parse(vectors[0]);
}
