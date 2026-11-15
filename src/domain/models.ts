import { z } from 'zod';

export const nonBlank = z.string().refine((value) => value.trim().length > 0, 'Must not be blank');
export const positiveInteger = z.number().int().positive();
export const positiveNumber = z.number().positive();
export const boundarySchema = z.enum(['character', 'sentence', 'paragraph']);
export type ChunkBoundary = z.infer<typeof boundarySchema>;
export const retrievalMethodSchema = z.enum(['semantic', 'bm25', 'tfidf', 'hybrid']);
export type RetrievalMethod = z.infer<typeof retrievalMethodSchema>;
export const lexicalMethodSchema = z.enum(['bm25', 'tfidf']);
export type LexicalMethod = z.infer<typeof lexicalMethodSchema>;
export const rerankProviderSchema = z.enum(['cohere', 'voyage']);
export const generationProviderSchema = z.enum(['openrouter']);

// Serialized names are the artifact contract; internal code uses the same names.
export const chunkSchema = z
  .object({
    chunk_id: nonBlank,
    document_id: nonBlank,
    source: z.string(),
    chunk_index: z.number().int().nonnegative(),
    text: z.string(),
    start_char: z.number().int().nonnegative(),
    end_char: z.number().int().nonnegative(),
    context: z.string().default(''),
  })
  .strict();
export type Chunk = z.infer<typeof chunkSchema>;
export const vectorSchema = z
  .array(z.number())
  .nonempty()
  .refine((values) => {
    const norm = Math.hypot(...values);
    return Number.isFinite(norm) && norm > 0;
  }, 'Embeddings must have a finite, non-zero norm');
export const embeddingSchema = z.object({
  chunk_id: nonBlank,
  model: nonBlank,
  embedding: vectorSchema,
});
export type Embedding = z.infer<typeof embeddingSchema>;
export interface Document {
  document_id: string;
  source: string;
  text: string;
}
export interface IngestionFailure {
  source: string;
  error: string;
}
export interface RetrievedChunk {
  chunk: Chunk;
  score: number;
}
export interface RawChunk {
  text: string;
  start_idx: number;
  end_idx: number;
}
export const retrievalText = (chunk: Chunk): string =>
  chunk.context ? `${chunk.context}\n\n${chunk.text}` : chunk.text;

// Code-point offsets stay exact for supplementary Unicode characters as well.
export const characters = (text: string): string[] => Array.from(text);
export const characterLength = (text: string): number => characters(text).length;
export const characterSlice = (text: string, start: number, end?: number): string =>
  characters(text).slice(start, end).join('');

export interface Corpus {
  chunks: Chunk[];
  embeddings: Embedding[];
}
