import { z } from 'zod';
import {
  lexicalMethodSchema,
  nonBlank,
  positiveInteger,
  retrievalMethodSchema,
  vectorSchema,
} from '../domain/models.ts';
import type { Chunk, RetrievedChunk, RetrievalMethod, LexicalMethod } from '../domain/models.ts';
import type { Encoder } from '../domain/ports.ts';
import { embedQuery } from '../embeddings/encoder.ts';
import type { Corpus } from '../domain/models.ts';
import { LexicalIndex } from './lexical.ts';

export interface RetrievalOptions {
  topK?: number;
  candidateK?: number;
  method?: RetrievalMethod;
  lexicalMethod?: LexicalMethod;
  rrfConstant?: number;
  encoder?: Encoder;
}
export function reciprocalRankFusion(
  rankings: RetrievedChunk[][],
  constant = 60,
): RetrievedChunk[] {
  positiveInteger.parse(constant);
  const results = new Map<string, RetrievedChunk>();
  for (const ranking of rankings) {
    const seen = new Set<string>();
    ranking.forEach((item, index) => {
      const key = item.chunk.chunk_id;
      if (seen.has(key)) return;
      seen.add(key);
      results.set(key, {
        chunk: item.chunk,
        score: (results.get(key)?.score ?? 0) + 1 / (constant + index + 1),
      });
    });
  }
  return [...results.values()].sort((a, b) => b.score - a.score);
}
export function unitVector(values: unknown): number[] {
  const vector = vectorSchema.parse(values);
  const norm = Math.hypot(...vector);
  return vector.map((value) => value / norm);
}
function lexicalRanking(
  chunks: Chunk[],
  question: string,
  method: LexicalMethod,
): RetrievedChunk[] {
  const scores = new LexicalIndex(chunks)[method](question);
  return chunks
    .map((chunk, index) => ({ chunk, score: z.number().parse(scores[index]) }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score);
}

export async function retrieve(
  question: string,
  corpus: Corpus,
  options: RetrievalOptions = {},
): Promise<RetrievedChunk[]> {
  nonBlank.parse(question);
  const topK = positiveInteger.parse(options.topK ?? 3);
  const candidateK = positiveInteger.parse(options.candidateK ?? 150);
  const constant = positiveInteger.parse(options.rrfConstant ?? 60);
  const method = retrievalMethodSchema.parse(options.method ?? 'semantic');
  const lexical = lexicalMethodSchema.parse(options.lexicalMethod ?? 'bm25');
  if (candidateK < topK) throw new Error('candidate_k must be >= top_k');
  const { chunks } = corpus;
  if (!chunks.length) throw new Error('The index is empty. Ingest non-empty documents first');
  const byId = new Map(chunks.map((chunk) => [chunk.chunk_id, chunk]));
  if (byId.size !== chunks.length) throw new Error('chunks.jsonl contains duplicate chunk IDs');
  if (method === 'bm25' || method === 'tfidf')
    return lexicalRanking(chunks, question, method).slice(0, topK);

  const vectors = new Map<string, number[]>();
  let model: string | undefined;
  let dimensions: number | undefined;
  for (const record of corpus.embeddings) {
    if (!byId.has(record.chunk_id) || vectors.has(record.chunk_id))
      throw new Error('Unknown or duplicate embedding chunk ID');
    if (model !== undefined && model !== record.model)
      throw new Error('All embeddings must use the same model');
    if (dimensions !== undefined && dimensions !== record.embedding.length)
      throw new Error('Inconsistent embedding dimensions');
    model = record.model;
    dimensions = record.embedding.length;
    vectors.set(record.chunk_id, unitVector(record.embedding));
  }
  if (vectors.size !== chunks.length || model === undefined)
    throw new Error('Every chunk must have exactly one embedding. Re-run ingestion');
  const query = unitVector(
    await embedQuery(question, {
      model,
      ...(options.encoder ? { encoder: options.encoder } : {}),
    }),
  );
  if (query.length !== dimensions)
    throw new Error('Query and stored embedding dimensions differ. Re-run ingestion');
  let ranked = chunks
    .map((chunk) => {
      const vector = vectors.get(chunk.chunk_id);
      if (!vector) throw new Error(`Missing vector for ${chunk.chunk_id}`);
      return {
        chunk,
        score: vector.reduce(
          (sum, value, index) => sum + value * z.number().parse(query[index]),
          0,
        ),
      };
    })
    .sort((a, b) => b.score - a.score);
  if (method === 'hybrid')
    ranked = reciprocalRankFusion(
      [ranked.slice(0, candidateK), lexicalRanking(chunks, question, lexical).slice(0, candidateK)],
      constant,
    );
  return ranked.slice(0, topK);
}
