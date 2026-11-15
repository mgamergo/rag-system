import { z } from 'zod';
import {
  nonBlank,
  positiveInteger,
  positiveNumber,
  rerankProviderSchema,
  retrievalText,
} from '../domain/models.ts';
import type { RetrievedChunk } from '../domain/models.ts';
import type { HttpClient, PairScorer } from '../domain/ports.ts';
export const DEFAULT_RERANK_MODEL = 'rerank-v3.5';
import { apiKey, postJson } from '../infrastructure/http.ts';

const recordSchema = z.object({
  index: z.number().int().nonnegative(),
  relevance_score: z.number(),
});
export interface RerankOptions {
  model?: string;
  provider?: z.infer<typeof rerankProviderSchema>;
  topK?: number;
  batchSize?: number;
  timeout?: number;
  scorer?: PairScorer;
  http?: HttpClient;
}
export async function rerank(
  question: string,
  candidates: RetrievedChunk[],
  options: RerankOptions = {},
): Promise<RetrievedChunk[]> {
  nonBlank.parse(question);
  const model = nonBlank.parse(options.model ?? DEFAULT_RERANK_MODEL);
  const provider = rerankProviderSchema.parse(options.provider ?? 'cohere');
  const topK = positiveInteger.parse(options.topK ?? 20);
  const batchSize = positiveInteger.parse(options.batchSize ?? 32);
  const timeout = positiveNumber.parse(options.timeout ?? 120);
  if (!candidates.length) return [];
  if (new Set(candidates.map((item) => item.chunk.chunk_id)).size !== candidates.length)
    throw new Error('Reranking candidates must have unique chunk IDs');
  const texts = candidates.map((item) => retrievalText(item.chunk));
  let records: z.infer<typeof recordSchema>[];
  if (options.scorer) {
    const scores = await options.scorer.score(question, texts, {
      model,
      batchSize,
    });
    if (scores.length !== candidates.length)
      throw new Error('Reranker returned a different number of scores than candidates');
    records = scores.map((score, index) => recordSchema.parse({ index, relevance_score: score }));
  } else {
    const data = await postJson({
      url:
        provider === 'cohere'
          ? 'https://api.cohere.com/v2/rerank'
          : 'https://api.voyageai.com/v1/rerank',
      provider,
      timeout,
      ...(options.http ? { http: options.http } : {}),
      headers: {
        Authorization: `Bearer ${apiKey(provider === 'cohere' ? 'COHERE_API_KEY' : 'VOYAGE_API_KEY')}`,
      },
      payload: {
        model,
        query: question,
        documents: texts,
        [provider === 'cohere' ? 'top_n' : 'top_k']: Math.min(topK, texts.length),
      },
    });
    records =
      provider === 'cohere'
        ? z.object({ results: z.array(recordSchema) }).parse(data).results
        : z.object({ data: z.array(recordSchema) }).parse(data).data;
    if (records.length !== Math.min(topK, texts.length))
      throw new Error('Incorrect reranking result count');
  }
  const seen = new Set<number>();
  return records
    .map((record) => {
      const candidate = candidates[record.index];
      if (!candidate || seen.has(record.index))
        throw new Error('Reranker returned invalid or duplicate candidate indexes');
      seen.add(record.index);
      return { chunk: candidate.chunk, score: record.relevance_score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}
