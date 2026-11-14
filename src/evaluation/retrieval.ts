import { join } from 'node:path';
import { z } from 'zod';
import { nonBlank, positiveInteger } from '../domain/models.js';
import type { HttpClient, PairScorer } from '../domain/ports.js';
import { loadChunks } from '../storage/jsonl.js';
import { retrieve } from '../retrieval/search.js';
import type { RetrievalOptions } from '../retrieval/search.js';
import { rerank } from '../retrieval/reranking.js';
import type { RerankOptions } from '../retrieval/reranking.js';

export const evaluationCaseSchema = z.union([
  z.object({
    question: nonBlank,
    relevant_chunk_ids: z.array(nonBlank).nonempty(),
    relevant_sources: z.never().optional(),
  }),
  z.object({
    question: nonBlank,
    relevant_sources: z.array(nonBlank).nonempty(),
    relevant_chunk_ids: z.never().optional(),
  }),
]);
export interface EvaluationOptions extends RetrievalOptions {
  rerankModel?: string;
  rerankProvider?: RerankOptions['provider'];
  scorer?: PairScorer;
  http?: HttpClient;
}
export async function evaluate(
  artifactDir: string,
  input: unknown,
  options: EvaluationOptions = {},
) {
  const cases = z.array(evaluationCaseSchema).nonempty().parse(input);
  const topK = positiveInteger.parse(options.topK ?? 20);
  const candidateK = positiveInteger.parse(options.candidateK ?? 150);
  const method = options.method ?? 'hybrid';
  if (candidateK < topK) throw new Error('candidate_k must be >= positive top_k');
  const chunks = await loadChunks(join(artifactDir, 'chunks.jsonl'));
  const knownIds = new Set(chunks.map((chunk) => chunk.chunk_id));
  const knownSources = new Set(chunks.map((chunk) => chunk.source));
  const validated = cases.map((item) => {
    const kind = item.relevant_chunk_ids ? 'chunk' : 'source';
    const labels = item.relevant_chunk_ids ?? item.relevant_sources;
    const relevant = new Set(labels);
    const known = kind === 'chunk' ? knownIds : knownSources;
    if ([...relevant].some((label) => !known.has(label)))
      throw new Error('Unknown evaluation labels');
    return { question: item.question, relevant, kind };
  });
  const rows = [];
  for (const item of validated) {
    const start = performance.now();
    let results = await retrieve(item.question, artifactDir, {
      ...options,
      method,
      topK: options.rerankModel ? candidateK : topK,
    });
    if (options.rerankModel)
      results = await rerank(item.question, results, {
        ...options,
        model: options.rerankModel,
        provider: options.rerankProvider ?? 'local',
        topK,
      });
    const labels = results.map((result) =>
      item.kind === 'chunk' ? result.chunk.chunk_id : result.chunk.source,
    );
    const hits = new Set(labels.filter((label) => item.relevant.has(label)));
    const first = labels.findIndex((label) => item.relevant.has(label));
    rows.push({
      question: item.question,
      recall: hits.size / item.relevant.size,
      hit: hits.size > 0,
      reciprocal_rank: first < 0 ? 0 : 1 / (first + 1),
      seconds: (performance.now() - start) / 1000,
      retrieved_chunk_ids: results.map((result) => result.chunk.chunk_id),
    });
  }
  const mean = (values: number[]): number => values.reduce((a, b) => a + b, 0) / rows.length;
  const recall = mean(rows.map((row) => row.recall));
  return {
    method,
    top_k: topK,
    candidate_k: candidateK,
    rerank_model: options.rerankModel ?? null,
    rerank_provider: options.rerankProvider ?? 'local',
    case_count: rows.length,
    recall_at_k: recall,
    failure_rate: 1 - recall,
    hit_rate_at_k: mean(rows.map((row) => Number(row.hit))),
    mrr_at_k: mean(rows.map((row) => row.reciprocal_rank)),
    mean_seconds: mean(rows.map((row) => row.seconds)),
    cases: rows,
  };
}
