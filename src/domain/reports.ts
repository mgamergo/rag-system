import { z } from 'zod';
export const evaluationReportSchema = z.object({
  method: z.string(),
  top_k: z.number(),
  candidate_k: z.number(),
  rerank_model: z.string().nullable(),
  rerank_provider: z.string(),
  case_count: z.number(),
  recall_at_k: z.number(),
  failure_rate: z.number(),
  hit_rate_at_k: z.number(),
  mrr_at_k: z.number(),
  mean_seconds: z.number(),
  cases: z.array(
    z.object({
      question: z.string(),
      recall: z.number(),
      hit: z.boolean(),
      reciprocal_rank: z.number(),
      seconds: z.number(),
      retrieved_chunk_ids: z.array(z.string()),
    }),
  ),
});
export const savedReportSchema = z.object({ reports: z.array(evaluationReportSchema) });
