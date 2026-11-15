import { z } from 'zod';
import {
  boundarySchema,
  lexicalMethodSchema,
  nonBlank,
  positiveInteger,
  positiveNumber,
  retrievalMethodSchema,
  rerankProviderSchema,
} from './models.ts';

export const embeddingConfigSchema = z.object({
  provider: z.enum(['openrouter', 'voyage', 'gemini']).default('openrouter'),
  model: nonBlank.default('nvidia/nemotron-3-embed-1b:free'),
  batchSize: positiveInteger.default(32),
  timeout: positiveNumber.max(600).default(120),
});
export const ingestionConfigSchema = z
  .object({
    chunkSize: positiveInteger.default(500),
    overlap: z.number().int().nonnegative().default(50),
    boundary: boundarySchema.default('character'),
    embedding: embeddingConfigSchema.prefault({}),
    context: z
      .object({
        enabled: z.boolean().default(false),
        model: z.string().default(''),
        instruction: z.string().default(''),
        maxTokens: positiveInteger.default(160),
        timeout: positiveNumber.max(600).default(120),
        cacheDocument: z.boolean().default(true),
      })
      .prefault({}),
  })
  .superRefine((value, ctx) => {
    if (value.overlap >= value.chunkSize)
      ctx.addIssue({ code: 'custom', message: 'Overlap must be smaller than chunk size' });
    if (value.context.enabled && !value.context.model.trim())
      ctx.addIssue({ code: 'custom', message: 'Choose a context model' });
    if (!value.context.enabled && value.context.instruction.trim())
      ctx.addIssue({
        code: 'custom',
        message: 'A custom instruction requires contextual retrieval',
      });
  });
export const retrievalConfigSchema = z.object({
  embedding: embeddingConfigSchema.prefault({}),
  method: retrievalMethodSchema.default('semantic'),
  lexicalMethod: lexicalMethodSchema.default('bm25'),
  topK: positiveInteger.default(20),
  candidateK: positiveInteger.default(150),
  rrfConstant: positiveInteger.default(60),
  timeout: positiveNumber.max(600).default(120),
  rerank: z
    .object({
      enabled: z.boolean().default(false),
      provider: rerankProviderSchema.default('cohere'),
      model: z.string().default(''),
    })
    .prefault({}),
});
export const askConfigSchema = retrievalConfigSchema
  .extend({
    fullContext: z.boolean().default(false),
    fullContextMaxChars: positiveInteger.default(600_000),
    generation: z
      .object({
        provider: z.literal('openrouter').default('openrouter'),
        model: nonBlank.default('google/gemma-4-26b-a4b-it'),
        maxTokens: positiveInteger.default(512),
      })
      .prefault({}),
  })
  .superRefine((value, ctx) => {
    if (value.candidateK < value.topK)
      ctx.addIssue({ code: 'custom', message: 'Candidate count must be at least top k' });
    if (value.rerank.enabled && !value.rerank.model.trim())
      ctx.addIssue({ code: 'custom', message: 'Choose a reranking model' });
    if (value.fullContext && value.rerank.enabled)
      ctx.addIssue({ code: 'custom', message: 'Full context cannot be combined with reranking' });
  });
export const evaluationConfigSchema = z
  .object({
    retrieval: askConfigSchema,
    methods: z
      .array(retrievalMethodSchema)
      .nonempty()
      .default(['semantic', 'bm25', 'tfidf', 'hybrid']),
    ks: z.array(positiveInteger).nonempty().default([5, 10, 20]),
  })
  .superRefine((value, ctx) => {
    if (Math.max(...value.ks) > value.retrieval.candidateK)
      ctx.addIssue({ code: 'custom', message: 'Candidate count must cover every k' });
    if (value.retrieval.fullContext)
      ctx.addIssue({ code: 'custom', message: 'Retrieval evaluation does not use full context' });
  });
export type EmbeddingConfig = z.infer<typeof embeddingConfigSchema>;
export type IngestionConfig = z.infer<typeof ingestionConfigSchema>;
export type AskConfig = z.infer<typeof askConfigSchema>;
export type EvaluationConfig = z.infer<typeof evaluationConfigSchema>;
export const embeddingKey = (config: EmbeddingConfig): string =>
  `${config.provider}:${config.model.trim()}`;
export const DEFAULT_INGESTION = ingestionConfigSchema.parse({});
export const DEFAULT_ASK = askConfigSchema.parse({});
