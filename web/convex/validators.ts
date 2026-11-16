import { v } from 'convex/values';
export const embedding = v.object({
  provider: v.union(v.literal('openrouter'), v.literal('voyage'), v.literal('gemini')),
  model: v.string(),
  batchSize: v.number(),
  timeout: v.number(),
});
export const method = v.union(
  v.literal('semantic'),
  v.literal('bm25'),
  v.literal('tfidf'),
  v.literal('hybrid'),
);
export const ingestion = v.object({
  chunkSize: v.number(),
  overlap: v.number(),
  boundary: v.union(v.literal('character'), v.literal('sentence'), v.literal('paragraph')),
  embedding,
  context: v.object({
    enabled: v.boolean(),
    provider: v.optional(
      v.union(
        v.literal('anthropic'),
        v.literal('openrouter'),
        v.literal('openai'),
        v.literal('gemini'),
      ),
    ),
    model: v.string(),
    instruction: v.string(),
    maxTokens: v.number(),
    timeout: v.number(),
    cacheDocument: v.boolean(),
  }),
});
export const ask = v.object({
  embedding,
  method,
  lexicalMethod: v.union(v.literal('bm25'), v.literal('tfidf')),
  topK: v.number(),
  candidateK: v.number(),
  rrfConstant: v.number(),
  timeout: v.number(),
  rerank: v.object({
    enabled: v.boolean(),
    provider: v.union(v.literal('cohere'), v.literal('voyage'), v.literal('openrouter')),
    model: v.string(),
  }),
  fullContext: v.boolean(),
  fullContextMaxChars: v.number(),
  generation: v.object({
    provider: v.literal('openrouter'),
    model: v.string(),
    maxTokens: v.number(),
  }),
});
export const evaluation = v.object({
  retrieval: ask,
  methods: v.array(method),
  ks: v.array(v.number()),
});
