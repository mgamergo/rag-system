import { z } from 'zod';
import { v } from 'convex/values';
import { paginationOptsValidator } from 'convex/server';
import { internalMutation, internalQuery, query } from './_generated/server';

export const list = query({
  args: {},
  handler: async (ctx) =>
    await ctx.db.query('documents').withIndex('by_created_at').order('desc').collect(),
});
export const runs = query({
  args: {},
  handler: async (ctx) =>
    await ctx.db.query('runs').withIndex('by_created_at').order('desc').take(100),
});
export const traces = query({
  args: {},
  handler: async (ctx) =>
    await ctx.db.query('traces').withIndex('by_created_at').order('desc').take(50),
});
export const evaluations = query({
  args: {},
  handler: async (ctx) =>
    await ctx.db.query('evaluations').withIndex('by_created_at').order('desc').take(50),
});
export const reserve = internalMutation({
  args: {
    filename: v.string(),
    contentHash: v.string(),
    configHash: v.string(),
    configJson: v.string(),
    name: v.string(),
    model: v.string(),
  },
  handler: async (ctx, args) => {
    const matches = await ctx.db
      .query('documents')
      .withIndex('by_hash', (q) =>
        q.eq('contentHash', args.contentHash).eq('configHash', args.configHash),
      )
      .collect();
    const existing = matches.find((d) => d.status !== 'failed');
    if (existing)
      return {
        documentId: existing._id,
        runId: existing.activeRun ?? existing.pendingRun ?? null,
        reused: true,
      };
    const documentId = await ctx.db.insert('documents', {
      filename: args.filename,
      contentHash: args.contentHash,
      configHash: args.configHash,
      configJson: args.configJson,
      status: 'indexing',
      chunkCount: 0,
      createdAt: Date.now(),
      embeddingModel: args.model,
    });
    const runId = await ctx.db.insert('runs', {
      documentId,
      kind: 'ingest',
      name: args.name,
      status: 'indexing',
      configJson: args.configJson,
      createdAt: Date.now(),
    });
    await ctx.db.patch(documentId, { pendingRun: runId });
    return { documentId, runId, reused: false };
  },
});
export const beginReembed = internalMutation({
  args: { documentId: v.id('documents'), configJson: v.string(), name: v.string() },
  handler: async (ctx, args) => {
    const doc = await ctx.db.get(args.documentId);
    if (!doc || doc.status !== 'ready' || !doc.activeRun)
      throw new Error('Select a document indexed by the shared engine');
    if (doc.pendingRun) throw new Error('This document already has an indexing run');
    const runId = await ctx.db.insert('runs', {
      documentId: args.documentId,
      kind: 'reembed',
      name: args.name,
      status: 'indexing',
      configJson: args.configJson,
      createdAt: Date.now(),
    });
    await ctx.db.patch(args.documentId, { pendingRun: runId });
    return runId;
  },
});
export const addBatch = internalMutation({
  args: {
    documentId: v.id('documents'),
    runId: v.id('runs'),
    chunks: v.array(
      v.object({
        chunk_id: v.string(),
        chunk_index: v.number(),
        text: v.string(),
        start_char: v.number(),
        end_char: v.number(),
        context: v.string(),
      }),
    ),
    vectors: v.array(
      v.object({ chunk_id: v.string(), model: v.string(), embedding: v.array(v.float64()) }),
    ),
  },
  handler: async (ctx, args) => {
    const doc = await ctx.db.get(args.documentId);
    if (doc?.pendingRun !== args.runId) throw new Error('Indexing run is no longer active');
    for (const c of args.chunks)
      await ctx.db.insert('chunks', {
        documentId: args.documentId,
        runId: args.runId,
        engineId: c.chunk_id,
        chunkIndex: c.chunk_index,
        text: c.text,
        startChar: c.start_char,
        endChar: c.end_char,
        context: c.context,
        createdAt: Date.now(),
      });
    for (const e of args.vectors)
      await ctx.db.insert('vectors', {
        documentId: args.documentId,
        runId: args.runId,
        chunkId: e.chunk_id,
        model: e.model,
        embedding: e.embedding,
      });
  },
});
export const finish = internalMutation({
  args: {
    documentId: v.id('documents'),
    runId: v.id('runs'),
    chunkCount: v.number(),
    model: v.string(),
    usageJson: v.string(),
    configHash: v.optional(v.string()),
    error: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const doc = await ctx.db.get(args.documentId);
    const run = await ctx.db.get(args.runId);
    if (!doc || !run || doc.pendingRun !== args.runId)
      throw new Error('Indexing run is no longer active');
    await ctx.db.patch(args.runId, {
      status: args.error ? 'failed' : 'ready',
      error: args.error,
      usageJson: args.usageJson,
      completedAt: Date.now(),
    });
    if (args.error) {
      await ctx.db.patch(doc._id, {
        pendingRun: undefined,
        ...(!doc.activeRun ? { status: 'failed' as const, error: args.error } : {}),
      });
      return;
    }
    // Activation is one transaction; staged rows are never read until this pointer changes.
    await ctx.db.patch(doc._id, {
      status: 'ready',
      ...(run.kind === 'reembed'
        ? { configJson: run.configJson, configHash: args.configHash }
        : {}),
      activeRun: args.runId,
      chunkRun: doc.chunkRun ?? doc.activeRun ?? args.runId,
      pendingRun: undefined,
      chunkCount: args.chunkCount,
      embeddingModel: args.model,
      indexedAt: Date.now(),
      error: undefined,
    });
  },
});
export const selection = internalQuery({
  args: {
    documentIds: v.array(v.id('documents')),
    model: v.optional(v.string()),
    allModels: v.boolean(),
  },
  handler: async (ctx, args) => {
    const all = args.documentIds.length
      ? await Promise.all(args.documentIds.map((id) => ctx.db.get(id)))
      : await ctx.db.query('documents').withIndex('by_created_at').collect();
    if (args.documentIds.length && all.some((d) => !d || d.status !== 'ready' || !d.activeRun))
      throw new Error('Selected documents must be ready and have a shared-engine manifest');
    const ready = all.flatMap((d) =>
      d !== null && d.status === 'ready' && d.activeRun ? [d] : [],
    );
    if (
      !args.allModels &&
      args.documentIds.length &&
      ready.some((d) => d.embeddingModel !== args.model)
    )
      throw new Error(
        'Selected documents use different embedding models; choose a compatible embedding space',
      );
    return ready.filter((d) => args.allModels || d.embeddingModel === args.model);
  },
});
export const chunkPage = internalQuery({
  args: { runId: v.id('runs'), paginationOpts: paginationOptsValidator },
  handler: async (ctx, args) =>
    await ctx.db
      .query('chunks')
      .withIndex('by_run', (q) => q.eq('runId', args.runId))
      .paginate(args.paginationOpts),
});
export const vectorPage = internalQuery({
  args: { runId: v.id('runs'), paginationOpts: paginationOptsValidator },
  handler: async (ctx, args) =>
    await ctx.db
      .query('vectors')
      .withIndex('by_run', (q) => q.eq('runId', args.runId))
      .paginate(args.paginationOpts),
});
export const startTrace = internalMutation({
  args: {
    question: v.string(),
    configJson: v.string(),
    model: v.string(),
    embeddingModel: v.string(),
  },
  handler: async (ctx, args) =>
    await ctx.db.insert('traces', {
      question: args.question,
      answer: '',
      configJson: args.configJson,
      generationModel: args.model,
      embeddingModel: args.embeddingModel,
      retrievalLatencyMs: 0,
      generationLatencyMs: 0,
      totalLatencyMs: 0,
      sourceChunkIds: [],
      createdAt: Date.now(),
      status: 'running',
    }),
});
export const endTrace = internalMutation({
  args: {
    id: v.id('traces'),
    answer: v.string(),
    sourcesJson: v.string(),
    sourceChunkIds: v.array(v.id('chunks')),
    retrievalLatencyMs: v.number(),
    generationLatencyMs: v.number(),
    totalLatencyMs: v.number(),
    usageJson: v.string(),
    error: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const usage = z
      .object({
        prompt_tokens: z.number().optional(),
        completion_tokens: z.number().optional(),
        total_tokens: z.number().optional(),
      })
      .parse(JSON.parse(args.usageJson));
    const { id, usageJson, ...rest } = args;
    void usageJson;
    await ctx.db.patch(id, {
      ...rest,
      promptTokens: usage.prompt_tokens,
      completionTokens: usage.completion_tokens,
      totalTokens: usage.total_tokens,
      status: args.error ? 'failed' : 'success',
    });
  },
});
export const startEvaluation = internalMutation({
  args: {
    name: v.string(),
    datasetJson: v.string(),
    configJson: v.string(),
    documentIds: v.array(v.id('documents')),
  },
  handler: async (ctx, args) =>
    await ctx.db.insert('evaluations', { ...args, status: 'running', createdAt: Date.now() }),
});
export const endEvaluation = internalMutation({
  args: { id: v.id('evaluations'), reportJson: v.string(), error: v.optional(v.string()) },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.id, {
      status: args.error ? 'failed' : 'success',
      reportJson: args.reportJson,
      error: args.error,
      completedAt: Date.now(),
    });
  },
});
