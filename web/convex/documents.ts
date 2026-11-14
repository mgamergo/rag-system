import { v } from "convex/values";
import { internalMutation, internalQuery, query } from "./_generated/server";

const statusValidator = v.union(
  v.literal("indexing"),
  v.literal("ready"),
  v.literal("failed"),
);

export const list = query({
  args: {},
  handler: async (ctx) =>
    await ctx.db.query("documents").withIndex("by_created_at").order("desc").collect(),
});

export const create = internalMutation({
  args: { filename: v.string(), contentHash: v.string(), createdAt: v.number() },
  handler: async (ctx, args) =>
    await ctx.db.insert("documents", {
      filename: args.filename,
      contentHash: args.contentHash,
      status: "indexing",
      chunkCount: 0,
      createdAt: args.createdAt,
    }),
});

export const addChunkBatch = internalMutation({
  args: {
    documentId: v.id("documents"),
    embeddingModel: v.string(),
    createdAt: v.number(),
    chunks: v.array(
      v.object({
        chunkIndex: v.number(),
        text: v.string(),
        startChar: v.number(),
        endChar: v.number(),
        embedding: v.array(v.float64()),
      }),
    ),
  },
  handler: async (ctx, args) => {
    for (const chunk of args.chunks) {
      await ctx.db.insert("chunks", { ...chunk, documentId: args.documentId, embeddingModel: args.embeddingModel, status: "indexing", createdAt: args.createdAt });
    }
  },
});

export const finish = internalMutation({
  args: { documentId: v.id("documents"), status: statusValidator, chunkCount: v.number(), error: v.optional(v.string()), indexedAt: v.number() },
  handler: async (ctx, args) => {
    const update = {
      status: args.status,
      chunkCount: args.chunkCount,
      indexedAt: args.indexedAt,
    };
    if (args.error === undefined) {
      await ctx.db.patch(args.documentId, update);
      if (args.status === "ready") {
        const chunks = await ctx.db.query("chunks").withIndex("by_document", (query) => query.eq("documentId", args.documentId)).collect();
        await Promise.all(chunks.map((chunk) => ctx.db.patch(chunk._id, { status: "ready" })));
      }
      return;
    }
    await ctx.db.patch(args.documentId, { ...update, error: args.error });
  },
});

export const byIds = internalQuery({
  args: { ids: v.array(v.id("chunks")) },
  handler: async (ctx, args) => {
    const chunks = await Promise.all(args.ids.map((id) => ctx.db.get(id)));
    return chunks.filter((chunk) => chunk !== null);
  },
});

export const saveTrace = internalMutation({
  args: {
    question: v.string(), answer: v.string(), generationModel: v.string(), embeddingModel: v.string(),
    retrievalLatencyMs: v.number(), generationLatencyMs: v.number(), totalLatencyMs: v.number(),
    promptTokens: v.optional(v.number()), completionTokens: v.optional(v.number()), totalTokens: v.optional(v.number()),
    sourceChunkIds: v.array(v.id("chunks")), createdAt: v.number(),
  },
  handler: async (ctx, args) => await ctx.db.insert("traces", args),
});
