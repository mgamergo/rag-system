import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  documents: defineTable({
    filename: v.string(),
    contentHash: v.string(),
    status: v.union(v.literal("indexing"), v.literal("ready"), v.literal("failed")),
    chunkCount: v.number(),
    error: v.optional(v.string()),
    createdAt: v.number(),
    indexedAt: v.optional(v.number()),
  }).index("by_created_at", ["createdAt"]),

  chunks: defineTable({
    documentId: v.id("documents"),
    chunkIndex: v.number(),
    text: v.string(),
    startChar: v.number(),
    endChar: v.number(),
    embedding: v.array(v.float64()),
    embeddingModel: v.string(),
    status: v.union(v.literal("indexing"), v.literal("ready")),
    createdAt: v.number(),
  })
    .index("by_document", ["documentId", "chunkIndex"])
    .vectorIndex("by_embedding", {
      vectorField: "embedding",
      dimensions: 2048,
      filterFields: ["documentId", "status"],
    }),

  traces: defineTable({
    question: v.string(),
    answer: v.string(),
    generationModel: v.string(),
    embeddingModel: v.string(),
    retrievalLatencyMs: v.number(),
    generationLatencyMs: v.number(),
    totalLatencyMs: v.number(),
    promptTokens: v.optional(v.number()),
    completionTokens: v.optional(v.number()),
    totalTokens: v.optional(v.number()),
    sourceChunkIds: v.array(v.id("chunks")),
    createdAt: v.number(),
  }).index("by_created_at", ["createdAt"]),
});
