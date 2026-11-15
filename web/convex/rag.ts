'use node';
import { createHash } from 'node:crypto';
import { z } from 'zod';
import { v } from 'convex/values';
import { action } from './_generated/server';
import type { ActionCtx } from './_generated/server';
import type { PaginationResult } from 'convex/server';
import type { Doc, Id } from './_generated/dataModel';
import { internal } from './_generated/api';
import * as validators from './validators';
import {
  askConfigSchema,
  embeddingConfigSchema,
  embeddingKey,
  evaluationConfigSchema,
  ingestionConfigSchema,
} from '../../src/domain/config';
import type { AskConfig } from '../../src/domain/config';
import type { Corpus, Embedding } from '../../src/domain/models';
import { chunkText } from '../../src/ingestion/chunking';
import { normalizeDocument } from '../../src/ingestion/documents';
import { DEFAULT_INSTRUCTION } from '../../src/ingestion/contextualize';
import {
  askConfigured,
  embeddingOptions,
  ingestConfigured,
  retrievalOptions,
} from '../../src/pipelines/configured';
import { embedChunks } from '../../src/embeddings/encoder';
import { evaluate, evaluationCaseSchema } from '../../src/evaluation/retrieval';
import { parseJsonl, serializeJsonl } from '../../src/storage/jsonl';
const hash = (text: string) => createHash('sha256').update(text).digest('hex');
const message = (error: unknown) => (error instanceof Error ? error.message : 'Processing failed');

// Load immutable active run snapshots in bounded pages. Filter model BEFORE scoring.
async function snapshot(
  ctx: ActionCtx,
  documentIds: Id<'documents'>[],
  config: AskConfig,
  allModels = false,
) {
  const documents = await ctx.runQuery(internal.documents.selection, {
    documentIds,
    model: embeddingKey(config.embedding),
    allModels,
  });
  const corpus: Corpus = { chunks: [], embeddings: [] };
  const ids = new Map<string, Id<'chunks'>>();
  const versions: { documentId: Id<'documents'>; runId: Id<'runs'> }[] = [];
  for (const doc of documents) {
    if (!doc.activeRun) continue;
    versions.push({ documentId: doc._id, runId: doc.activeRun });
    let cursor: string | null = null;
    do {
      const page: PaginationResult<Doc<'chunks'>> = await ctx.runQuery(
        internal.documents.chunkPage,
        { runId: doc.chunkRun ?? doc.activeRun, paginationOpts: { numItems: 64, cursor } },
      );
      for (const c of page.page) {
        const id = c.engineId ?? c._id;
        ids.set(id, c._id);
        corpus.chunks.push({
          chunk_id: id,
          document_id: doc._id,
          source: doc.filename,
          chunk_index: c.chunkIndex,
          text: c.text,
          start_char: c.startChar,
          end_char: c.endChar,
          context: c.context ?? '',
        });
      }
      cursor = page.isDone ? null : page.continueCursor;
    } while (cursor);
    if (!allModels) {
      do {
        const page: PaginationResult<Doc<'vectors'>> = await ctx.runQuery(
          internal.documents.vectorPage,
          { runId: doc.activeRun, paginationOpts: { numItems: 16, cursor } },
        );
        for (const e of page.page) {
          if (e.model !== embeddingKey(config.embedding))
            throw new Error('Stored embedding model differs from the selected space');
          corpus.embeddings.push({ chunk_id: e.chunkId, model: e.model, embedding: e.embedding });
        }
        cursor = page.isDone ? null : page.continueCursor;
      } while (cursor);
    }
    if (corpus.chunks.length > 10_000)
      throw new Error(
        'Select a smaller corpus: exact retrieval is limited to 10,000 chunks per request',
      );
  }
  return { corpus, ids, versions, documents };
}
async function persist(
  ctx: ActionCtx,
  documentId: Id<'documents'>,
  runId: Id<'runs'>,
  corpus: Corpus,
  includeChunks: boolean,
) {
  // Vector dimensions vary by provider. Bound batches by encoded bytes as well as count.
  let start = 0;
  while (start < corpus.embeddings.length) {
    let end = start;
    let bytes = 0;
    while (end < corpus.embeddings.length && end - start < 16) {
      const vector = corpus.embeddings[end];
      if (!vector) break;
      const size = Buffer.byteLength(JSON.stringify(vector));
      if (size > 500_000) throw new Error('An embedding exceeds the storage batch limit');
      if (bytes + size > 500_000 && end > start) break;
      bytes += size;
      end++;
    }
    await ctx.runMutation(internal.documents.addBatch, {
      documentId,
      runId,
      chunks: includeChunks
        ? corpus.chunks.slice(start, end).map((c) => ({
            chunk_id: c.chunk_id,
            chunk_index: c.chunk_index,
            text: c.text,
            start_char: c.start_char,
            end_char: c.end_char,
            context: c.context,
          }))
        : [],
      vectors: corpus.embeddings.slice(start, end),
    });
    start = end;
  }
}
export const ingest = action({
  args: {
    files: v.array(v.object({ filename: v.string(), content: v.string() })),
    runName: v.string(),
    config: validators.ingestion,
  },
  handler: async (
    ctx,
    args,
  ): Promise<
    { filename: string; documentId?: Id<'documents'>; reused?: boolean; error?: string }[]
  > => {
    const config = ingestionConfigSchema.parse(args.config);
    if (config.context.enabled && !config.context.instruction.trim())
      config.context.instruction = DEFAULT_INSTRUCTION;
    const files = z
      .array(z.object({ filename: z.string().min(1), content: z.string().max(1_000_000) }))
      .min(1)
      .max(8)
      .parse(args.files);
    const name = args.runName.trim() || `Ingestion ${new Date().toISOString()}`;
    const results = [];
    for (const file of files) {
      let reservation:
        { documentId: Id<'documents'>; runId: Id<'runs'> | null; reused: boolean } | undefined;
      try {
        const content = normalizeDocument(file.filename, file.content);
        if (Array.from(content).length > 500_000)
          throw new Error('Upload limit is 500,000 code points per file');
        if (chunkText(content, config).length > 10_000)
          throw new Error('Choose larger chunks: at most 10,000 chunks per document');
        reservation = await ctx.runMutation(internal.documents.reserve, {
          filename: file.filename,
          contentHash: hash(content),
          configHash: hash(JSON.stringify(config)),
          configJson: JSON.stringify(config),
          name,
          model: embeddingKey(config.embedding),
        });
        if (reservation.reused) {
          results.push({
            filename: file.filename,
            documentId: reservation.documentId,
            reused: true,
          });
          continue;
        }
        if (!reservation.runId) throw new Error('Missing indexing run');
        const result = await ingestConfigured(
          { document_id: reservation.documentId, source: file.filename, text: content },
          config,
        );
        await persist(ctx, reservation.documentId, reservation.runId, result.corpus, true);
        await ctx.runMutation(internal.documents.finish, {
          documentId: reservation.documentId,
          runId: reservation.runId,
          chunkCount: result.corpus.chunks.length,
          model: embeddingKey(config.embedding),
          usageJson: JSON.stringify(result.contextUsage),
        });
        results.push({
          filename: file.filename,
          documentId: reservation.documentId,
          reused: false,
        });
      } catch (error) {
        if (reservation?.runId && !reservation.reused)
          await ctx.runMutation(internal.documents.finish, {
            documentId: reservation.documentId,
            runId: reservation.runId,
            chunkCount: 0,
            model: embeddingKey(config.embedding),
            usageJson: 'null',
            error: message(error),
          });
        results.push({ filename: file.filename, error: message(error) });
      }
    }
    return results;
  },
});
export const ask = action({
  args: { question: v.string(), documentIds: v.array(v.id('documents')), config: validators.ask },
  handler: async (
    ctx,
    args,
  ): Promise<{
    answer: string;
    sources: Corpus['chunks'][number][];
    scores: number[];
    traceId: Id<'traces'>;
    latencyMs: number;
    usage: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
  }> => {
    const start = Date.now();
    const traceId = await ctx.runMutation(internal.documents.startTrace, {
      question: args.question,
      configJson: JSON.stringify({ ...args.config, documentIds: args.documentIds }),
      model: args.config.generation.model,
      embeddingModel: embeddingKey(args.config.embedding),
    });
    try {
      const config = askConfigSchema.parse(args.config);
      const question = z.string().trim().min(1).parse(args.question);
      const data = await snapshot(
        ctx,
        args.documentIds,
        config,
        config.fullContext || config.method === 'bm25' || config.method === 'tfidf',
      );
      const result = await askConfigured(question, data.corpus, config);
      const sourceChunkIds = result.sources.flatMap((s) => {
        const id = data.ids.get(s.chunk.chunk_id);
        return id ? [id] : [];
      });
      await ctx.runMutation(internal.documents.endTrace, {
        id: traceId,
        answer: result.answer,
        sourcesJson: JSON.stringify({ sources: result.sources, versions: data.versions }),
        sourceChunkIds,
        retrievalLatencyMs: result.retrievalLatencyMs,
        generationLatencyMs: result.generationLatencyMs,
        totalLatencyMs: Date.now() - start,
        usageJson: JSON.stringify(result.usage),
      });
      return {
        answer: result.answer,
        sources: result.sources.map((s) => s.chunk),
        scores: result.sources.map((s) => s.score),
        traceId,
        latencyMs: Date.now() - start,
        usage: result.usage,
      };
    } catch (error) {
      await ctx.runMutation(internal.documents.endTrace, {
        id: traceId,
        answer: '',
        sourcesJson: '[]',
        sourceChunkIds: [],
        retrievalLatencyMs: 0,
        generationLatencyMs: 0,
        totalLatencyMs: Date.now() - start,
        usageJson: '{}',
        error: message(error),
      });
      throw new Error(message(error));
    }
  },
});
export const reembed = action({
  args: { documentId: v.id('documents'), embedding: validators.embedding, runName: v.string() },
  handler: async (ctx, args): Promise<Id<'runs'>> => {
    const embedding = embeddingConfigSchema.parse(args.embedding);
    const config = askConfigSchema.parse({ embedding });
    const data = await snapshot(ctx, [args.documentId], config, true);
    const doc = data.documents[0];
    if (!doc?.configJson) throw new Error('Document is missing its ingestion manifest');
    const current = ingestionConfigSchema.parse(JSON.parse(doc.configJson));
    const manifest = ingestionConfigSchema.parse({ ...current, embedding });
    const configJson = JSON.stringify(manifest);
    const runId = await ctx.runMutation(internal.documents.beginReembed, {
      documentId: args.documentId,
      configJson,
      name: args.runName.trim() || 'Re-embedding',
    });
    try {
      const embeddings: Embedding[] = await embedChunks(
        data.corpus.chunks,
        embeddingOptions(embedding),
      );
      await persist(ctx, args.documentId, runId, { chunks: data.corpus.chunks, embeddings }, false);
      await ctx.runMutation(internal.documents.finish, {
        documentId: args.documentId,
        runId,
        chunkCount: data.corpus.chunks.length,
        model: embeddingKey(embedding),
        usageJson: 'null',
        configHash: hash(configJson),
      });
      return runId;
    } catch (error) {
      await ctx.runMutation(internal.documents.finish, {
        documentId: args.documentId,
        runId,
        chunkCount: 0,
        model: embeddingKey(embedding),
        usageJson: 'null',
        error: message(error),
      });
      throw new Error(message(error));
    }
  },
});
export const exportDocument = action({
  args: { documentId: v.id('documents') },
  handler: async (ctx, args): Promise<{ chunks: string; embeddings: string }> => {
    const config = askConfigSchema.parse({});
    const docs = await ctx.runQuery(internal.documents.selection, {
      documentIds: [args.documentId],
      allModels: true,
    });
    const doc = docs[0];
    if (!doc?.embeddingModel) throw new Error('Document has no shared-engine embeddings');
    const separator = doc.embeddingModel.indexOf(':');
    config.embedding = embeddingConfigSchema.parse({
      provider: doc.embeddingModel.slice(0, separator),
      model: doc.embeddingModel.slice(separator + 1),
    });
    const data = await snapshot(ctx, [args.documentId], config);
    return {
      chunks: serializeJsonl(data.corpus.chunks),
      embeddings: serializeJsonl(data.corpus.embeddings),
    };
  },
});
export const runEvaluation = action({
  args: {
    datasetName: v.string(),
    datasetJsonl: v.string(),
    documentIds: v.array(v.id('documents')),
    config: validators.evaluation,
  },
  handler: async (ctx, args): Promise<Id<'evaluations'>> => {
    const id = await ctx.runMutation(internal.documents.startEvaluation, {
      name: args.datasetName,
      datasetJson: args.datasetJsonl,
      configJson: JSON.stringify(args.config),
      documentIds: args.documentIds,
    });
    const reports = [];
    try {
      const config = evaluationConfigSchema.parse(args.config);
      const cases = parseJsonl(args.datasetJsonl, evaluationCaseSchema);
      if (cases.length > 200) throw new Error('Run at most 200 questions per evaluation');
      const semantic = config.methods.some((m) => m === 'semantic' || m === 'hybrid');
      const data = await snapshot(ctx, args.documentIds, config.retrieval, !semantic);
      for (const method of config.methods)
        for (const topK of config.ks)
          reports.push(
            await evaluate(data.corpus, cases, {
              ...retrievalOptions(config.retrieval),
              method,
              topK,
            }),
          );
      await ctx.runMutation(internal.documents.endEvaluation, {
        id,
        reportJson: JSON.stringify({ reports, versions: data.versions }),
      });
      return id;
    } catch (error) {
      await ctx.runMutation(internal.documents.endEvaluation, {
        id,
        reportJson: JSON.stringify({ reports }),
        error: message(error),
      });
      throw new Error(message(error));
    }
  },
});
