/// <reference types="vite/client" />
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { convexTest } from 'convex-test';
import { z } from 'zod';
import schema from '../convex/schema';
import { api } from '../convex/_generated/api';
import { askConfigSchema, ingestionConfigSchema } from '../../src/domain/config';
const createTest = () => convexTest(schema, modules);
const modules = import.meta.glob('../convex/**/*.{ts,js}');
let embeddingCalls = 0;
let chatCalls = 0;
let failEmbedding = false;
let truncate = false;
beforeEach(() => {
  embeddingCalls = 0;
  chatCalls = 0;
  failEmbedding = false;
  truncate = false;
  for (const name of [
    'OPENROUTER_API_KEY',
    'VOYAGE_API_KEY',
    'GEMINI_API_KEY',
    'ANTHROPIC_API_KEY',
  ])
    vi.stubEnv(name, 'test-only');
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: unknown, init?: RequestInit) => {
      if (typeof input !== 'string' || typeof init?.body !== 'string')
        throw new Error('Unexpected offline request');
      if (input.endsWith('/embeddings')) {
        embeddingCalls++;
        if (failEmbedding) return Response.json({}, { status: 503 });
        const body = z.object({ input: z.array(z.string()) }).parse(JSON.parse(init.body));
        return Response.json({
          data: body.input.map((text, index) => ({
            index,
            embedding: text.includes('Cats') ? [0, 1] : [1, 0],
          })),
        });
      }
      if (input.endsWith('/chat/completions')) {
        chatCalls++;
        return Response.json({
          choices: [
            {
              finish_reason: truncate ? 'length' : 'stop',
              message: { content: 'Evidence answer [1]' },
            },
          ],
          usage: { prompt_tokens: 20, completion_tokens: 5, total_tokens: 25 },
        });
      }
      if (input.endsWith('/messages'))
        return Response.json({
          content: [{ type: 'text', text: 'ACME Q2 report' }],
          usage: { input_tokens: 30, output_tokens: 5, cache_read_input_tokens: 20 },
        });
      throw new Error('No network access permitted in this test');
    }),
  );
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});
async function ingest(
  t: ReturnType<typeof createTest>,
  filename = 'notes.md',
  content = 'ACME revenue grew 3%.',
  config = ingestionConfigSchema.parse({}),
) {
  const result = await t.action(api.rag.ingest, {
    files: [{ filename, content }],
    runName: 'Test ingestion',
    config,
  });
  const id = result[0]?.documentId;
  expect(result[0]?.error).toBeUndefined();
  if (!id) throw new Error('No document');
  return id;
}
test('identical content/config deduplicates transactionally; different config gets a manifest', async () => {
  const t = convexTest(schema, modules);
  await ingest(t);
  await ingest(t);
  expect(embeddingCalls).toBe(1);
  expect(await t.query(api.documents.list, {})).toHaveLength(1);
  await ingest(
    t,
    'notes.md',
    'ACME revenue grew 3%.',
    ingestionConfigSchema.parse({ chunkSize: 100 }),
  );
  expect(await t.query(api.documents.list, {})).toHaveLength(2);
  expect(await t.query(api.documents.runs)).toHaveLength(2);
});
test('context and exact Unicode offsets survive storage/export; context usage saved', async () => {
  const t = convexTest(schema, modules);
  const id = await ingest(
    t,
    'a.md',
    'ACME 😀 Q2 revenue.',
    ingestionConfigSchema.parse({ context: { enabled: true, model: 'test-model' } }),
  );
  const exported = await t.action(api.rag.exportDocument, { documentId: id });
  expect(exported.chunks).toContain('ACME Q2 report');
  expect(JSON.parse(exported.chunks).end_char).toBe(Array.from('ACME 😀 Q2 revenue.').length);
  const runs = await t.query(api.documents.runs);
  expect(runs[0].usageJson).toContain('cache_read_input_tokens');
  expect(JSON.parse(runs[0].configJson).context.instruction).toContain('50-100');
});
test('semantic search filters embedding space and document IDs before scoring', async () => {
  const t = convexTest(schema, modules);
  const a = await ingest(t, 'a.md');
  const b = await ingest(
    t,
    'b.md',
    'Cats sleep.',
    ingestionConfigSchema.parse({ embedding: { model: 'other' } }),
  );
  const result = await t.action(api.rag.ask, {
    question: 'ACME',
    documentIds: [a],
    config: askConfigSchema.parse({}),
  });
  expect(result.sources.map((s) => s.source)).toEqual(['a.md']);
  await expect(
    t.action(api.rag.ask, {
      question: 'ACME',
      documentIds: [b],
      config: askConfigSchema.parse({}),
    }),
  ).rejects.toThrow(/different embedding/u);
  const traces = await t.query(api.documents.traces, {});
  expect(traces.some((t) => t.status === 'failed')).toBe(true);
  expect(traces.find((t) => t.status === 'success')?.totalTokens).toBe(25);
});
test('lexical search needs no query embedding and abstains without generation', async () => {
  const t = convexTest(schema, modules);
  await ingest(t);
  const before = embeddingCalls;
  const result = await t.action(api.rag.ask, {
    question: 'unmatched',
    documentIds: [],
    config: askConfigSchema.parse({ method: 'bm25' }),
  });
  expect(result.answer).toContain("I don't know");
  expect(embeddingCalls).toBe(before);
  expect(chatCalls).toBe(0);
});
test('generation finish_reason length persists a failed trace', async () => {
  const t = convexTest(schema, modules);
  await ingest(t);
  truncate = true;
  await expect(
    t.action(api.rag.ask, {
      question: 'revenue',
      documentIds: [],
      config: askConfigSchema.parse({ method: 'bm25' }),
    }),
  ).rejects.toThrow(/output limit/u);
  const traces = await t.query(api.documents.traces, {});
  expect(traces[0].status).toBe('failed');
  expect(traces[0].error).toContain('output limit');
});
test('failed indexing never activates partial data', async () => {
  const t = convexTest(schema, modules);
  failEmbedding = true;
  const result = await t.action(api.rag.ingest, {
    files: [{ filename: 'a.md', content: 'ACME' }],
    runName: 'failure',
    config: ingestionConfigSchema.parse({}),
  });
  expect(result[0].error).toContain('HTTP 503');
  const docs = await t.query(api.documents.list, {});
  expect(docs[0].status).toBe('failed');
  expect(docs[0].activeRun).toBeUndefined();
});
test('re-embedding preserves chunk IDs and retains prior active run on failure', async () => {
  const t = convexTest(schema, modules);
  const id = await ingest(t);
  const before = await t.action(api.rag.exportDocument, { documentId: id });
  await t.action(api.rag.reembed, {
    documentId: id,
    runName: 'new vectors',
    embedding: { provider: 'openrouter', model: 'new-model', batchSize: 32, timeout: 120 },
  });
  const after = await t.action(api.rag.exportDocument, { documentId: id });
  expect(after.chunks).toBe(before.chunks);
  expect(after.embeddings).toContain('new-model');
  failEmbedding = true;
  await expect(
    t.action(api.rag.reembed, {
      documentId: id,
      runName: 'failed',
      embedding: { provider: 'openrouter', model: 'bad', batchSize: 32, timeout: 120 },
    }),
  ).rejects.toThrow();
  expect((await t.action(api.rag.exportDocument, { documentId: id })).embeddings).toBe(
    after.embeddings,
  );
});
test('evaluation saves method/k reports and dataset; labels outside filter fail', async () => {
  const t = convexTest(schema, modules);
  const id = await ingest(t);
  await t.action(api.rag.runEvaluation, {
    datasetName: 'gold.jsonl',
    datasetJsonl: JSON.stringify({ question: 'revenue', relevant_sources: ['notes.md'] }),
    documentIds: [id],
    config: { retrieval: askConfigSchema.parse({}), methods: ['bm25', 'tfidf'], ks: [1, 2] },
  });
  const runs = await t.query(api.documents.evaluations, {});
  expect(runs[0].status).toBe('success');
  const report = JSON.parse(runs[0].reportJson ?? '{}');
  expect(report.reports).toHaveLength(4);
  expect(report.reports[0].recall_at_k).toBe(1);
  expect(chatCalls).toBe(0);
  await expect(
    t.action(api.rag.runEvaluation, {
      datasetName: 'bad.jsonl',
      datasetJsonl: JSON.stringify({ question: 'q', relevant_sources: ['missing.md'] }),
      documentIds: [id],
      config: { retrieval: askConfigSchema.parse({}), methods: ['bm25'], ks: [1] },
    }),
  ).rejects.toThrow(/Unknown evaluation labels/u);
  expect((await t.query(api.documents.evaluations, {})).some((r) => r.status === 'failed')).toBe(
    true,
  );
});

test('re-embedding updates the document manifest and deduplication config', async () => {
  const t = convexTest(schema, modules);
  const id = await ingest(t);
  const embedding = {
    provider: 'openrouter' as const,
    model: 'new-model',
    batchSize: 32,
    timeout: 120,
  };
  await t.action(api.rag.reembed, { documentId: id, runName: 'updated config', embedding });
  const docs = await t.query(api.documents.list, {});
  expect(JSON.parse(docs[0].configJson ?? '{}').embedding.model).toBe('new-model');
  await ingest(t, 'notes.md', 'ACME revenue grew 3%.', ingestionConfigSchema.parse({ embedding }));
  expect(await t.query(api.documents.list, {})).toHaveLength(1);
});
test('full context spans model spaces and does not embed the query', async () => {
  const t = convexTest(schema, modules);
  await ingest(t, 'a.md', 'ACME revenue.');
  await ingest(
    t,
    'b.md',
    'Cats sleep.',
    ingestionConfigSchema.parse({ embedding: { model: 'other' } }),
  );
  const before = embeddingCalls;
  const result = await t.action(api.rag.ask, {
    question: 'Summarize',
    documentIds: [],
    config: askConfigSchema.parse({ fullContext: true }),
  });
  expect(result.sources).toHaveLength(2);
  expect(embeddingCalls).toBe(before);
  expect(result.sources.every((s) => s.context === '')).toBe(true);
});

test('projects isolate empty selections, explicit selections, deduplication, and chat history', async () => {
  const t = createTest();
  const a = await t.mutation(api.projects.create, { name: 'Research A' });
  const b = await t.mutation(api.projects.create, { name: 'Research B' });
  async function upload(projectId: typeof a) {
    const result = await t.action(api.rag.ingest, {
      projectId,
      files: [{ filename: 'same.md', content: 'ACME project evidence' }],
      runName: 'test',
      config: ingestionConfigSchema.parse({}),
    });
    expect(result[0]?.error).toBeUndefined();
    const id = result[0]?.documentId;
    if (!id) throw new Error('Missing document');
    return id;
  }
  const docA = await upload(a);
  const docB = await upload(b);
  expect(docA).not.toBe(docB);
  expect(await upload(a)).toBe(docA);
  expect(await t.query(api.documents.list, { projectId: a })).toHaveLength(1);
  expect(await t.query(api.documents.list, {})).toHaveLength(0);
  const result = await t.action(api.rag.ask, {
    projectId: a,
    question: 'ACME',
    documentIds: [],
    config: askConfigSchema.parse({ fullContext: true }),
  });
  expect(result.sources.map((s) => s.document_id)).toEqual([docA]);
  await expect(
    t.action(api.rag.ask, {
      projectId: a,
      question: 'ACME',
      documentIds: [docB],
      config: askConfigSchema.parse({ fullContext: true }),
    }),
  ).rejects.toThrow(/does not belong/);
  expect(await t.query(api.documents.traces, { projectId: b })).toHaveLength(0);
  expect(await t.query(api.documents.traces, { projectId: a })).toHaveLength(2);
  await expect(
    t.action(api.rag.exportDocument, { projectId: b, documentId: docA }),
  ).rejects.toThrow(/does not belong/);
  await expect(
    t.action(api.rag.reembed, {
      projectId: b,
      documentId: docA,
      embedding: askConfigSchema.parse({}).embedding,
      runName: 'invalid',
    }),
  ).rejects.toThrow(/does not belong/);
});

test('project settings persist provider models and legacy documents move explicitly', async () => {
  const t = createTest();
  const doc = await ingest(t);
  const projectId = await t.mutation(api.projects.create, { name: 'Project' });
  const ingestion = ingestionConfigSchema.parse({
    context: { enabled: true, provider: 'openrouter', model: 'custom/context' },
  });
  const retrieval = askConfigSchema.parse({
    rerank: { enabled: true, provider: 'openrouter', model: 'cohere/rerank-v3.5' },
  });
  await t.mutation(api.projects.saveSettings, { projectId, ingestion, retrieval });
  const project = (await t.query(api.projects.list, {}))[0];
  expect(project.ingestion.context.provider).toBe('openrouter');
  expect(project.retrieval.rerank.model).toBe('cohere/rerank-v3.5');
  expect(await t.query(api.documents.list, { projectId })).toHaveLength(0);
  await t.mutation(api.projects.importDocument, { projectId, documentId: doc });
  expect(await t.query(api.documents.list, {})).toHaveLength(0);
  expect(await t.query(api.documents.list, { projectId })).toHaveLength(1);
  const exported = await t.action(api.rag.exportDocument, { projectId, documentId: doc });
  expect(exported.chunks).toContain('ACME');
  await expect(
    t.mutation(api.projects.importDocument, { projectId, documentId: doc }),
  ).rejects.toThrow(/unassigned/);
  await expect(t.mutation(api.projects.create, { name: ' ' })).rejects.toThrow(/Project name/);
});
