import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { z } from 'zod';
import { runIngestion } from '../src/pipelines/ingestion.js';
import { runRag } from '../src/pipelines/answer.js';
import { UNKNOWN_ANSWER } from '../src/generation/chat.js';
import { chunks, encoder, fakeHttp, index, neverEncoder, temporary } from './helpers.js';

async function cli(
  file: string,
  args: string[],
): Promise<{ code: number | null; stdout: string; stderr: string }> {
  const child = spawn(process.execPath, ['--import', 'tsx', `src/cli/${file}.ts`, ...args], {
    env: { ...process.env, OPENROUTER_API_KEY: '' },
  });
  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (data: Buffer) => {
    stdout += data.toString();
  });
  child.stderr.on('data', (data: Buffer) => {
    stderr += data.toString();
  });
  const result: unknown = await once(child, 'exit');
  const [code] = z.tuple([z.number().nullable(), z.unknown()]).parse(result);
  return { code, stdout, stderr };
}
void test('ingestion through a real local chat server to CLI answer and citations', async (t) => {
  const root = await temporary(t);
  const source = join(root, 'notes.md');
  await writeFile(source, 'RAG retrieves evidence before it generates an answer.');
  await runIngestion(source, { outputDir: join(root, 'index'), encoder });
  const server = createServer((request, response) => {
    assert.equal(request.url, '/v1/chat/completions');
    assert.equal(request.headers.authorization, undefined);
    let text = '';
    request.on('data', (data: Buffer) => {
      text += data.toString();
    });
    request.on('end', () => {
      const payload = z
        .object({ messages: z.array(z.object({ content: z.string() })) })
        .parse(JSON.parse(text));
      assert.ok(payload.messages[1]?.content.includes('RAG retrieves evidence'));
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(
        JSON.stringify({ choices: [{ message: { content: 'Retrieve, then generate. [1]' } }] }),
      );
    });
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(
    () =>
      new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve())),
      ),
  );
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  const result = await cli('ask', [
    'What is RAG?',
    '--artifact-dir',
    join(root, 'index'),
    '--model',
    'test',
    '--retrieval-method',
    'bm25',
    '--ollama-base-url',
    `http://127.0.0.1:${address.port}/v1`,
  ]);
  assert.equal(result.code, 0, result.stderr);
  assert.ok(result.stdout.includes('Retrieve, then generate. [1]'));
  assert.ok(result.stdout.includes(source));
  assert.ok(result.stdout.includes('chars 0:'));
});
void test('contextual hybrid retrieval overfetches and generation receives only reranked top-k', async (t) => {
  const root = await temporary(t);
  const source = join(root, 'notes.md');
  await writeFile(source, 'ACME Q2 report. Revenue grew 3%. Unrelated text.');
  await runIngestion(source, {
    outputDir: join(root, 'index'),
    chunkSize: 32,
    overlap: 0,
    encoder,
    contextualize: () => Promise.resolve('ACME Q2 results'),
  });
  const calls: Parameters<typeof fakeHttp>[1] = [];
  const result = await runRag('ACME Q2 revenue', join(root, 'index'), {
    provider: 'ollama',
    model: 'chat',
    method: 'hybrid',
    rerankModel: 'ranker',
    topK: 1,
    encoder,
    scorer: {
      score: (_question, texts) => {
        assert.ok(texts.length > 1);
        return Promise.resolve(texts.map((text) => (text.includes('Revenue') ? 10 : 0)));
      },
    },
    http: fakeHttp({ choices: [{ message: { content: 'Revenue grew 3%. [1]' } }] }, calls),
  });
  assert.equal(result.sources.length, 1);
  assert.equal(result.question, 'ACME Q2 revenue');
  const payload = z
    .object({ messages: z.array(z.object({ content: z.string() })) })
    .parse(calls[0]?.body);
  const evidence = z
    .object({ sources: z.array(z.object({ context: z.string(), text: z.string() })) })
    .parse(JSON.parse(payload.messages[1]?.content ?? '{}'));
  assert.equal(evidence.sources.length, 1);
  assert.equal(evidence.sources[0]?.context, 'ACME Q2 results');
  assert.equal(evidence.sources[0].text, result.sources[0]?.chunk.text);
});
void test('no lexical matches returns unknown without a generation call', async (t) => {
  const root = await temporary(t);
  await index(root);
  const result = await runRag('unmatched', root, {
    provider: 'ollama',
    model: 'chat',
    method: 'bm25',
    http: () => {
      throw new Error('Must not request');
    },
  });
  assert.equal(result.answer, UNKNOWN_ANSWER);
  assert.deepEqual(result.sources, []);
});
void test('full context bypasses retrieval and rejects reranking', async (t) => {
  const root = await temporary(t);
  await index(root, chunks('whole document'));
  const result = await runRag('q', root, {
    provider: 'ollama',
    model: 'chat',
    fullContext: true,
    encoder: neverEncoder,
    http: fakeHttp({ choices: [{ message: { content: 'answer' } }] }),
  });
  assert.equal(result.sources[0]?.chunk.text, 'whole document');
  await assert.rejects(
    runRag('q', root, {
      provider: 'ollama',
      model: 'chat',
      fullContext: true,
      rerankModel: 'model',
    }),
    /cannot be combined/u,
  );
});
void test('all CLI help works without loading models and syntax errors exit 2', async () => {
  for (const name of ['ask', 'ingest', 'evaluate', 'embed-chunks']) {
    const result = await cli(name, ['--help']);
    assert.equal(result.code, 0, result.stderr);
    assert.ok(result.stdout.includes('Usage:'));
  }
  assert.equal((await cli('ask', [])).code, 2);
  assert.equal((await cli('ingest', ['notes.md', '--chunk-size', 'nonsense'])).code, 2);
});
void test('ingestion CLI handles empty and invalid inputs; evaluation accepts lists', async (t) => {
  const root = await temporary(t);
  assert.equal((await cli('ingest', [root, '--output-dir', join(root, 'out')])).code, 1);
  assert.equal((await cli('ingest', [join(root, 'missing')])).code, 1);
  await index(root, chunks('alpha', 'beta'));
  const dataset = join(root, 'dataset.jsonl');
  await writeFile(dataset, JSON.stringify({ question: 'alpha', relevant_chunk_ids: ['0'] }) + '\n');
  const result = await cli('evaluate', [
    dataset,
    '--artifact-dir',
    root,
    '--methods',
    'bm25',
    'tfidf',
    '--top-k',
    '1',
    '2',
  ]);
  assert.equal(result.code, 0, result.stderr);
  const reports = z
    .array(z.object({ method: z.string(), top_k: z.number() }))
    .parse(JSON.parse(result.stdout));
  assert.deepEqual(reports, [
    { method: 'bm25', top_k: 1 },
    { method: 'bm25', top_k: 2 },
    { method: 'tfidf', top_k: 1 },
    { method: 'tfidf', top_k: 2 },
  ]);
});
void test('missing OpenRouter key fails before retrieval', async () => {
  const result = await cli('ask', [
    'q',
    '--artifact-dir',
    'missing',
    '--provider',
    'openrouter',
    '--model',
    'chat',
  ]);
  assert.equal(result.code, 1);
  assert.ok(result.stderr.includes('OPENROUTER_API_KEY'));
});
