import assert from 'node:assert/strict';
import { test } from 'node:test';
import { z } from 'zod';
import { runIngestion } from '../src/pipelines/ingestion.ts';
import { runRag } from '../src/pipelines/answer.ts';
import { UNKNOWN_ANSWER, buildMessages } from '../src/generation/chat.ts';
import { chunks, encoder, fakeHttp, index, neverEncoder } from './helpers.ts';
void test('contextual hybrid overfetches and generation sees reranked top-k only', async () => {
  const corpus = await runIngestion(
    [
      {
        document_id: 'doc',
        source: 'notes.md',
        text: 'ACME Q2 report. Revenue grew 3%. Unrelated text.',
      },
    ],
    { chunkSize: 32, overlap: 0, encoder, contextualize: () => Promise.resolve('ACME Q2 results') },
  );
  const calls: Parameters<typeof fakeHttp>[1] = [];
  const result = await runRag('ACME Q2 revenue', corpus, {
    provider: 'openrouter',
    apiKey: 'test-only',
    model: 'chat',
    method: 'hybrid',
    rerankModel: 'ranker',
    topK: 1,
    encoder,
    scorer: {
      score: (_q, texts) => {
        assert.ok(texts.length > 1);
        return Promise.resolve(texts.map((t) => (t.includes('Revenue') ? 10 : 0)));
      },
    },
    http: fakeHttp(
      { choices: [{ message: { content: 'Revenue grew 3%. [1]' } }], usage: { total_tokens: 20 } },
      calls,
    ),
  });
  assert.equal(result.sources.length, 1);
  assert.equal(result.usage.total_tokens, 20);
  const body = z
    .object({ messages: z.array(z.object({ content: z.string() })) })
    .parse(calls[0]?.body);
  assert.equal(
    body.messages[1]?.content,
    buildMessages(result.question, result.sources)[1]?.content,
  );
});
void test('no lexical matches abstains without requesting generation', async () => {
  const result = await runRag('unmatched', index(), {
    provider: 'openrouter',
    apiKey: 'test-only',
    model: 'chat',
    method: 'bm25',
    http: () => {
      throw new Error('Must not request');
    },
  });
  assert.equal(result.answer, UNKNOWN_ANSWER);
  assert.deepEqual(result.sources, []);
});
void test('full context bypasses embeddings and rejects reranking', async () => {
  const corpus = index(chunks('whole document'));
  const result = await runRag('q', corpus, {
    provider: 'openrouter',
    apiKey: 'test-only',
    model: 'chat',
    fullContext: true,
    encoder: neverEncoder,
    http: fakeHttp({ choices: [{ message: { content: 'answer' } }] }),
  });
  assert.equal(result.sources[0]?.chunk.text, 'whole document');
  await assert.rejects(
    runRag('q', corpus, {
      provider: 'openrouter',
      apiKey: 'test-only',
      model: 'chat',
      fullContext: true,
      rerankModel: 'ranker',
    }),
  );
});
