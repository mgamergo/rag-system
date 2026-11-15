import assert from 'node:assert/strict';
import { test } from 'node:test';
import { chunkText } from '../src/ingestion/chunking.ts';
import { normalizeDocument, decodeDocument } from '../src/ingestion/documents.ts';
import { runIngestion } from '../src/pipelines/ingestion.ts';
import { embedChunks } from '../src/embeddings/encoder.ts';
import { characterSlice, chunkSchema } from '../src/domain/models.ts';
import {
  ingestionConfigSchema,
  askConfigSchema,
  evaluationConfigSchema,
} from '../src/domain/config.ts';
import { parseJsonl, serializeJsonl } from '../src/storage/jsonl.ts';
import { chunks, encoder, neverEncoder } from './helpers.ts';
for (const boundary of ['character', 'sentence', 'paragraph'] as const) {
  for (const overlap of [0, 5, 29])
    void test(`${boundary} offsets, progress, overlap, Unicode and coverage (${overlap})`, () => {
      const text =
        'First 😀 sentence. Next sentence.\n\nA paragraph with a long sentence and punctuation.\n\nEnd.';
      const result = chunkText(text, { chunkSize: 30, overlap, boundary });
      assert.equal(result[0]?.start_idx, 0);
      assert.equal(result.at(-1)?.end_idx, Array.from(text).length);
      for (const [i, item] of result.entries()) {
        assert.equal(item.text, characterSlice(text, item.start_idx, item.end_idx));
        assert.ok(Array.from(item.text).length <= 30);
        const previous = result[i - 1];
        if (previous) {
          assert.equal(item.start_idx, previous.end_idx - overlap);
          assert.ok(item.start_idx > previous.start_idx);
        }
      }
    });
}
void test('natural boundaries select the final eligible sentence or paragraph', () => {
  assert.equal(
    chunkText('First sentence. Next sentence. More text.', {
      chunkSize: 25,
      overlap: 2,
      boundary: 'sentence',
    })[0]?.text,
    'First sentence. ',
  );
  assert.equal(
    chunkText('First paragraph.\n\nNext paragraph is longer.', {
      chunkSize: 26,
      overlap: 2,
      boundary: 'paragraph',
    })[0]?.text,
    'First paragraph.\n\n',
  );
});

void test('strict decoding and normalization preserve normalized offsets', () => {
  assert.equal(normalizeDocument('notes.MD', '\uFEFFACME 😀\r\nRevenue.\r'), 'ACME 😀\nRevenue.\n');
  assert.throws(() => decodeDocument(new Uint8Array([255]).buffer));
  assert.throws(() => normalizeDocument('x.pdf', 'text'));
  assert.throws(() => normalizeDocument('x.md', '   '));
});
void test('ingestion preserves original evidence and contextual embedding inputs', async () => {
  const text = 'ACME 😀 Q2. Revenue grew 3%.';
  const seen: string[] = [];
  const corpus = await runIngestion([{ document_id: 'doc', source: 'a.md', text }], {
    chunkSize: 15,
    overlap: 2,
    contextualize: (whole) => {
      assert.equal(whole, text);
      return Promise.resolve(' ACME report ');
    },
    encoder: {
      encode: (texts) => {
        seen.push(...texts);
        return Promise.resolve(texts.map(() => [1, 0]));
      },
    },
  });
  assert.equal(corpus.chunks.length, corpus.embeddings.length);
  for (const c of corpus.chunks) {
    assert.equal(c.context, 'ACME report');
    assert.equal(c.text, characterSlice(text, c.start_char, c.end_char));
    assert.ok(seen.includes(`ACME report\n\n${c.text}`));
  }
});
void test('empty corpus avoids models and invalid options fail before inference', async () => {
  assert.deepEqual(await runIngestion([], { encoder: neverEncoder }), {
    chunks: [],
    embeddings: [],
  });
  for (const config of [
    { chunkSize: 0 },
    { overlap: -1 },
    { chunkSize: 5, overlap: 5 },
    { batchSize: 0 },
  ])
    await assert.rejects(runIngestion([], { ...config, encoder: neverEncoder }));
});
void test('context and embedding failures yield no partial ingestion result', async () => {
  const docs = [{ document_id: 'd', source: 'a.md', text: 'original' }];
  await assert.rejects(runIngestion(docs, { encoder: neverEncoder }));
  await assert.rejects(runIngestion(docs, { encoder, contextualize: () => Promise.resolve(' ') }));
});
void test('invalid vectors, dimensions and counts are rejected', async () => {
  for (const vectors of [[], [[0, 0]], [[NaN, 0]], [[Infinity]], [[]], [[1], [1, 2]]])
    await assert.rejects(
      embedChunks(chunks('one', 'two'), { encoder: { encode: () => Promise.resolve(vectors) } }),
    );
});
void test('JSONL upload/export validates lines and supports legacy empty context', () => {
  const [chunk] = chunks('one');
  assert.ok(chunk);
  assert.equal(parseJsonl(serializeJsonl([chunk]), chunkSchema)[0]?.text, 'one');
  assert.throws(() => parseJsonl('{}\ninvalid', chunkSchema), /line 1/u);
  assert.throws(() => serializeJsonl([{ n: NaN }]));
});
void test('UI configuration schemas enforce conditional controls and defaults', () => {
  assert.equal(ingestionConfigSchema.parse({}).chunkSize, 500);
  assert.equal(askConfigSchema.parse({}).topK, 20);
  for (const input of [
    { chunkSize: 10, overlap: 10 },
    { context: { enabled: true } },
    { context: { instruction: 'custom' } },
  ])
    assert.equal(ingestionConfigSchema.safeParse(input).success, false);
  for (const input of [
    { topK: 2.5 },
    { candidateK: 1 },
    { rerank: { enabled: true } },
    { fullContext: true, rerank: { enabled: true, model: 'rank' } },
    { timeout: 0 },
  ])
    assert.equal(askConfigSchema.safeParse(input).success, false);
  assert.equal(
    evaluationConfigSchema.safeParse({ retrieval: askConfigSchema.parse({}), ks: [200] }).success,
    false,
  );
});
