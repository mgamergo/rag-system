import assert from 'node:assert/strict';
import { test } from 'node:test';
import { rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { LexicalIndex, tokenize } from '../src/retrieval/lexical.js';
import { reciprocalRankFusion, retrieve } from '../src/retrieval/search.js';
import { fullCorpus } from '../src/retrieval/full-context.js';
import { evaluate } from '../src/evaluation/retrieval.js';
import { saveChunks } from '../src/storage/jsonl.js';
import { chunks, encoder, index, neverEncoder, temporary } from './helpers.js';

void test('TF-IDF uses sublinear TF, smoothed IDF and cosine normalization', () => {
  const lexical = new LexicalIndex(chunks('alpha beta', 'alpha', 'gamma'));
  assert.ok((lexical.tfidf('beta')[0] ?? 0) > 0);
  assert.deepEqual(lexical.tfidf('beta').slice(1), [0, 0]);
  assert.ok(Math.abs((lexical.tfidf('alpha beta')[0] ?? 0) - 1) < 1e-12);
  assert.deepEqual(lexical.tfidf('unknown'), [0, 0, 0]);
  assert.deepEqual(new LexicalIndex(chunks('')).tfidf('q'), [0]);
});
void test('BM25 keeps identifiers and normalizes by length', () => {
  const lexical = new LexicalIndex(
    chunks('Error TS-999', 'Error TS-998', 'Error TS-999 ' + 'noise '.repeat(50)),
  );
  const scores = lexical.bm25('TS-999');
  assert.ok((scores[0] ?? 0) > (scores[2] ?? 0));
  assert.equal(scores[1], 0);
  assert.deepEqual(tokenize('Café TS-999'), ['café', 'ts-999']);
  assert.deepEqual(lexical.bm25('missing'), [0, 0, 0]);
  assert.deepEqual(new LexicalIndex([]).bm25('q'), []);
  for (const options of [{ k1: 0 }, { b: 2 }, { b: NaN }])
    assert.throws(() => lexical.bm25('q', options));
});
void test('lexical search needs no vectors and hybrid fusion combines branches', async (t) => {
  const root = await temporary(t);
  await index(root);
  await rm(join(root, 'embeddings.jsonl'));
  for (const method of ['bm25', 'tfidf'] as const)
    assert.equal(
      (await retrieve('Cats', root, { method, encoder: neverEncoder }))[0]?.chunk.chunk_id,
      '1',
    );
  await index(root);
  const results = await retrieve('Cats', root, { method: 'hybrid', encoder });
  assert.equal(results[0]?.chunk.chunk_id, '1');
  assert.equal(new Set(results.map((item) => item.chunk.chunk_id)).size, results.length);
});
void test('fusion ignores score scales and duplicate chunk IDs per branch', () => {
  const [a, b] = chunks('a', 'b').map((chunk) => ({ chunk, score: 1000 }));
  assert.ok(a && b);
  const fused = reciprocalRankFusion([
    [a, b],
    [b, b],
  ]);
  assert.equal(fused[0]?.chunk.chunk_id, '1');
  assert.equal(fused[0].score, 1 / 62 + 1 / 61);
  assert.throws(() => reciprocalRankFusion([], 0));
});
void test('semantic search joins by ID, normalizes vectors and uses the recorded model', async (t) => {
  const root = await temporary(t);
  await index(root);
  const results = await retrieve('RAG?', root, {
    topK: 5,
    device: 'cpu',
    encoder: {
      encode: (texts, options) => {
        assert.deepEqual(texts, ['RAG?']);
        assert.equal(options.model, 'test-encoder');
        assert.equal(options.device, 'cpu');
        assert.equal(options.query, true);
        return Promise.resolve([[2, 0]]);
      },
    },
  });
  assert.deepEqual(
    results.map((item) => [item.chunk.chunk_id, item.score]),
    [
      ['0', 1],
      ['1', 0],
    ],
  );
});
void test('invalid indexes fail before query encoding', async (t) => {
  const root = await temporary(t);
  const records = await index(root);
  const a = { chunk_id: '0', model: 'model', embedding: [1, 0] };
  const b = { chunk_id: '1', model: 'model', embedding: [0, 1] };
  for (const invalid of [
    [],
    [a],
    [a, a],
    [{ ...a, chunk_id: 'unknown' }, b],
    [a, { ...b, model: 'other' }],
    [a, { ...b, model: ' ' }],
    ...[[1], [], [0, 0], [null, 0], ['1', 0], [true, 0]].map((embedding) => [
      a,
      { ...b, embedding },
    ]),
  ]) {
    await writeFile(
      join(root, 'embeddings.jsonl'),
      invalid.map((record) => JSON.stringify(record)).join('\n'),
    );
    await assert.rejects(retrieve('q', root, { encoder: neverEncoder }));
  }
  await saveChunks([...records, ...records], join(root, 'chunks.jsonl'));
  await assert.rejects(retrieve('q', root, { encoder: neverEncoder }), /duplicate/u);
});
void test('invalid query vectors and retrieval limits fail', async (t) => {
  const root = await temporary(t);
  await index(root);
  for (const vector of [[1], [0, 0], [NaN, 0]])
    await assert.rejects(
      retrieve('q', root, { encoder: { encode: () => Promise.resolve([vector]) } }),
    );
  await assert.rejects(retrieve(' ', root));
  await assert.rejects(retrieve('q', root, { topK: 0 }));
  await assert.rejects(retrieve('q', root, { topK: 5, candidateK: 1 }));
});
void test('full corpus reconstructs overlap in offset order and enforces budget', async (t) => {
  const root = await temporary(t);
  const [a, b] = chunks('abcdef', 'defghi');
  assert.ok(a && b);
  await saveChunks([{ ...b, start_char: 3, end_char: 9 }, a], join(root, 'chunks.jsonl'));
  assert.equal((await fullCorpus(root))[0]?.chunk.text, 'abcdefghi');
  await assert.rejects(fullCorpus(root, 8), /exceeds/u);
  for (const invalid of [
    { ...b, start_char: 7, end_char: 13 },
    { ...b, start_char: 1, end_char: 7 },
    { ...b, source: 'other' },
  ]) {
    await saveChunks([a, invalid], join(root, 'chunks.jsonl'));
    await assert.rejects(fullCorpus(root), /Cannot reconstruct/u);
  }
});
void test('evaluation reports macro recall, failure, hit rate and MRR; source recall deduplicates', async (t) => {
  const root = await temporary(t);
  await saveChunks(chunks('alpha', 'beta'), join(root, 'chunks.jsonl'));
  const report = await evaluate(
    root,
    [
      { question: 'alpha', relevant_chunk_ids: ['0', '1'] },
      { question: 'unknown', relevant_chunk_ids: ['1'] },
    ],
    { method: 'bm25', topK: 1 },
  );
  assert.equal(report.recall_at_k, 0.25);
  assert.equal(report.failure_rate, 0.75);
  assert.equal(report.mrr_at_k, 0.5);
  assert.equal(report.hit_rate_at_k, 0.5);
  assert.equal(
    (
      await evaluate(root, [{ question: 'alpha beta', relevant_sources: ['notes.md'] }], {
        method: 'bm25',
      })
    ).recall_at_k,
    1,
  );
  for (const value of [
    [],
    [{ question: 'q' }],
    [{ question: 'q', relevant_chunk_ids: [] }],
    [{ question: 'q', relevant_chunk_ids: ['missing'] }],
    [{ question: 'q', relevant_chunk_ids: ['0'], relevant_sources: ['notes.md'] }],
  ])
    await assert.rejects(evaluate(root, value, { method: 'bm25' }));
});
