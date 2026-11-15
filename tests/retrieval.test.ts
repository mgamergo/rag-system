import assert from 'node:assert/strict';
import { test } from 'node:test';
import { LexicalIndex, tokenize } from '../src/retrieval/lexical.ts';
import { reciprocalRankFusion, retrieve } from '../src/retrieval/search.ts';
import { fullCorpus } from '../src/retrieval/full-context.ts';
import { evaluate } from '../src/evaluation/retrieval.ts';
import { chunks, encoder, index, neverEncoder } from './helpers.ts';
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

void test('lexical methods require no vectors; hybrid deduplicates branch IDs', async () => {
  const corpus = index();
  for (const method of ['bm25', 'tfidf'] as const)
    assert.equal(
      (
        await retrieve(
          'Cats',
          { chunks: corpus.chunks, embeddings: [] },
          { method, encoder: neverEncoder },
        )
      )[0]?.chunk.chunk_id,
      '1',
    );
  const results = await retrieve('Cats', corpus, { method: 'hybrid', encoder });
  assert.equal(results[0]?.chunk.chunk_id, '1');
  assert.equal(new Set(results.map((r) => r.chunk.chunk_id)).size, results.length);
});
void test('semantic search joins by ID, normalizes, and uses recorded model', async () => {
  const results = await retrieve('RAG?', index(), {
    topK: 5,
    encoder: {
      encode: (texts, options) => {
        assert.deepEqual(texts, ['RAG?']);
        assert.equal(options.model, 'test-encoder');
        assert.equal(options.query, true);
        return Promise.resolve([[2, 0]]);
      },
    },
  });
  assert.deepEqual(
    results.map((r) => [r.chunk.chunk_id, r.score]),
    [
      ['0', 1],
      ['1', 0],
    ],
  );
});
void test('invalid vector indexes fail before query encoding', async () => {
  const corpus = index();
  const a = { chunk_id: '0', model: 'm', embedding: [1, 0] };
  const b = { chunk_id: '1', model: 'm', embedding: [0, 1] };
  for (const embeddings of [
    [],
    [a],
    [a, a],
    [{ ...a, chunk_id: 'unknown' }, b],
    [a, { ...b, model: 'other' }],
    [a, { ...b, embedding: [1] }],
    [a, { ...b, embedding: [] }],
    [a, { ...b, embedding: [0, 0] }],
    [a, { ...b, embedding: [NaN, 0] }],
  ])
    await assert.rejects(
      retrieve('q', { chunks: corpus.chunks, embeddings }, { encoder: neverEncoder }),
    );
  await assert.rejects(
    retrieve(
      'q',
      { ...corpus, chunks: [...corpus.chunks, ...corpus.chunks] },
      { encoder: neverEncoder },
    ),
  );
});
void test('invalid query vectors and limits fail', async () => {
  for (const vector of [[1], [0, 0], [NaN, 0]])
    await assert.rejects(
      retrieve('q', index(), { encoder: { encode: () => Promise.resolve([vector]) } }),
    );
  await assert.rejects(retrieve(' ', index()));
  await assert.rejects(retrieve('q', index(), { topK: 0 }));
  await assert.rejects(retrieve('q', index(), { topK: 5, candidateK: 1 }));
});
void test('full corpus reconstructs Unicode overlap and enforces continuity/budget', () => {
  const [a, b] = chunks('abc😀ef', '😀efghi');
  assert.ok(a && b);
  const records = [{ ...b, start_char: 3, end_char: 9 }, a];
  assert.equal(fullCorpus(records)[0]?.chunk.text, 'abc😀efghi');
  assert.throws(() => fullCorpus(records, 8));
  for (const invalid of [
    { ...b, start_char: 7, end_char: 13 },
    { ...b, start_char: 1, end_char: 7 },
    { ...b, source: 'other' },
  ])
    assert.throws(() => fullCorpus([a, invalid]));
});
void test('evaluation macro recall, miss rate, hit rate, MRR and source dedupe', async () => {
  const corpus = { chunks: chunks('alpha', 'beta'), embeddings: [] };
  const report = await evaluate(
    corpus,
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
      await evaluate(corpus, [{ question: 'alpha beta', relevant_sources: ['notes.md'] }], {
        method: 'bm25',
      })
    ).recall_at_k,
    1,
  );
  for (const input of [
    [],
    [{ question: 'q' }],
    [{ question: 'q', relevant_chunk_ids: [] }],
    [{ question: 'q', relevant_chunk_ids: ['missing'] }],
    [{ question: 'q', relevant_chunk_ids: ['0'], relevant_sources: ['notes.md'] }],
  ])
    await assert.rejects(evaluate(corpus, input, { method: 'bm25' }));
});
