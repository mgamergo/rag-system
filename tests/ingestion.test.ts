import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdir, readFile, writeFile, access } from 'node:fs/promises';
import { join } from 'node:path';
import { chunkText } from '../src/ingestion/chunking.js';
import { findDocuments } from '../src/ingestion/documents.js';
import { runIngestion } from '../src/pipelines/ingestion.js';
import { embedChunks } from '../src/embeddings/encoder.js';
import { characterSlice } from '../src/domain/models.js';
import { loadChunks, loadEmbeddings, saveJsonl } from '../src/storage/jsonl.js';
import { chunks, encoder, neverEncoder, temporary } from './helpers.js';

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
void test('directory ingestion persists linked artifacts and exact normalized offsets', async (t) => {
  const root = await temporary(t);
  const input = join(root, 'input');
  await mkdir(join(input, 'nested'), { recursive: true });
  for (const directory of ['.git', 'node_modules', '.cache', '.venv', 'venv', '__pycache__']) {
    await mkdir(join(input, directory));
    await writeFile(join(input, directory, 'skip.md'), 'skip');
  }
  await writeFile(join(input, 'skip.pdf'), 'skip');
  await writeFile(join(input, 'nested', 'notes.MD'), '\uFEFFACME 😀 Q2.\r\nRevenue grew by 3%.\r');
  const calls: string[][] = [];
  const result = await runIngestion(input, {
    outputDir: join(root, 'output'),
    chunkSize: 15,
    overlap: 2,
    encoder,
    contextualize: (whole, part) => {
      calls.push([whole, part]);
      return Promise.resolve(' ACME financial results ');
    },
  });
  assert.equal((await findDocuments(input)).length, 1);
  const saved = await loadChunks(join(result.outputDir, 'chunks.jsonl'));
  const vectors = await loadEmbeddings(join(result.outputDir, 'embeddings.jsonl'));
  assert.equal(saved.length, vectors.length);
  assert.equal(new Set(saved.map((c) => c.chunk_id)).size, saved.length);
  for (const chunk of saved) {
    assert.equal(chunk.context, 'ACME financial results');
    assert.equal(
      chunk.text,
      characterSlice('ACME 😀 Q2.\nRevenue grew by 3%.\n', chunk.start_char, chunk.end_char),
    );
    assert.ok(vectors.some((v) => v.chunk_id === chunk.chunk_id));
  }
  assert.ok(calls.every(([whole]) => whole === 'ACME 😀 Q2.\nRevenue grew by 3%.\n'));
});
void test('empty and invalid UTF-8 documents are failures while valid documents continue', async (t) => {
  const root = await temporary(t);
  await writeFile(join(root, 'empty.md'), ' \n');
  await writeFile(join(root, 'invalid.txt'), Buffer.from([0xff]));
  await writeFile(join(root, 'good.md'), 'valid');
  const result = await runIngestion(root, { outputDir: join(root, 'output'), encoder });
  assert.equal(result.failures.length, 2);
  assert.equal(result.chunkCount, 1);
});
void test('empty input does not load an encoder; empty saved chunks stay supported', async (t) => {
  const root = await temporary(t);
  const result = await runIngestion(root, {
    outputDir: join(root, 'output'),
    encoder: neverEncoder,
  });
  assert.equal(result.chunkCount, 0);
  assert.deepEqual(await embedChunks([], { encoder: neverEncoder }), []);
});
void test('invalid input and chunk configurations do not write artifacts', async (t) => {
  const root = await temporary(t);
  for (const config of [
    { chunkSize: 0 },
    { overlap: -1 },
    { chunkSize: 5, overlap: 5 },
    { batchSize: 0 },
  ]) {
    await assert.rejects(
      runIngestion(root, { ...config, outputDir: join(root, 'output'), encoder: neverEncoder }),
    );
  }
  await assert.rejects(
    runIngestion(join(root, 'missing'), { outputDir: join(root, 'output'), encoder: neverEncoder }),
  );
  await writeFile(join(root, 'unsupported.pdf'), 'x');
  await assert.rejects(
    runIngestion(join(root, 'unsupported.pdf'), {
      outputDir: join(root, 'output'),
      encoder: neverEncoder,
    }),
  );
  await assert.rejects(access(join(root, 'output')));
});
void test('context and embedding failures preserve existing artifacts', async (t) => {
  const root = await temporary(t);
  await writeFile(join(root, 'doc.md'), 'original text');
  const result = await runIngestion(join(root, 'doc.md'), {
    outputDir: join(root, 'output'),
    encoder,
  });
  const before = await Promise.all(
    ['chunks', 'embeddings', 'failures'].map((name) =>
      readFile(join(result.outputDir, `${name}.jsonl`), 'utf8'),
    ),
  );
  await assert.rejects(
    runIngestion(join(root, 'doc.md'), { outputDir: result.outputDir, encoder: neverEncoder }),
  );
  await assert.rejects(
    runIngestion(join(root, 'doc.md'), {
      outputDir: result.outputDir,
      encoder,
      contextualize: () => Promise.resolve(' '),
    }),
  );
  assert.deepEqual(
    await Promise.all(
      ['chunks', 'embeddings', 'failures'].map((name) =>
        readFile(join(result.outputDir, `${name}.jsonl`), 'utf8'),
      ),
    ),
    before,
  );
});
void test('invalid vectors, dimensions and counts are rejected', async () => {
  for (const vectors of [[], [[0, 0]], [[NaN, 0]], [[Infinity]], [[]], [[1], [1, 2]]]) {
    await assert.rejects(
      embedChunks(chunks('one', 'two'), { encoder: { encode: () => Promise.resolve(vectors) } }),
    );
  }
});
void test('serialization failure preserves a destination', async (t) => {
  const root = await temporary(t);
  const path = join(root, 'data.jsonl');
  await writeFile(path, 'previous');
  await assert.rejects(saveJsonl([{ value: NaN }], path));
  assert.equal(await readFile(path, 'utf8'), 'previous');
});
