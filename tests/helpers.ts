import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { TestContext } from 'node:test';
import type { Chunk, Embedding } from '../src/domain/models.js';
import type { Encoder, HttpClient } from '../src/domain/ports.js';
import { saveChunks, saveEmbeddings } from '../src/storage/jsonl.js';

export async function temporary(t: TestContext): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'rag-test-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}
export function environment(t: TestContext, name: string, value: string): void {
  const previous = process.env[name];
  process.env[name] = value;
  t.after(() => {
    if (previous === undefined) Reflect.deleteProperty(process.env, name);
    else process.env[name] = previous;
  });
}
export function chunks(...texts: string[]): Chunk[] {
  return texts.map((text, index) => ({
    chunk_id: String(index),
    document_id: 'doc',
    source: 'notes.md',
    chunk_index: index,
    text,
    start_char: 0,
    end_char: Array.from(text).length,
    context: '',
  }));
}
export const encoder: Encoder = { encode: (texts) => Promise.resolve(texts.map(() => [1, 0])) };
export const neverEncoder: Encoder = {
  encode: () => {
    throw new Error('Model must not load');
  },
};
export async function index(
  root: string,
  records = chunks('RAG retrieves evidence.', 'Cats sleep.'),
): Promise<Chunk[]> {
  await saveChunks(records, join(root, 'chunks.jsonl'));
  const embeddings: Embedding[] = records.map((chunk, i) => ({
    chunk_id: chunk.chunk_id,
    model: 'test-encoder',
    embedding: i === 0 ? [3, 0] : [0, 2],
  }));
  await saveEmbeddings(embeddings.reverse(), join(root, 'embeddings.jsonl'));
  return records;
}
export function fakeHttp(
  response: unknown,
  calls: { url: string; headers: Headers; body: unknown }[] = [],
  status = 200,
): HttpClient {
  return (input, init) => {
    if (typeof init?.body !== 'string') throw new Error('Expected a JSON request');
    const body: unknown = JSON.parse(init.body);
    calls.push({
      url: typeof input === 'string' ? input : input instanceof URL ? input.href : input.url,
      headers: new Headers(init.headers),
      body,
    });
    return Promise.resolve(Response.json(response, { status }));
  };
}
