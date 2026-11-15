import type { TestContext } from 'node:test';
import type { Chunk, Corpus } from '../src/domain/models.ts';
import type { Encoder, HttpClient } from '../src/domain/ports.ts';
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
export function index(records = chunks('RAG retrieves evidence.', 'Cats sleep.')): Corpus {
  return {
    chunks: records,
    embeddings: records
      .map((chunk, i) => ({
        chunk_id: chunk.chunk_id,
        model: 'test-encoder',
        embedding: i === 0 ? [3, 0] : [0, 2],
      }))
      .reverse(),
  };
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
