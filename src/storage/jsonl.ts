import { readFile, mkdir, writeFile, rename, rm } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { homedir } from 'node:os';
import { randomUUID } from 'node:crypto';
import { z } from 'zod';
import { chunkSchema, embeddingSchema } from '../domain/models.js';
import type { Chunk, Embedding, IngestionFailure } from '../domain/models.js';

export function expandPath(path: string): string {
  return resolve(
    path === '~' ? homedir() : /^~[/\\]/u.test(path) ? join(homedir(), path.slice(2)) : path,
  );
}

export async function loadJsonl<T>(path: string, schema: z.ZodType<T>): Promise<T[]> {
  const text = await readFile(expandPath(path), 'utf8');
  return text.split(/\r?\n/u).flatMap((line, index) => {
    if (!line.trim()) return [];
    try {
      const value: unknown = JSON.parse(line);
      return [schema.parse(value)];
    } catch (error) {
      throw new Error(`Invalid ${path} record on line ${index + 1}`, { cause: error });
    }
  });
}

export async function saveJsonl(records: readonly unknown[], path: string): Promise<void> {
  // Serialize everything before touching the destination. JSON silently replaces
  // non-finite numbers with null unless the replacer explicitly rejects them.
  const text = records
    .map(
      (record) =>
        JSON.stringify(record, (_key: string, value: unknown) => {
          if (typeof value === 'number' && !Number.isFinite(value))
            throw new Error('Cannot serialize non-finite numbers');
          return value;
        }) + '\n',
    )
    .join('');
  const target = expandPath(path);
  await mkdir(dirname(target), { recursive: true });
  const temporary = `${target}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporary, text, { encoding: 'utf8', flag: 'wx' });
    await rename(temporary, target);
  } finally {
    await rm(temporary, { force: true });
  }
}

export const loadChunks = (path: string): Promise<Chunk[]> => loadJsonl(path, chunkSchema);
export const loadEmbeddings = (path: string): Promise<Embedding[]> =>
  loadJsonl(path, embeddingSchema);
export const saveChunks = (records: Chunk[], path: string): Promise<void> =>
  saveJsonl(
    records.map((record) => chunkSchema.parse(record)),
    path,
  );
export const saveEmbeddings = (records: Embedding[], path: string): Promise<void> =>
  saveJsonl(
    records.map((record) => embeddingSchema.parse(record)),
    path,
  );
export const saveFailures = (records: IngestionFailure[], path: string): Promise<void> =>
  saveJsonl(records, path);
