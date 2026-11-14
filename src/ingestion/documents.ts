import { readdir, stat, readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { expandPath } from '../storage/jsonl.js';
import type { Document, IngestionFailure } from '../domain/models.js';

const supported = new Set(['.md', '.txt']);
const excluded = new Set(['.git', 'node_modules', '.cache', '.venv', 'venv', '__pycache__']);

export async function findDocuments(input: string): Promise<string[]> {
  const root = expandPath(input);
  const info = await stat(root);
  if (info.isFile()) {
    if (!supported.has(extname(root).toLowerCase()))
      throw new Error('Unsupported document type. Supported: .md, .txt');
    return [root];
  }
  if (!info.isDirectory()) throw new Error(`Expected a directory: ${root}`);
  const paths: string[] = [];
  async function visit(directory: string): Promise<void> {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory() && !excluded.has(entry.name.toLowerCase())) await visit(path);
      else if (
        supported.has(extname(path).toLowerCase()) &&
        (entry.isFile() || entry.isSymbolicLink()) &&
        (await stat(path)).isFile()
      )
        paths.push(path);
    }
  }
  await visit(root);
  return paths.sort();
}

export async function parseTextFile(path: string): Promise<string> {
  return new TextDecoder('utf-8', { fatal: true })
    .decode(await readFile(path))
    .replace(/\r\n?/gu, '\n');
}

export async function ingestDocuments(
  root: string,
): Promise<{ documents: Document[]; failures: IngestionFailure[] }> {
  const paths = await findDocuments(root);
  const documents: Document[] = [];
  const failures: IngestionFailure[] = [];
  for (const source of paths) {
    try {
      const text = await parseTextFile(source);
      if (!text.trim()) failures.push({ source, error: 'Document is Empty' });
      else documents.push({ document_id: randomUUID(), source, text });
    } catch (error) {
      failures.push({ source, error: error instanceof Error ? error.message : String(error) });
    }
  }
  return { documents, failures };
}
