import { nonBlank } from '../domain/models.ts';
export function normalizeDocument(filename: string, content: string): string {
  if (!/\.(md|txt)$/iu.test(filename)) throw new Error('Only .md and .txt files are supported');
  return nonBlank.parse(content.replace(/^\uFEFF/u, '').replace(/\r\n?/gu, '\n'));
}
export function decodeDocument(bytes: ArrayBuffer): string {
  return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
}
