import { boundarySchema, characters, positiveInteger } from '../domain/models.ts';
import type { ChunkBoundary, RawChunk } from '../domain/models.ts';

export function validateChunkConfig(chunkSize: number, overlap: number): void {
  positiveInteger.parse(chunkSize);
  if (!Number.isInteger(overlap) || overlap < 0 || overlap >= chunkSize)
    throw new Error('Overlap must be nonnegative and less than chunk size');
}

export function chunkText(
  text: string,
  options: { chunkSize: number; overlap: number; boundary?: ChunkBoundary },
): RawChunk[] {
  const { chunkSize, overlap, boundary = 'character' } = options;
  validateChunkConfig(chunkSize, overlap);
  boundarySchema.parse(boundary);
  const chars = characters(text);
  const chunks: RawChunk[] = [];
  let start = 0;
  while (start < chars.length) {
    let end = Math.min(start + chunkSize, chars.length);
    if (boundary !== 'character' && end < chars.length) {
      const window = chars.slice(start, end).join('');
      const pattern = boundary === 'paragraph' ? /\n\s*\n/gu : /[.!?](?:\s+|$)/gu;
      for (const match of window.matchAll(pattern)) {
        const offset = Array.from(window.slice(0, match.index + match[0].length)).length;
        if (offset > overlap && offset >= Math.floor(chunkSize / 2)) end = start + offset;
      }
    }
    chunks.push({ text: chars.slice(start, end).join(''), start_idx: start, end_idx: end });
    if (end === chars.length) break;
    start = end - overlap;
  }
  return chunks;
}
