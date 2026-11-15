import { characterLength, characterSlice, positiveInteger } from '../domain/models.ts';
import type { Chunk, RetrievedChunk } from '../domain/models.ts';

export function fullCorpus(input: Chunk[], maxChars = 600_000): RetrievedChunk[] {
  positiveInteger.parse(maxChars);
  const documents = new Map<string, Chunk[]>();
  for (const chunk of input) {
    const group = documents.get(chunk.document_id) ?? [];
    group.push(chunk);
    documents.set(chunk.document_id, group);
  }
  if (!documents.size) throw new Error('The index is empty');
  const sources: RetrievedChunk[] = [];
  let total = 0;
  for (const [id, chunks] of documents) {
    chunks.sort((a, b) => a.start_char - b.start_char);
    const first = chunks[0];
    if (!first) continue;
    let text = '';
    for (const chunk of chunks) {
      const length = characterLength(text);
      if (
        chunk.source !== first.source ||
        chunk.start_char > length ||
        chunk.end_char !== chunk.start_char + characterLength(chunk.text)
      )
        throw new Error('Cannot reconstruct corpus: invalid offsets or missing chunks');
      const overlap = Math.min(length - chunk.start_char, characterLength(chunk.text));
      if (
        characterSlice(text, chunk.start_char, chunk.start_char + overlap) !==
        characterSlice(chunk.text, 0, overlap)
      )
        throw new Error('Cannot reconstruct corpus: inconsistent chunk overlap');
      text += characterSlice(chunk.text, overlap);
    }
    total += characterLength(text);
    if (total > maxChars)
      throw new Error(
        'Corpus exceeds full-context character budget; use retrieval or increase the budget',
      );
    sources.push({
      chunk: {
        chunk_id: id,
        document_id: id,
        source: first.source,
        chunk_index: 0,
        text,
        start_char: 0,
        end_char: characterLength(text),
        context: '',
      },
      score: 1,
    });
  }
  return sources;
}
