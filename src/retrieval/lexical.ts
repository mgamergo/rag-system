import type { Chunk } from '../domain/models.js';
import { retrievalText } from '../domain/models.js';

export function tokenize(text: string): string[] {
  return (
    text
      .toLowerCase()
      .replace(/ß/gu, 'ss')
      .replace(/ς/gu, 'σ')
      .match(/[\p{L}\p{N}_]+(?:[-.][\p{L}\p{N}_]+)*/gu) ?? []
  );
}
function counts(tokens: string[]): Map<string, number> {
  const result = new Map<string, number>();
  for (const token of tokens) result.set(token, (result.get(token) ?? 0) + 1);
  return result;
}

export class LexicalIndex {
  private readonly counts: Map<string, number>[];
  private readonly df = new Map<string, number>();
  private readonly averageLength: number;
  constructor(chunks: Chunk[]) {
    this.counts = chunks.map((chunk) => counts(tokenize(retrievalText(chunk))));
    for (const count of this.counts)
      for (const term of count.keys()) this.df.set(term, (this.df.get(term) ?? 0) + 1);
    this.averageLength =
      this.counts.reduce((sum, count) => sum + [...count.values()].reduce((a, b) => a + b, 0), 0) /
      Math.max(chunks.length, 1);
  }

  tfidf(question: string): number[] {
    const vector = (count: Map<string, number>): Map<string, number> =>
      new Map(
        [...count].flatMap(([term, tf]) => {
          const df = this.df.get(term);
          return df === undefined
            ? []
            : [[term, (1 + Math.log(tf)) * (1 + Math.log((1 + this.counts.length) / (1 + df)))]];
        }),
      );
    const query = vector(counts(tokenize(question)));
    const queryNorm = Math.hypot(...query.values());
    return this.counts.map((count) => {
      const document = vector(count);
      const denominator = queryNorm * Math.hypot(...document.values());
      return denominator
        ? [...query].reduce((sum, [term, weight]) => sum + weight * (document.get(term) ?? 0), 0) /
            denominator
        : 0;
    });
  }

  bm25(question: string, options: { k1?: number; b?: number } = {}): number[] {
    const { k1 = 1.5, b = 0.75 } = options;
    if (!Number.isFinite(k1) || k1 <= 0 || !Number.isFinite(b) || b < 0 || b > 1)
      throw new Error('BM25 requires finite k1 > 0 and 0 <= b <= 1');
    return this.counts.map((count) => {
      const length = [...count.values()].reduce((a, value) => a + value, 0);
      let score = 0;
      for (const term of new Set(tokenize(question))) {
        const frequency = count.get(term) ?? 0;
        const df = this.df.get(term) ?? 0;
        if (frequency) {
          const idf = Math.log1p((this.counts.length - df + 0.5) / (df + 0.5));
          const norm = 1 - b + (b * length) / (this.averageLength || 1);
          score += (idf * frequency * (k1 + 1)) / (frequency + k1 * norm);
        }
      }
      return score;
    });
  }
}
