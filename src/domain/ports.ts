export interface Encoder {
  encode(
    texts: string[],
    options: { model: string; batchSize: number; query: boolean },
  ): Promise<number[][]>;
}
export interface PairScorer {
  score(
    question: string,
    texts: string[],
    options: { model: string; batchSize: number },
  ): Promise<number[]>;
}
export type Contextualize = (document: string, chunk: string) => Promise<string>;
export type HttpClient = typeof fetch;
