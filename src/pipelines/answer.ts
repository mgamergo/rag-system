import { positiveInteger } from '../domain/models.ts';
import type { Corpus, RetrievedChunk } from '../domain/models.ts';
import { ChatGenerator, UNKNOWN_ANSWER } from '../generation/chat.ts';
import type { GenerationOptions } from '../generation/chat.ts';
import type { PairScorer } from '../domain/ports.ts';
import { fullCorpus } from '../retrieval/full-context.ts';
import { retrieve } from '../retrieval/search.ts';
import type { RetrievalOptions } from '../retrieval/search.ts';
import { rerank } from '../retrieval/reranking.ts';
import type { RerankOptions } from '../retrieval/reranking.ts';

export interface AnswerOptions extends GenerationOptions, RetrievalOptions {
  rerankModel?: string | undefined;
  rerankProvider?: RerankOptions['provider'];
  scorer?: PairScorer;
  fullContext?: boolean;
  fullContextMaxChars?: number;
}
export interface RAGResult {
  question: string;
  answer: string;
  sources: RetrievedChunk[];
  retrievalLatencyMs: number;
  generationLatencyMs: number;
  usage: {
    prompt_tokens?: number | undefined;
    completion_tokens?: number | undefined;
    total_tokens?: number | undefined;
  };
}
export async function runRag(
  question: string,
  corpus: Corpus,
  options: AnswerOptions,
): Promise<RAGResult> {
  const generator = new ChatGenerator(options);
  const topK = positiveInteger.parse(options.topK ?? 20);
  const candidateK = positiveInteger.parse(options.candidateK ?? 150);
  if (candidateK < topK) throw new Error('candidate_k must be >= positive top_k');
  if (options.fullContext && options.rerankModel)
    throw new Error('Full-context mode cannot be combined with reranking');
  const started = performance.now();
  let sources = options.fullContext
    ? fullCorpus(corpus.chunks, options.fullContextMaxChars)
    : await retrieve(question, corpus, {
        ...options,
        topK: options.rerankModel ? candidateK : topK,
      });
  if (options.rerankModel)
    sources = await rerank(question, sources, {
      ...options,
      model: options.rerankModel,
      provider: options.rerankProvider ?? 'cohere',
      topK,
    });
  const retrievalLatencyMs = performance.now() - started;
  const generationStart = performance.now();
  const answer = sources.length ? await generator.generate(question, sources) : UNKNOWN_ANSWER;
  return {
    question,
    answer,
    sources,
    retrievalLatencyMs,
    generationLatencyMs: performance.now() - generationStart,
    usage: generator.usage,
  };
}
