import { positiveInteger } from '../domain/models.js';
import type { RetrievedChunk } from '../domain/models.js';
import { ChatGenerator, UNKNOWN_ANSWER } from '../generation/chat.js';
import type { GenerationOptions } from '../generation/chat.js';
import type { PairScorer } from '../domain/ports.js';
import { fullCorpus } from '../retrieval/full-context.js';
import { retrieve } from '../retrieval/search.js';
import type { RetrievalOptions } from '../retrieval/search.js';
import { rerank } from '../retrieval/reranking.js';
import type { RerankOptions } from '../retrieval/reranking.js';

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
}
export async function runRag(
  question: string,
  artifactDir: string,
  options: AnswerOptions,
): Promise<RAGResult> {
  const generator = new ChatGenerator(options);
  const topK = positiveInteger.parse(options.topK ?? 20);
  const candidateK = positiveInteger.parse(options.candidateK ?? 150);
  if (candidateK < topK) throw new Error('candidate_k must be >= positive top_k');
  if (options.fullContext && options.rerankModel)
    throw new Error('Full-context mode cannot be combined with reranking');
  let sources = options.fullContext
    ? await fullCorpus(artifactDir, options.fullContextMaxChars)
    : await retrieve(question, artifactDir, {
        ...options,
        topK: options.rerankModel ? candidateK : topK,
      });
  if (options.rerankModel)
    sources = await rerank(question, sources, {
      ...options,
      model: options.rerankModel,
      provider: options.rerankProvider ?? 'local',
      topK,
    });
  return {
    question,
    answer: sources.length ? await generator.generate(question, sources) : UNKNOWN_ANSWER,
    sources,
  };
}
