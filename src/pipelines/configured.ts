import { embeddingKey } from '../domain/config.ts';
import type { AskConfig, EmbeddingConfig, IngestionConfig } from '../domain/config.ts';
import type { Corpus, Document } from '../domain/models.ts';
import type { HttpClient } from '../domain/ports.ts';
import { remoteEncoder } from '../embeddings/remote.ts';
import { Contextualizer } from '../ingestion/contextualize.ts';
import { runIngestion } from './ingestion.ts';
import { runRag } from './answer.ts';

export function embeddingOptions(config: EmbeddingConfig, http: HttpClient = fetch) {
  return {
    model: embeddingKey(config),
    batchSize: config.batchSize,
    encoder: remoteEncoder(http, config.timeout),
  };
}
export function retrievalOptions(config: AskConfig, http: HttpClient = fetch) {
  return {
    method: config.method,
    lexicalMethod: config.lexicalMethod,
    topK: config.topK,
    candidateK: config.candidateK,
    rrfConstant: config.rrfConstant,
    timeout: config.timeout,
    encoder: remoteEncoder(http, config.embedding.timeout),
    http,
    ...(config.rerank.enabled
      ? {
          rerankModel: config.rerank.model,
          rerankProvider: config.rerank.provider,
        }
      : {}),
  };
}
export async function ingestConfigured(
  document: Document,
  config: IngestionConfig,
  http: HttpClient = fetch,
) {
  const contextualizer = config.context.enabled
    ? new Contextualizer({
        provider: config.context.provider,
        model: config.context.model,
        maxTokens: config.context.maxTokens,
        timeout: config.context.timeout,
        cacheDocument: config.context.cacheDocument,
        ...(config.context.instruction.trim() ? { instruction: config.context.instruction } : {}),
        http,
      })
    : undefined;
  const corpus = await runIngestion([document], {
    ...embeddingOptions(config.embedding, http),
    chunkSize: config.chunkSize,
    overlap: config.overlap,
    boundary: config.boundary,
    ...(contextualizer
      ? {
          contextualize: (whole: string, chunk: string) =>
            contextualizer.contextualize(whole, chunk),
        }
      : {}),
  });
  return { corpus, contextUsage: contextualizer?.usage ?? null };
}
export function askConfigured(
  question: string,
  corpus: Corpus,
  config: AskConfig,
  http: HttpClient = fetch,
) {
  return runRag(question, corpus, {
    ...retrievalOptions(config, http),
    provider: config.generation.provider,
    model: config.generation.model,
    maxTokens: config.generation.maxTokens,
    fullContext: config.fullContext,
    fullContextMaxChars: config.fullContextMaxChars,
  });
}
