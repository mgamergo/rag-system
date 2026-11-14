import { z } from 'zod';
import type { Encoder, HttpClient } from '../domain/ports.js';
import { nonBlank, positiveInteger, vectorSchema } from '../domain/models.js';
import { apiKey, postJson } from '../infrastructure/http.js';

const voyageResponse = z.object({
  data: z.array(z.object({ index: z.number().int().nonnegative(), embedding: vectorSchema })),
});
const geminiResponse = z.object({ embeddings: z.array(z.object({ values: vectorSchema })) });

export function remoteEncoder(http: HttpClient = fetch): Encoder {
  return {
    async encode(texts, options) {
      const separator = options.model.indexOf(':');
      const provider = z.enum(['voyage', 'gemini']).parse(options.model.slice(0, separator));
      const model = nonBlank.parse(options.model.slice(separator + 1));
      const batchSize = Math.min(positiveInteger.parse(options.batchSize), 100);
      const key = apiKey(provider === 'gemini' ? 'GEMINI_API_KEY' : 'VOYAGE_API_KEY');
      const vectors: number[][] = [];
      for (let start = 0; start < texts.length; start += batchSize) {
        const batch = texts.slice(start, start + batchSize);
        if (provider === 'voyage') {
          const data = voyageResponse.parse(
            await postJson({
              url: 'https://api.voyageai.com/v1/embeddings',
              provider,
              http,
              headers: { Authorization: `Bearer ${key}` },
              payload: {
                model,
                input: batch,
                input_type: options.query ? 'query' : 'document',
                truncation: false,
              },
            }),
          );
          const records = data.data.sort((a, b) => a.index - b.index);
          if (
            records.length !== batch.length ||
            records.some((record, index) => record.index !== index)
          )
            throw new Error('Invalid embedding indexes or count');
          vectors.push(...records.map((record) => record.embedding));
        } else {
          const data = geminiResponse.parse(
            await postJson({
              url: `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:batchEmbedContents`,
              provider,
              http,
              headers: { 'x-goog-api-key': key },
              payload: {
                requests: batch.map((text) => ({
                  model: `models/${model}`,
                  content: { parts: [{ text }] },
                  taskType: options.query ? 'RETRIEVAL_QUERY' : 'RETRIEVAL_DOCUMENT',
                })),
              },
            }),
          );
          if (data.embeddings.length !== batch.length) throw new Error('Wrong embedding count');
          vectors.push(...data.embeddings.map((record) => record.values));
        }
      }
      return vectors;
    },
  };
}
