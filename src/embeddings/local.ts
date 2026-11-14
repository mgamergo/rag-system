import type { Encoder, PairScorer } from '../domain/ports.js';
import { z } from 'zod';

export const DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2';
export const DEFAULT_RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2';

// Resolve only known equivalents; custom IDs and local ONNX directories pass through.
function onnxModel(model: string): string {
  if (model === DEFAULT_EMBEDDING_MODEL) return 'Xenova/all-MiniLM-L6-v2';
  if (model === DEFAULT_RERANK_MODEL) return 'Xenova/ms-marco-MiniLM-L-6-v2';
  return model;
}

function deviceOptions(device: string | undefined): { device?: 'cpu' | 'cuda' | 'webgpu' } {
  if (device === undefined) return {};
  return { device: z.enum(['cpu', 'cuda', 'webgpu']).parse(device) };
}

export const localEncoder: Encoder = {
  async encode(texts, options) {
    // Lazy import keeps help, lexical retrieval, and empty ingestion model-free.
    const { pipeline } = await import('@huggingface/transformers');
    const extractor = await pipeline('feature-extraction', onnxModel(options.model), {
      ...deviceOptions(options.device),
      dtype: 'fp32',
    });
    try {
      const vectors: number[][] = [];
      for (let offset = 0; offset < texts.length; offset += options.batchSize) {
        const tensor = await extractor(texts.slice(offset, offset + options.batchSize), {
          pooling: 'mean',
          normalize: true,
        });
        vectors.push(...z.array(z.array(z.number())).parse(tensor.tolist()));
      }
      return vectors;
    } finally {
      await extractor.dispose();
    }
  },
};

export const localPairScorer: PairScorer = {
  async score(question, texts, options) {
    const { AutoTokenizer, AutoModelForSequenceClassification, Tensor } =
      await import('@huggingface/transformers');
    const name = onnxModel(options.model);
    const tokenizer = await AutoTokenizer.from_pretrained(name);
    const model = await AutoModelForSequenceClassification.from_pretrained(name, {
      ...deviceOptions(options.device),
      dtype: 'fp32',
    });
    try {
      const scores: number[] = [];
      for (let offset = 0; offset < texts.length; offset += options.batchSize) {
        const batch = texts.slice(offset, offset + options.batchSize);
        const inputs = tokenizer(
          batch.map(() => question),
          { text_pair: batch, padding: true, truncation: true },
        );
        const output: unknown = await model(inputs);
        const { logits: tensor } = z.object({ logits: z.instanceof(Tensor) }).parse(output);
        const logits = z.array(z.tuple([z.number()])).parse(tensor.tolist());
        scores.push(...logits.map(([score]) => 1 / (1 + Math.exp(-score))));
      }
      return scores;
    } finally {
      await model.dispose();
    }
  },
};
