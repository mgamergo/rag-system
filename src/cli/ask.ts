import { z } from 'zod';
import { Option } from 'commander';
import {
  generationProviderSchema,
  lexicalMethodSchema,
  rerankProviderSchema,
  retrievalMethodSchema,
} from '../domain/models.js';
import { DEFAULT_RERANK_MODEL } from '../embeddings/local.js';
import { runRag } from '../pipelines/answer.js';
import { choice, command, execute, integer, number } from './shared.js';

const schema = z.object({
  artifactDir: z.string(),
  provider: generationProviderSchema,
  model: z.string(),
  topK: z.number(),
  device: z.string().optional(),
  retrievalMethod: retrievalMethodSchema,
  lexicalMethod: lexicalMethodSchema,
  candidateK: z.number(),
  rerankModel: z.string().optional(),
  rerankProvider: rerankProviderSchema,
  fullContext: z.boolean().default(false),
  fullContextMaxChars: z.number(),
  ollamaBaseUrl: z.string(),
  maxTokens: z.number(),
  timeout: z.number(),
});
const cli = command('ask', 'Answer a question using saved artifacts and OpenRouter or Ollama.')
  .argument('<question>')
  .requiredOption('--artifact-dir <path>')
  .option(
    '--provider <provider>',
    'ollama or openrouter',
    choice(generationProviderSchema),
    'ollama',
  )
  .requiredOption('--model <id>')
  .option('--top-k <count>', 'Final source count', integer, 20)
  .option('--device <device>')
  .option(
    '--retrieval-method <method>',
    'Retrieval method',
    choice(retrievalMethodSchema),
    'semantic',
  )
  .option('--lexical-method <method>', 'Hybrid lexical branch', choice(lexicalMethodSchema), 'bm25')
  .option('--candidate-k <count>', 'Candidates per branch', integer, 150)
  .addOption(new Option('--rerank-model [id]', 'Enable reranking').preset(DEFAULT_RERANK_MODEL))
  .option(
    '--rerank-provider <provider>',
    'Reranking provider',
    choice(rerankProviderSchema),
    'local',
  )
  .option('--full-context')
  .option('--full-context-max-chars <count>', 'Corpus character budget', integer, 600_000)
  .option('--ollama-base-url <url>', 'OpenAI-compatible server URL', 'http://localhost:11434/v1')
  .option('--max-tokens <count>', 'Answer output limit', integer, 512)
  .option('--timeout <seconds>', 'Request timeout', number, 120);
cli.action(async (question: string) =>
  execute('RAG', async () => {
    const options = schema.parse(cli.opts());
    const result = await runRag(question, options.artifactDir, {
      ...options,
      method: options.retrievalMethod,
    });
    process.stdout.write(`${result.answer}\n\nRetrieved sources:\n`);
    result.sources.forEach(({ chunk, score }, index) =>
      process.stdout.write(
        `[${index + 1}] ${chunk.source} | chunk ${chunk.chunk_index} | chars ${chunk.start_char}:${chunk.end_char} | score ${score.toFixed(4)}\n`,
      ),
    );
  }),
);
await cli.parseAsync();
