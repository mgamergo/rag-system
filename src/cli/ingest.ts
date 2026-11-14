import { readFile } from 'node:fs/promises';
import { z } from 'zod';
import { boundarySchema } from '../domain/models.js';
import { DEFAULT_EMBEDDING_MODEL } from '../embeddings/local.js';
import { Contextualizer } from '../ingestion/contextualize.js';
import { runIngestion } from '../pipelines/ingestion.js';
import { choice, command, execute, integer, number } from './shared.js';

const optionsSchema = z.object({
  outputDir: z.string().optional(),
  chunkSize: z.number(),
  overlap: z.number(),
  model: z.string(),
  batchSize: z.number(),
  device: z.string().optional(),
  chunkBoundary: boundarySchema,
  contextModel: z.string().optional(),
  contextPrompt: z.string().optional(),
  contextMaxTokens: z.number(),
  contextTimeout: z.number(),
  contextCache: z.boolean(),
});
const cli = command('ingest', 'Discover, chunk, contextualize, embed, and save local documents.')
  .argument('<document-path>')
  .option('--output-dir <path>')
  .option('--chunk-size <characters>', 'Chunk size', integer, 500)
  .option('--overlap <characters>', 'Overlap', integer, 50)
  .option(
    '--model <id>',
    'Local ONNX model or gemini:MODEL / voyage:MODEL',
    DEFAULT_EMBEDDING_MODEL,
  )
  .option('--batch-size <count>', 'Embedding batch size', integer, 32)
  .option('--device <device>')
  .option(
    '--chunk-boundary <boundary>',
    'character, sentence, or paragraph',
    choice(boundarySchema),
    'character',
  )
  .option('--context-model <id>')
  .option('--context-prompt <path>')
  .option('--context-max-tokens <count>', 'Context output limit', integer, 160)
  .option('--context-timeout <seconds>', 'Context request timeout', number, 120)
  .option('--no-context-cache', 'Disable document-prefix caching');
cli.action(async (documentPath: string) =>
  execute('Ingestion', async () => {
    const options = optionsSchema.parse(cli.opts());
    if (options.contextPrompt && !options.contextModel)
      throw new Error('--context-prompt requires --context-model');
    const contextualizer = options.contextModel
      ? new Contextualizer({
          model: options.contextModel,
          maxTokens: options.contextMaxTokens,
          timeout: options.contextTimeout,
          cacheDocument: options.contextCache,
          ...(options.contextPrompt
            ? { instruction: await readFile(options.contextPrompt, 'utf8') }
            : {}),
        })
      : undefined;
    const result = await runIngestion(documentPath, {
      chunkSize: options.chunkSize,
      overlap: options.overlap,
      model: options.model,
      batchSize: options.batchSize,
      device: options.device,
      boundary: options.chunkBoundary,
      ...(options.outputDir === undefined ? {} : { outputDir: options.outputDir }),
      ...(contextualizer
        ? { contextualize: (document, chunk) => contextualizer.contextualize(document, chunk) }
        : {}),
    });
    process.stdout.write(
      `Chunks saved: ${result.chunkCount}\nVectors saved: ${result.embeddingCount}\nDocument failures: ${result.failures.length}\nOutput directory: ${result.outputDir}\n`,
    );
    if (contextualizer)
      process.stdout.write(`Context token usage: ${JSON.stringify(contextualizer.usage)}\n`);
    for (const name of ['chunks.jsonl', 'embeddings.jsonl', 'failures.jsonl'])
      process.stdout.write(`  ${result.outputDir}/${name}\n`);
    for (const failure of result.failures)
      process.stderr.write(`Skipped ${failure.source}: ${failure.error}\n`);
    if (!result.chunkCount)
      process.stderr.write(
        'No non-empty supported documents were found; no vectors were generated.\n',
      );
    if (result.failures.length || !result.chunkCount) process.exitCode = 1;
  }),
);
await cli.parseAsync();
