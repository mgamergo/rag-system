import { writeFile } from 'node:fs/promises';
import { z } from 'zod';
import { rerankProviderSchema, retrievalMethodSchema } from '../domain/models.js';
import { evaluate, evaluationCaseSchema } from '../evaluation/retrieval.js';
import { loadJsonl } from '../storage/jsonl.js';
import { choice, command, execute, integer } from './shared.js';

const schema = z.object({
  artifactDir: z.string(),
  methods: z.array(retrievalMethodSchema),
  topK: z.array(z.number()),
  candidateK: z.number(),
  rerankModel: z.string().optional(),
  rerankProvider: rerankProviderSchema,
  device: z.string().optional(),
  output: z.string().optional(),
});
const cli = command('evaluate', 'Measure retrieval quality against labeled JSONL questions.')
  .argument('<dataset>')
  .requiredOption('--artifact-dir <path>')
  .option(
    '--methods <methods...>',
    'Retrieval methods',
    (value: string, previous: string[] | undefined) => [
      ...(previous ?? []),
      choice(retrievalMethodSchema)(value),
    ],
  )
  .option(
    '--top-k <counts...>',
    'Source counts',
    (value: string, previous: number[] | undefined) => [...(previous ?? []), integer(value)],
  )
  .option('--candidate-k <count>', 'Candidate count', integer, 150)
  .option('--rerank-model <id>')
  .option(
    '--rerank-provider <provider>',
    'Reranking provider',
    choice(rerankProviderSchema),
    'local',
  )
  .option('--device <device>')
  .option('--output <path>');
cli.action(async (dataset: string) =>
  execute('Evaluation', async () => {
    const options = schema.parse({
      methods: ['semantic', 'bm25', 'tfidf', 'hybrid'],
      topK: [5, 10, 20],
      ...cli.opts(),
    });
    const cases = await loadJsonl(dataset, evaluationCaseSchema);
    const report = [];
    for (const method of options.methods)
      for (const topK of options.topK)
        report.push(
          await evaluate(options.artifactDir, cases, {
            method,
            topK,
            candidateK: options.candidateK,
            device: options.device,
            rerankProvider: options.rerankProvider,
            ...(options.rerankModel ? { rerankModel: options.rerankModel } : {}),
          }),
        );
    const text = JSON.stringify(report, null, 2) + '\n';
    if (options.output) await writeFile(options.output, text, 'utf8');
    process.stdout.write(text);
  }),
);
await cli.parseAsync();
