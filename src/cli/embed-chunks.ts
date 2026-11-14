import { z } from 'zod';
import { embedChunks } from '../embeddings/encoder.js';
import { loadChunks, saveEmbeddings } from '../storage/jsonl.js';
import { command, execute } from './shared.js';

const cli = command('embed-chunks', 'Embed a saved chunks JSONL file.')
  .argument('<chunks-path>')
  .option('--output <path>', 'Output JSONL', 'artifacts/embeddings.jsonl');
cli.action(async (path: string) =>
  execute('Embedding', async () => {
    const options = z.object({ output: z.string() }).parse(cli.opts());
    const records = await embedChunks(await loadChunks(path));
    await saveEmbeddings(records, options.output);
    process.stdout.write(`Saved ${records.length} embeddings to ${options.output}\n`);
  }),
);
await cli.parseAsync();
