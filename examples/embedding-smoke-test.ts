// Offline contract smoke check. No model download or provider request.
import assert from 'node:assert/strict';
import { embedQuery } from '../src/embeddings/encoder.ts';
const vector = await embedQuery('Example question', {
  encoder: { encode: () => Promise.resolve([[1, 0]]) },
});
assert.deepEqual(vector, [1, 0]);
process.stdout.write('Embedding adapter contract passed (offline).\n');
