import { embedQuery } from '../src/embeddings/encoder.js';

const sentences = [
  'TypeScript supports asynchronous programming.',
  'The weather is sunny today.',
  'Async and await allow asynchronous code in TypeScript.',
];
const embeddings = [];
for (const sentence of sentences) embeddings.push(await embedQuery(sentence));
process.stdout.write(
  `Shape: (${embeddings.length}, ${embeddings[0]?.length ?? 0})\nDimensions: ${embeddings[0]?.length ?? 0}\n`,
);
