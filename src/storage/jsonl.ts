import type { z } from 'zod';
export function parseJsonl<T>(text: string, schema: z.ZodType<T>): T[] {
  return text.split(/\r?\n/u).flatMap((line, index) => {
    if (!line.trim()) return [];
    try {
      return [schema.parse(JSON.parse(line))];
    } catch {
      throw new Error(`Invalid JSONL record on line ${index + 1}`);
    }
  });
}
export function serializeJsonl(records: readonly unknown[]): string {
  return records
    .map(
      (record) =>
        JSON.stringify(record, (_key: string, value: unknown) => {
          if (typeof value === 'number' && !Number.isFinite(value))
            throw new Error('Cannot serialize non-finite numbers');
          return value;
        }) + '\n',
    )
    .join('');
}
