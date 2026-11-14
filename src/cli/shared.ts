import { Command, InvalidArgumentError } from 'commander';
import { z } from 'zod';

export function integer(value: string): number {
  if (!/^-?\d+$/u.test(value)) throw new InvalidArgumentError('Expected an integer');
  const result = Number(value);
  if (!Number.isSafeInteger(result)) throw new InvalidArgumentError('Expected a safe integer');
  return result;
}
export function number(value: string): number {
  const result = Number(value);
  if (!value.trim() || !Number.isFinite(result))
    throw new InvalidArgumentError('Expected a finite number');
  return result;
}
export function choice<T>(schema: z.ZodType<T>): (value: string) => T {
  return (value) => {
    const parsed = schema.safeParse(value);
    if (!parsed.success) throw new InvalidArgumentError(parsed.error.message);
    return parsed.data;
  };
}
export function command(name: string, description: string): Command {
  return new Command(name)
    .description(description)
    .configureOutput({ outputError: (message, write) => write(message) })
    .exitOverride((error) => {
      if (error.code === 'commander.helpDisplayed' || error.code === 'commander.version')
        process.exit(0);
      process.exit(2);
    });
}
export async function execute(label: string, action: () => Promise<void>): Promise<void> {
  try {
    await action();
  } catch (error) {
    process.stderr.write(
      `${label} failed: ${error instanceof Error ? error.message : String(error)}\n`,
    );
    process.exitCode = 1;
  }
}
