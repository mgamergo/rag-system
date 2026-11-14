import type { HttpClient } from '../domain/ports.js';
import { positiveNumber } from '../domain/models.js';

export class HttpError extends Error {
  constructor(
    public readonly provider: string,
    public readonly status: number,
  ) {
    super(`${provider} returned HTTP ${status}`);
  }
}

/** Never surface response bodies or request headers, which may contain credentials. */
export async function postJson(options: {
  url: string;
  provider: string;
  payload: unknown;
  headers?: Record<string, string>;
  timeout?: number;
  http?: HttpClient;
}): Promise<unknown> {
  const timeout = positiveNumber.parse(options.timeout ?? 120);
  try {
    const response = await (options.http ?? fetch)(options.url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...options.headers },
      body: JSON.stringify(options.payload),
      signal: AbortSignal.timeout(Math.ceil(timeout * 1000)),
    });
    if (!response.ok) {
      await response.body?.cancel();
      throw new HttpError(options.provider, response.status);
    }
    return await response.json();
  } catch (error) {
    if (error instanceof HttpError) throw error;
    throw new Error(`${options.provider} request failed, timed out, or returned invalid JSON`);
  }
}

export function apiKey(name: string, override?: string): string {
  const key = (override ?? process.env[name] ?? '').trim();
  if (!key) throw new Error(`Set ${name} before using this provider`);
  return key;
}
