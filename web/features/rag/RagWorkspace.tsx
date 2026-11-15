'use client';

import Link from 'next/link';

import { useAction, useQuery } from 'convex/react';

import { useState } from 'react';

import { api } from '../../convex/_generated/api';

import type { Id } from '../../convex/_generated/dataModel';

import {
  DEFAULT_ASK,
  DEFAULT_INGESTION,
  askConfigSchema,
  embeddingConfigSchema,
  embeddingKey,
  ingestionConfigSchema,
} from '../../../src/domain/config';

import { decodeDocument } from '../../../src/ingestion/documents';

import { EmbeddingControls, IngestionControls, NumberField, RetrievalControls } from './Controls';

import type { FunctionReturnType } from 'convex/server';

const errorText = (e: unknown) => (e instanceof Error ? e.message : 'Request failed');

export function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function RagWorkspace() {
  const docs = useQuery(api.documents.list) ?? [];
  const runs = useQuery(api.documents.runs) ?? [];
  const traces = useQuery(api.documents.traces) ?? [];

  const ingest = useAction(api.rag.ingest);
  const ask = useAction(api.rag.ask);
  const reembed = useAction(api.rag.reembed);
  const exportDoc = useAction(api.rag.exportDocument);

  const [ingestion, setIngestion] = useState(DEFAULT_INGESTION);
  const [config, setConfig] = useState(DEFAULT_ASK);

  const [files, setFiles] = useState<File[]>([]);
  const [runName, setRunName] = useState('');
  const [selected, setSelected] = useState<Id<'documents'>[]>([]);

  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<FunctionReturnType<typeof api.rag.ask> | null>(null);

  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState('');

  async function perform(label: string, fn: () => Promise<void>) {
    setBusy(label);
    setNotice('');
    try {
      await fn();
    } catch (e) {
      setNotice(errorText(e));
    } finally {
      setBusy('');
    }
  }

  const ready = docs.filter((d) => d.status === 'ready' && d.activeRun);

  return (
    <main className="shell">
      <header className="masthead">
        <div>
          <h1>Field Notes</h1>
          <p>Retrieval workbench</p>
        </div>
        <nav>
          <Link href="/">Workspace</Link>
          <Link href="/evals">Evaluations</Link>
        </nav>
      </header>
      <p>
        Index documents, inspect retrieval evidence and compare configurations. All runs are saved
        in Convex.
      </p>
      {notice && (
        <p role="status" className="notice">
          {notice}
        </p>
      )}

      <div className="workbench">
        <aside>
          <h2>Library</h2>
          <p className="hint">
            No boxes selected means all ready documents. Semantic and hybrid search use only the
            selected embedding space.
          </p>
          {docs.map((d) => (
            <div key={d._id} className="document">
              <label className="toggle">
                <input
                  type="checkbox"
                  disabled={!d.activeRun || d.status !== 'ready'}
                  checked={selected.includes(d._id)}
                  onChange={(e) =>
                    setSelected(
                      e.target.checked
                        ? [...selected, d._id]
                        : selected.filter((id) => id !== d._id),
                    )
                  }
                />
                {d.filename}
              </label>
              <small>
                {d.status} · {d.chunkCount} chunks
              </small>
              {!d.activeRun && d.status === 'ready' && (
                <p className="hint">Legacy document: upload again to create a manifest.</p>
              )}
              {d.error && <p className="error">{d.error}</p>}
              {d.activeRun && (
                <>
                  <small>{d.embeddingModel}</small>
                  <details>
                    <summary>Manifest and export</summary>
                    <pre>{d.configJson}</pre>
                    <p>Active embedding run: {d.activeRun}</p>
                    <button
                      type="button"
                      disabled={!!busy}
                      onClick={() =>
                        void perform('Exporting', async () => {
                          const result = await exportDoc({ documentId: d._id });
                          download(`${d.filename}.chunks.jsonl`, result.chunks);
                          download(`${d.filename}.embeddings.jsonl`, result.embeddings);
                        })
                      }
                    >
                      Download chunks and vectors
                    </button>
                  </details>
                  <button
                    type="button"
                    disabled={!!busy || !!d.pendingRun}
                    onClick={() =>
                      void perform('Re-embedding', async () => {
                        await reembed({
                          documentId: d._id,
                          embedding: config.embedding,
                          runName: runName || 'Re-embedding',
                        });
                        setNotice('Re-embedding saved. Chunk IDs and context were preserved.');
                      })
                    }
                  >
                    Re-embed with selected embedding settings
                  </button>
                </>
              )}
            </div>
          ))}
        </aside>

        <div>
          <section>
            <h2>Ingest documents</h2>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void perform('Indexing', async () => {
                  const normalized = ingestionConfigSchema.parse(ingestion);
                  const uploads = [];
                  const failures: string[] = [];
                  for (const file of files) {
                    try {
                      uploads.push({
                        filename: file.name,
                        content: decodeDocument(await file.arrayBuffer()),
                      });
                    } catch (error) {
                      failures.push(`${file.name}: ${errorText(error)}`);
                    }
                  }
                  const result = uploads.length
                    ? await ingest({ files: uploads, runName, config: normalized })
                    : [];
                  setNotice(
                    [
                      ...failures,
                      ...result.map(
                        (r) =>
                          `${r.filename}: ${r.error ?? (r.reused ? 'existing run reused' : 'indexed')}`,
                      ),
                    ].join('\n'),
                  );
                });
              }}
            >
              <div className="fields">
                <label>
                  Documents (.md or .txt)
                  <input
                    required
                    type="file"
                    multiple
                    accept=".md,.txt"
                    onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
                  />
                </label>
                <label>
                  Run name (optional)
                  <input value={runName} onChange={(e) => setRunName(e.target.value)} />
                </label>
              </div>
              <IngestionControls value={ingestion} onChange={setIngestion} />
              <button disabled={!!busy || !files.length}>
                {busy === 'Indexing' ? 'Indexing…' : 'Index selected files'}
              </button>
              <p className="hint">
                Up to 8 files per request; 500,000 code points each. Hosted providers may charge for
                indexing and answering.
              </p>
            </form>
          </section>

          <section>
            <h2>Ask the collection</h2>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void perform('Answering', async () => {
                  setAnswer(null);
                  setAnswer(
                    await ask({
                      question,
                      documentIds: selected,
                      config: askConfigSchema.parse(config),
                    }),
                  );
                });
              }}
            >
              <EmbeddingControls
                value={config.embedding}
                onChange={(embedding) => setConfig({ ...config, embedding })}
              />
              <label>
                Available indexed embedding spaces
                <select
                  value={embeddingKey(config.embedding)}
                  onChange={(e) => {
                    const doc = ready.find((d) => d.embeddingModel === e.target.value);
                    if (doc?.embeddingModel) {
                      const split = doc.embeddingModel.indexOf(':');
                      setConfig({
                        ...config,
                        embedding: {
                          ...config.embedding,
                          provider: embeddingConfigSchema.shape.provider.parse(
                            doc.embeddingModel.slice(0, split),
                          ),
                          model: doc.embeddingModel.slice(split + 1),
                        },
                      });
                    }
                  }}
                >
                  <option value={embeddingKey(config.embedding)}>
                    {embeddingKey(config.embedding)}
                  </option>
                  {Array.from(new Set(ready.map((d) => d.embeddingModel)))
                    .filter((m) => m && m !== embeddingKey(config.embedding))
                    .map((m) => (
                      <option key={m}>{m}</option>
                    ))}
                </select>
              </label>
              <RetrievalControls value={config} onChange={setConfig} />
              <fieldset>
                <legend>Answer generation</legend>
                <div className="fields">
                  <label>
                    Provider
                    <select value="openrouter" disabled>
                      <option>openrouter</option>
                    </select>
                  </label>
                  <label>
                    Generation model
                    <input
                      required
                      value={config.generation.model}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          generation: { ...config.generation, model: e.target.value },
                        })
                      }
                    />
                  </label>
                  <NumberField
                    label="Answer output tokens"
                    value={config.generation.maxTokens}
                    onChange={(maxTokens) =>
                      setConfig({ ...config, generation: { ...config.generation, maxTokens } })
                    }
                  />
                </div>
              </fieldset>
              <label>
                Question
                <textarea
                  required
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  rows={3}
                />
              </label>
              <button disabled={!!busy || !ready.length}>
                {busy === 'Answering' ? 'Answering…' : 'Ask'}
              </button>
            </form>

            {answer && (
              <article>
                <h2>Answer</h2>
                <p className="answer-text">{answer.answer}</p>
                <p>
                  {answer.latencyMs} ms · Trace {answer.traceId} ·{' '}
                  {answer.usage.total_tokens ?? 'unknown'} generation tokens
                </p>
                <h3>Evidence sent to the model</h3>
                {answer.sources.map((s, i) => (
                  <details key={s.chunk_id}>
                    <summary>
                      [{i + 1}] {s.source} · chunk {s.chunk_index} · code points {s.start_char}:
                      {s.end_char} · score {answer.scores[i]?.toFixed(4)}
                    </summary>
                    <small>Chunk ID: {s.chunk_id}</small>
                    {s.context && (
                      <blockquote>
                        <strong>Generated retrieval context</strong>
                        <p>{s.context}</p>
                      </blockquote>
                    )}
                    <p className="source-text">{s.text}</p>
                  </details>
                ))}
              </article>
            )}
          </section>
        </div>
      </div>

      <section>
        <h2>Indexing runs</h2>
        {runs.map((r) => (
          <details key={r._id}>
            <summary>
              {r.name} · {r.kind} · {r.status}
            </summary>
            <pre>{r.configJson}</pre>
            {r.usageJson && <pre>{r.usageJson}</pre>}
            {r.error && <p className="error">{r.error}</p>}
          </details>
        ))}
      </section>
      <section>
        <h2>Recent question traces</h2>
        {traces.map((t) => (
          <details key={t._id}>
            <summary>
              {t.status ?? 'legacy success'} · {t.question} · {t.totalLatencyMs} ms
            </summary>
            <p>{t.answer || t.error}</p>
            <pre>{t.configJson}</pre>
          </details>
        ))}
      </section>
    </main>
  );
}
