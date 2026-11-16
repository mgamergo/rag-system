'use client';

import Link from 'next/link';

import { useState } from 'react';

import { useAction, useQuery } from 'convex/react';

import { api } from '../../convex/_generated/api';

import type { Id } from '../../convex/_generated/dataModel';

import { DEFAULT_ASK, evaluationConfigSchema } from '../../../src/domain/config';

import { savedReportSchema } from '../../../src/domain/reports';

import { retrievalMethodSchema } from '../../../src/domain/models';

import type { RetrievalMethod } from '../../../src/domain/models';

import { EmbeddingControls, NumberField, RetrievalControls } from './Controls';

import { download } from './RagWorkspace';

export function EvaluationWorkspace() {
  const projects = useQuery(api.projects.list) ?? [];
  const [selected, setSelected] = useState('');
  const project = projects.find((p) => p._id === selected);
  return (
    <>
      <div className="shell">
        <label>
          Evaluation project
          <select value={selected} onChange={(e) => setSelected(e.target.value)}>
            <option value="">Previous library (unassigned)</option>
            {projects.map((p) => (
              <option key={p._id} value={p._id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
      </div>
      <EvaluationProject key={selected} projectId={project?._id} />
    </>
  );
}
function EvaluationProject({ projectId }: { projectId?: Id<'projects'> }) {
  const documents = useQuery(api.documents.list, { projectId }) ?? [];
  const runs = useQuery(api.documents.evaluations, { projectId }) ?? [];
  const run = useAction(api.rag.runEvaluation);

  const [config, setConfig] = useState(DEFAULT_ASK);
  const [methods, setMethods] = useState<RetrievalMethod[]>([
    'semantic',
    'bm25',
    'tfidf',
    'hybrid',
  ]);

  const [ks, setKs] = useState([5, 10, 20]);
  const [documentIds, setDocumentIds] = useState<Id<'documents'>[]>([]);

  const [dataset, setDataset] = useState({ name: '', text: '' });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  return (
    <main className="shell">
      <header className="masthead">
        <div>
          <h1>Evaluations</h1>
          <p>Compare retrieval against labelled evidence</p>
        </div>
        <nav>
          <Link href="/">Workspace</Link>
          <Link href="/evals">Evaluations</Link>
        </nav>
      </header>
      <p>
        Each JSONL row needs a question and exactly one of relevant_chunk_ids or relevant_sources.
        Use chunk IDs from exported chunks or exact filenames. Evaluation never generates answers.
      </p>
      <pre>{'{"question":"What changed?","relevant_sources":["notes.md"]}'}</pre>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          setBusy(true);
          setError('');
          void (async () => {
            try {
              const parsed = evaluationConfigSchema.parse({
                retrieval: config,
                methods,
                ks,
              });
              await run({
                ...(projectId ? { projectId } : {}),
                datasetName: dataset.name,
                datasetJsonl: dataset.text,
                documentIds,
                config: parsed,
              });
            } catch (e) {
              setError(e instanceof Error ? e.message : 'Evaluation failed');
            } finally {
              setBusy(false);
            }
          })();
        }}
      >
        <label>
          Gold set (.jsonl)
          <input
            required
            type="file"
            accept=".jsonl"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file)
                void file
                  .text()
                  .then((text) => setDataset({ name: file.name, text }))
                  .catch(() => setError('Unable to read the dataset'));
            }}
          />
        </label>
        <fieldset>
          <legend>Document filter</legend>
          <p className="hint">Leave empty for all compatible ready documents.</p>
          {documents
            .filter((d) => d.activeRun && d.status === 'ready')
            .map((d) => (
              <label key={d._id} className="toggle">
                <input
                  type="checkbox"
                  checked={documentIds.includes(d._id)}
                  onChange={(e) =>
                    setDocumentIds(
                      e.target.checked
                        ? [...documentIds, d._id]
                        : documentIds.filter((id) => id !== d._id),
                    )
                  }
                />
                {d.filename} ({d.embeddingModel})
              </label>
            ))}
        </fieldset>
        <EmbeddingControls
          value={config.embedding}
          onChange={(embedding) => setConfig({ ...config, embedding })}
        />
        <fieldset>
          <legend>Configurations</legend>
          <div className="method-list">
            {retrievalMethodSchema.options.map((m) => (
              <label key={m} className="toggle">
                <input
                  type="checkbox"
                  checked={methods.includes(m)}
                  onChange={(e) =>
                    setMethods(e.target.checked ? [...methods, m] : methods.filter((x) => x !== m))
                  }
                />
                {m}
              </label>
            ))}
          </div>
          <div className="fields">
            {ks.map((k, i) => (
              <div key={i}>
                <NumberField
                  label={`k value ${i + 1}`}
                  value={k}
                  onChange={(value) => setKs(ks.map((old, j) => (j === i ? value : old)))}
                />
                <button
                  type="button"
                  disabled={ks.length === 1}
                  onClick={() => setKs(ks.filter((_old, j) => j !== i))}
                >
                  Remove k
                </button>
              </div>
            ))}
          </div>
          <button type="button" onClick={() => setKs([...ks, (ks.at(-1) ?? 0) + 5])}>
            Add k value
          </button>
        </fieldset>
        <RetrievalControls evaluation value={config} onChange={setConfig} />
        <button disabled={busy || !dataset.text}>
          {busy ? 'Running evaluation…' : 'Run and save evaluation'}
        </button>
        <p className="hint">
          Semantic/hybrid query embeddings and optional hosted reranking can incur charges. Limit:
          200 questions per run. Latency includes inference; no warmed-service guarantee.
        </p>
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
      </form>
      <section>
        <h2>Saved runs</h2>
        {runs.length === 0 && <p>No evaluation runs yet.</p>}
        {runs.map((r) => {
          const parsed = r.reportJson
            ? savedReportSchema.safeParse(JSON.parse(r.reportJson))
            : null;
          return (
            <details key={r._id} open={r.status === 'running'}>
              <summary>
                {r.name} · {r.status} · {new Date(r.createdAt).toLocaleString()}
              </summary>
              {r.error && <p className="error">{r.error}</p>}
              <button
                onClick={() =>
                  download(
                    `${r.name}.report.json`,
                    JSON.stringify(
                      {
                        config: JSON.parse(r.configJson),
                        documentIds: r.documentIds,
                        report: r.reportJson ? JSON.parse(r.reportJson) : null,
                        status: r.status,
                        error: r.error,
                      },
                      null,
                      2,
                    ),
                  )
                }
              >
                Download report
              </button>
              <button onClick={() => download(r.name, r.datasetJson)}>Download gold set</button>
              <pre>{r.configJson}</pre>
              {parsed?.success && (
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Method</th>
                        <th>k</th>
                        <th>Recall@k</th>
                        <th>Hit rate</th>
                        <th>MRR</th>
                        <th>Mean latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {parsed.data.reports.map((row, i) => (
                        <tr key={i}>
                          <td>
                            {row.method}
                            {row.rerank_model ? ` + ${row.rerank_model}` : ''}
                          </td>
                          <td>{row.top_k}</td>
                          <td>
                            <meter min={0} max={1} value={row.recall_at_k} />
                            {row.recall_at_k.toFixed(3)}
                          </td>
                          <td>{row.hit_rate_at_k.toFixed(3)}</td>
                          <td>{row.mrr_at_k.toFixed(3)}</td>
                          <td>{row.mean_seconds.toFixed(3)} s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </details>
          );
        })}
      </section>
    </main>
  );
}
