'use client';
import Link from 'next/link';
import { useParams, useRouter } from 'next/navigation';
import { useAction, useMutation, useQuery } from 'convex/react';
import { useEffect, useRef, useState } from 'react';
import { Markdown } from './Markdown';
import { z } from 'zod';
import { api } from '../../convex/_generated/api';
import type { Doc } from '../../convex/_generated/dataModel';
import { askConfigSchema, embeddingKey, ingestionConfigSchema } from '../../../src/domain/config';
import { chunkSchema } from '../../../src/domain/models';
import { decodeDocument } from '../../../src/ingestion/documents';
import { EmbeddingControls, IngestionControls, NumberField, RetrievalControls } from './Controls';

const errorText = (e: unknown) => (e instanceof Error ? e.message : 'Request failed');
type View = 'chat' | 'documents' | 'settings';
const evidenceSchema = z.object({
  sources: z.array(z.object({ chunk: chunkSchema, score: z.number() })),
});
export function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function RagWorkspace({ view = 'chat' }: { view?: View }) {
  const projects = useQuery(api.projects.list);
  const create = useMutation(api.projects.create);
  const router = useRouter();
  const params = useParams();
  const project = projects?.find((p) => p._id === params.projectId);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  return (
    <div className="project-app">
      <aside className="project-sidebar">
        <Link className="brand" href="/">
          <span className="brand-mark" aria-hidden="true">
            ⌑
          </span>{' '}
          Field Notes
        </Link>
        <div className="sidebar-label">
          <span>Your projects</span>
          <button
            className="icon-button"
            aria-label="Create project"
            onClick={() => setCreating(true)}
          >
            +
          </button>
        </div>
        {projects === undefined && <p className="hint">Loading projects…</p>}
        <nav className="project-list" aria-label="Projects">
          {projects?.map((p) => (
            <Link
              key={p._id}
              href={`/projects/${p._id}`}
              aria-current={p._id === project?._id ? 'page' : undefined}
            >
              <span aria-hidden="true">▧</span>
              {p.name}
            </Link>
          ))}
        </nav>
        <button className="secondary new-project" onClick={() => setCreating(true)}>
          + New project
        </button>
        <Link className="sidebar-footer" href="/evals">
          Retrieval evaluations
        </Link>
      </aside>
      <main className="project-main">
        {creating ? (
          <div className="welcome narrow">
            <h1>A space for your research.</h1>
            <p>Keep related documents and conversations together.</p>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                setBusy(true);
                setError('');
                void create({ name })
                  .then((id) => {
                    setCreating(false);
                    setName('');
                    router.push(`/projects/${id}`);
                  })
                  .catch((e) => setError(errorText(e)))
                  .finally(() => setBusy(false));
              }}
            >
              <label>
                Project name
                <input
                  autoFocus
                  required
                  maxLength={100}
                  placeholder="e.g. Climate research"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </label>
              {error && (
                <p role="alert" className="error">
                  {error}
                </p>
              )}
              <button disabled={busy || !name.trim()}>
                {busy ? 'Creating…' : 'Create project'}
              </button>
              <button type="button" className="secondary" onClick={() => setCreating(false)}>
                Cancel
              </button>
            </form>
          </div>
        ) : project ? (
          <ProjectWorkspace key={project._id} project={project} view={view} />
        ) : (
          <div className="welcome">
            <span className="empty-mark" aria-hidden="true">
              ⌑
            </span>
            <h1>
              Your documents.
              <br />
              Room to think.
            </h1>
            <p>
              {projects === undefined
                ? 'Loading your workspace…'
                : params.projectId
                  ? 'This project could not be found. Choose a project or create one.'
                  : 'Create a project, add your sources, and ask better questions.'}
            </p>
            <button onClick={() => setCreating(true)}>Create a project</button>
          </div>
        )}
      </main>
    </div>
  );
}

function ProjectWorkspace({ project, view }: { project: Doc<'projects'>; view: View }) {
  const root = `/projects/${project._id}`;
  const docs = useQuery(api.documents.list, { projectId: project._id });
  const traces = useQuery(api.documents.traces, { projectId: project._id });
  const legacy = useQuery(api.documents.list, view === 'documents' ? {} : 'skip');
  const ingest = useAction(api.rag.ingest);
  const ask = useAction(api.rag.ask);
  const reembed = useAction(api.rag.reembed);
  const exportDoc = useAction(api.rag.exportDocument);
  const save = useMutation(api.projects.saveSettings);
  const importDoc = useMutation(api.projects.importDocument);
  const [ingestion, setIngestion] = useState(() => ingestionConfigSchema.parse(project.ingestion));
  const [retrieval, setRetrieval] = useState(() => askConfigSchema.parse(project.retrieval));
  const [settingsTab, setSettingsTab] = useState<'ingestion' | 'retrieval'>('ingestion');
  const [files, setFiles] = useState<File[]>([]);
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState('');
  const [failed, setFailed] = useState(false);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (view === 'chat') end.current?.scrollIntoView({ block: 'end' });
  }, [traces?.length, view]);
  const ready = docs?.filter((d) => d.status === 'ready' && d.activeRun) ?? [];
  async function perform(label: string, fn: () => Promise<void>) {
    setBusy(label);
    setNotice('');
    setFailed(false);
    try {
      await fn();
    } catch (e) {
      setNotice(errorText(e));
      setFailed(true);
    } finally {
      setBusy('');
    }
  }
  return (
    <>
      <header className="project-header">
        <div>
          <p className="breadcrumb">Projects / {project.name}</p>
          <h1>{project.name}</h1>
        </div>
        <Link className="button secondary" href={`${root}/documents`}>
          + Add documents
        </Link>
      </header>
      <nav className="workspace-tabs" aria-label="Project views">
        {(['chat', 'documents', 'settings'] satisfies View[]).map((v) => (
          <Link
            key={v}
            aria-current={view === v ? 'page' : undefined}
            href={v === 'chat' ? root : `${root}/${v}`}
          >
            {v === 'chat' ? 'Chat' : v === 'documents' ? 'Documents' : 'Settings'}
            {v === 'documents' && docs && <span className="count">{docs.length}</span>}
          </Link>
        ))}
      </nav>
      {notice && (
        <div className={`notice ${failed ? 'error' : ''}`} role={failed ? 'alert' : 'status'}>
          {notice}
        </div>
      )}
      {view === 'chat' && (
        <div className="chat-layout">
          <div className="conversation" aria-label="Project conversation">
            {traces === undefined ? (
              <p className="hint">Loading conversation…</p>
            ) : (
              !traces.length && (
                <div className="chat-empty">
                  <span className="empty-mark" aria-hidden="true">
                    ⌑
                  </span>
                  <h2>What would you like to know?</h2>
                  <p>
                    {ready.length
                      ? `Ask a question across ${ready.length} indexed document${ready.length === 1 ? '' : 's'} in ${project.name}. Answers include the sources behind them.`
                      : 'Add your first document to start a conversation grounded in your sources.'}
                  </p>
                  {!ready.length && (
                    <Link className="button" href={`${root}/documents`}>
                      Add your first document
                    </Link>
                  )}
                  {ready.length > 0 && (
                    <div className="suggestions">
                      {[
                        'Summarize the main ideas in these documents.',
                        'What are the key findings?',
                      ].map((q) => (
                        <button className="secondary" key={q} onClick={() => setQuestion(q)}>
                          {q}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )
            )}
            {traces?.map((t) => {
              const evidence = evidenceSchema.safeParse(JSON.parse(t.sourcesJson || '{}'));
              return (
                <article className="chat-turn" key={t._id}>
                  <div className="user-message">
                    <span className="message-label">You</span>
                    <p>{t.question}</p>
                  </div>
                  <div className="assistant-message">
                    <span className="message-label">Field Notes</span>
                    {t.status === 'running' ? (
                      <p role="status" className="thinking">
                        Searching your documents…
                      </p>
                    ) : t.status === 'failed' ? (
                      <p className="error">{t.error}</p>
                    ) : (
                      <Markdown text={t.answer} />
                    )}
                    {evidence.success && evidence.data.sources.length > 0 && (
                      <details className="evidence">
                        <summary>{evidence.data.sources.length} sources · View evidence</summary>
                        <div className="source-list">
                          {evidence.data.sources.map(({ chunk }, i) => (
                            <details key={chunk.chunk_id}>
                              <summary>
                                [{i + 1}] {chunk.source}{' '}
                                <span className="hint">Chunk {chunk.chunk_index + 1}</span>
                              </summary>
                              <p className="source-text">{chunk.text}</p>
                              {chunk.context && (
                                <blockquote>
                                  <small>Generated retrieval context</small>
                                  {chunk.context}
                                </blockquote>
                              )}
                            </details>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                </article>
              );
            })}
            <div ref={end} />
          </div>
          <div className="composer-wrap">
            <form
              className="composer"
              onSubmit={(e) => {
                e.preventDefault();
                const q = question.trim();
                if (!q) return;
                void perform('Answering', async () => {
                  await ask({
                    projectId: project._id,
                    question: q,
                    documentIds: [],
                    config: askConfigSchema.parse(project.retrieval),
                  });
                  setQuestion('');
                });
              }}
            >
              <label className="sr-only" htmlFor="question">
                Ask your documents
              </label>
              <textarea
                id="question"
                required
                rows={2}
                placeholder="Ask anything about your documents…"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    if (!busy && ready.length && question.trim())
                      e.currentTarget.form?.requestSubmit();
                  }
                }}
              />
              <button disabled={!!busy || !ready.length || !question.trim()}>
                {busy === 'Answering' ? 'Thinking…' : 'Send'}
              </button>
            </form>
            <p className="composer-note">
              Answers use this project&apos;s documents. Verify important details against the
              sources.
            </p>
          </div>
        </div>
      )}
      {view === 'documents' && (
        <div className="content-page">
          <div className="section-heading">
            <div>
              <h2>Your source material</h2>
              <p>
                Add documents here. Tune how they are indexed in{' '}
                <Link href={`${root}/settings`}>project settings</Link>.
              </p>
            </div>
            <span className="status-pill">{ready.length} ready</span>
          </div>
          <form
            className="upload-zone"
            onSubmit={(e) => {
              e.preventDefault();
              void perform('Indexing', async () => {
                if (files.length > 8) throw new Error('Choose up to 8 files per upload');
                const uploads = [];
                for (const file of files) {
                  if (file.size > 4_000_000)
                    throw new Error(`${file.name} exceeds the upload limit`);
                  uploads.push({
                    filename: file.name,
                    content: decodeDocument(await file.arrayBuffer()),
                  });
                }
                const results = await ingest({
                  projectId: project._id,
                  files: uploads,
                  runName: '',
                  config: ingestionConfigSchema.parse(project.ingestion),
                });
                setFailed(results.some((r) => !!r.error));
                setNotice(
                  results
                    .map(
                      (r) =>
                        `${r.filename}: ${r.error ?? (r.reused ? 'Already indexed' : 'Ready')}`,
                    )
                    .join('\n'),
                );
                setFiles([]);
              });
            }}
          >
            <span className="upload-symbol" aria-hidden="true">
              ＋
            </span>
            <h3>Add to {project.name}</h3>
            <p>Markdown and plain text. Up to 8 files, 500,000 characters each.</p>
            <label>
              Choose documents
              <input
                required
                type="file"
                multiple
                accept=".md,.txt"
                onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
              />
            </label>
            <button disabled={!!busy || !files.length}>
              {busy === 'Indexing' ? 'Indexing documents…' : 'Index selected documents'}
            </button>
            <small>
              Uses your saved ingestion settings. Existing documents are unaffected by settings
              changes.
            </small>
          </form>
          <div className="document-list">
            {docs === undefined ? (
              <p>Loading documents…</p>
            ) : !docs.length ? (
              <p className="hint">No documents yet. Add a source above to get started.</p>
            ) : (
              docs.map((d) => (
                <div className="document-row" key={d._id}>
                  <span className="file-symbol" aria-hidden="true">
                    ▤
                  </span>
                  <div className="document-info">
                    <strong>{d.filename}</strong>
                    <small>
                      {d.chunkCount} chunks · {d.embeddingModel ?? 'No embedding space'}
                    </small>
                    {d.error && <p className="error">{d.error}</p>}
                  </div>
                  <span className={`status-pill ${d.status}`}>
                    {d.pendingRun ? 'Indexing' : d.status}
                  </span>
                  <details className="document-actions">
                    <summary>Details</summary>
                    <p className="hint">Ingestion manifest</p>
                    <pre>{d.configJson}</pre>
                    {d.activeRun && (
                      <>
                        <button
                          className="secondary"
                          disabled={!!busy}
                          onClick={() =>
                            void perform('Exporting', async () => {
                              const r = await exportDoc({
                                projectId: project._id,
                                documentId: d._id,
                              });
                              download(`${d.filename}.chunks.jsonl`, r.chunks);
                              download(`${d.filename}.embeddings.jsonl`, r.embeddings);
                            })
                          }
                        >
                          Export chunks & vectors
                        </button>
                        <button
                          className="secondary"
                          disabled={!!busy || !!d.pendingRun}
                          onClick={() =>
                            void perform('Re-embedding', async () => {
                              await reembed({
                                projectId: project._id,
                                documentId: d._id,
                                embedding: project.retrieval.embedding,
                                runName: 'Re-embedding',
                              });
                              setNotice('Re-embedded with the saved retrieval embedding settings.');
                            })
                          }
                        >
                          Re-embed with project settings
                        </button>
                      </>
                    )}
                  </details>
                </div>
              ))
            )}
          </div>
          {!!legacy?.length && (
            <details className="legacy-import">
              <summary>Import documents from your previous library ({legacy.length})</summary>
              <p className="hint">
                Move an existing document into this project without indexing it again.
              </p>
              {legacy.map((d) => (
                <div className="import-row" key={d._id}>
                  <span>{d.filename}</span>
                  <button
                    className="secondary"
                    disabled={!!busy || !!d.pendingRun}
                    onClick={() =>
                      void perform('Importing', async () => {
                        await importDoc({ projectId: project._id, documentId: d._id });
                      })
                    }
                  >
                    Import
                  </button>
                </div>
              ))}
            </details>
          )}
        </div>
      )}
      {view === 'settings' && (
        <div className="content-page settings-page">
          <div className="section-heading">
            <div>
              <h2>Project settings</h2>
              <p>
                Saved defaults for {project.name}. Model credentials are configured on the server.
              </p>
            </div>
          </div>
          <div className="settings-tabs" role="group" aria-label="Settings category">
            <button
              className={settingsTab === 'ingestion' ? '' : 'secondary'}
              onClick={() => setSettingsTab('ingestion')}
            >
              Ingestion
            </button>
            <button
              className={settingsTab === 'retrieval' ? '' : 'secondary'}
              onClick={() => setSettingsTab('retrieval')}
            >
              Retrieval & answers
            </button>
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void perform('Saving', async () => {
                await save({
                  projectId: project._id,
                  ingestion: ingestionConfigSchema.parse(ingestion),
                  retrieval: askConfigSchema.parse(retrieval),
                });
                setNotice(
                  'Project settings saved. New uploads and questions will use these settings.',
                );
              });
            }}
          >
            {settingsTab === 'ingestion' ? (
              <>
                <p className="hint">
                  Applies to new uploads. Re-upload a document to change chunking or generated
                  context.
                </p>
                <IngestionControls value={ingestion} onChange={setIngestion} />
                <button
                  type="button"
                  className="secondary"
                  onClick={() => setRetrieval({ ...retrieval, embedding: ingestion.embedding })}
                >
                  Use this embedding space for retrieval too
                </button>
              </>
            ) : (
              <>
                <fieldset>
                  <legend>Answer model</legend>
                  <label>
                    OpenRouter model
                    <input
                      required
                      value={retrieval.generation.model}
                      onChange={(e) =>
                        setRetrieval({
                          ...retrieval,
                          generation: { ...retrieval.generation, model: e.target.value },
                        })
                      }
                    />
                  </label>
                  <NumberField
                    label="Answer output tokens"
                    value={retrieval.generation.maxTokens}
                    onChange={(maxTokens) =>
                      setRetrieval({
                        ...retrieval,
                        generation: { ...retrieval.generation, maxTokens },
                      })
                    }
                  />
                </fieldset>
                <details>
                  <summary>Retrieval and reranking</summary>
                  <RetrievalControls value={retrieval} onChange={setRetrieval} />
                </details>
                <details>
                  <summary>Query embedding space</summary>
                  <p className="hint">
                    Must match your indexed documents for semantic and hybrid search. Current
                    ingestion space: {embeddingKey(ingestion.embedding)}.
                  </p>
                  <EmbeddingControls
                    value={retrieval.embedding}
                    onChange={(embedding) => setRetrieval({ ...retrieval, embedding })}
                  />
                </details>
              </>
            )}
            <div className="settings-save">
              <button disabled={!!busy}>
                {busy === 'Saving' ? 'Saving…' : 'Save project settings'}
              </button>
            </div>
          </form>
        </div>
      )}
    </>
  );
}
