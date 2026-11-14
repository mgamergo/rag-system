"use client";

import { useAction, useQuery } from "convex/react";
import { useRef, useState } from "react";
import { api } from "../../convex/_generated/api";

type Answer = { answer: string; latencyMs: number; traceId: string; usage: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number }; sources: { id: string; chunkIndex: number; text: string; score: number }[] };

export function RagWorkspace() {
  const inputRef = useRef<HTMLInputElement>(null);
  const documents = useQuery(api.documents.list) ?? [];
  const ingest = useAction(api.rag.ingest);
  const ask = useAction(api.rag.ask);
  const [uploadState, setUploadState] = useState<"idle" | "indexing" | "failed">("idle");
  const [uploadError, setUploadError] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [askState, setAskState] = useState<"idle" | "asking" | "failed">("idle");
  const [askError, setAskError] = useState("");

  async function uploadDocument(file: File) {
    if (!/\.(md|txt)$/i.test(file.name)) { setUploadState("failed"); setUploadError("Choose a Markdown or text file."); return; }
    setUploadState("indexing"); setUploadError("");
    try { await ingest({ filename: file.name, content: await file.text() }); setUploadState("idle"); if (inputRef.current) inputRef.current.value = ""; }
    catch (error) { setUploadState("failed"); setUploadError(error instanceof Error ? error.message : "Indexing failed."); }
  }

  async function submitQuestion(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault(); if (!question.trim()) return;
    setAskState("asking"); setAskError(""); setAnswer(null);
    try { const result = await ask({ question, topK: 6 }); setAnswer(result); setAskState("idle"); }
    catch (error) { setAskState("failed"); setAskError(error instanceof Error ? error.message : "The question could not be answered."); }
  }

  const readyCount = documents.filter((document) => document.status === "ready").length;
  return <main className="shell">
    <header className="masthead"><div><p className="edition">RESEARCH DESK / LOCAL CORPUS</p><h1>Field Notes</h1></div><p className="status"><span aria-hidden="true">●</span> {readyCount} indexed {readyCount === 1 ? "document" : "documents"}</p></header>
    <section className="intro"><p>Upload notes, index them remotely, and ask questions with source evidence. Vectors live in Convex; embeddings and answers come from OpenRouter.</p></section>
    <div className="workspace"><aside className="library" aria-label="Document library"><div className="section-title"><h2>Library</h2><span>{documents.length}</span></div><label className="upload-control"><input ref={inputRef} type="file" accept=".md,.txt,text/markdown,text/plain" onChange={(event) => { const file = event.target.files?.[0]; if (file) void uploadDocument(file); }} /><span>{uploadState === "indexing" ? "Indexing…" : "Add document"}</span></label>{uploadState === "failed" && <p className="error">{uploadError}</p>}<ul className="documents">{documents.length === 0 && <li className="empty">No papers or notes yet.</li>}{documents.map((document) => <li key={document._id}><p>{document.filename}</p><small className={`document-status ${document.status}`}>{document.status === "ready" ? `${document.chunkCount} chunks` : document.status}</small>{document.error && <small className="error">{document.error}</small>}</li>)}</ul></aside>
      <section className="conversation" aria-label="Question and answer"><form onSubmit={submitQuestion} className="question-form"><label htmlFor="question">Ask your research collection</label><div><textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="What does the paper claim about…" rows={3} /><button type="submit" disabled={askState === "asking" || readyCount === 0}>{askState === "asking" ? "Searching…" : "Ask"}</button></div></form>{readyCount === 0 && <p className="hint">Index a `.md` or `.txt` document before asking a question.</p>}{askState === "failed" && <p className="error answer-error">{askError}</p>}{answer && <AnswerView answer={answer} />}</section>
    </div></main>;
}

function AnswerView({ answer }: { answer: Answer }) {
  return <article className="answer"><div className="answer-heading"><h2>Answer</h2><span>{answer.latencyMs} ms</span></div><p className="answer-text">{answer.answer}</p><p className="trace">Trace {answer.traceId} · {answer.usage.total_tokens ?? "unknown"} tokens</p><section className="sources"><h3>Evidence sent to Gemma</h3>{answer.sources.map((source, index) => <details key={source.id}><summary><span>[{index + 1}]</span> chunk {source.chunkIndex + 1} <small>score {source.score.toFixed(3)}</small></summary><p>{source.text}</p></details>)}</section></article>;
}
