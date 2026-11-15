'use client';
import { z } from 'zod';
import type { AskConfig, EmbeddingConfig, IngestionConfig } from '../../../src/domain/config';
import {
  boundarySchema,
  lexicalMethodSchema,
  retrievalMethodSchema,
  rerankProviderSchema,
} from '../../../src/domain/models';
export function NumberField({
  label,
  value,
  onChange,
  min = 1,
  max,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
}) {
  return (
    <label>
      {label}
      <input
        type="number"
        required
        min={min}
        max={max}
        step="any"
        value={Number.isFinite(value) ? value : ''}
        onChange={(e) => onChange(e.target.valueAsNumber)}
      />
    </label>
  );
}
export function Toggle({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );
}
const providerSchema = z.enum(['openrouter', 'voyage', 'gemini']);
const defaults = {
  openrouter: 'nvidia/nemotron-3-embed-1b:free',
  voyage: 'voyage-3.5',
  gemini: 'gemini-embedding-001',
};
export function EmbeddingControls({
  value,
  onChange,
}: {
  value: EmbeddingConfig;
  onChange: (value: EmbeddingConfig) => void;
}) {
  return (
    <fieldset>
      <legend>Embedding space</legend>
      <div className="fields">
        <label>
          Provider
          <select
            value={value.provider}
            onChange={(e) => {
              const provider = providerSchema.parse(e.target.value);
              onChange({ ...value, provider, model: defaults[provider] });
            }}
          >
            {providerSchema.options.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
        </label>
        <label>
          Model
          <input
            required
            value={value.model}
            onChange={(e) => onChange({ ...value, model: e.target.value })}
          />
        </label>
        <NumberField
          label="Batch size"
          value={value.batchSize}
          onChange={(batchSize) => onChange({ ...value, batchSize })}
        />
        <NumberField
          label="Embedding timeout (seconds)"
          max={600}
          value={value.timeout}
          onChange={(timeout) => onChange({ ...value, timeout })}
        />
      </div>
      <p className="hint">
        Hosted inference. Local ONNX, GPU/device selection and localhost Ollama are unavailable in
        the deployed app. Model changes require re-embedding.
      </p>
    </fieldset>
  );
}
export function IngestionControls({
  value,
  onChange,
}: {
  value: IngestionConfig;
  onChange: (value: IngestionConfig) => void;
}) {
  const context = value.context;
  return (
    <>
      <fieldset>
        <legend>Chunking</legend>
        <div className="fields">
          <NumberField
            label="Chunk size (code points)"
            value={value.chunkSize}
            onChange={(chunkSize) => onChange({ ...value, chunkSize })}
          />
          <NumberField
            label="Overlap"
            min={0}
            value={value.overlap}
            onChange={(overlap) => onChange({ ...value, overlap })}
          />
          <label>
            Boundary
            <select
              value={value.boundary}
              onChange={(e) =>
                onChange({ ...value, boundary: boundarySchema.parse(e.target.value) })
              }
            >
              {boundarySchema.options.map((x) => (
                <option key={x}>{x}</option>
              ))}
            </select>
          </label>
        </div>
      </fieldset>
      <EmbeddingControls
        value={value.embedding}
        onChange={(embedding) => onChange({ ...value, embedding })}
      />
      <fieldset>
        <legend>Contextual retrieval</legend>
        <Toggle
          label="Generate document-aware context with Anthropic"
          value={context.enabled}
          onChange={(enabled) =>
            onChange({
              ...value,
              context: { ...context, enabled, instruction: enabled ? context.instruction : '' },
            })
          }
        />
        {context.enabled && (
          <>
            <div className="fields">
              <label>
                Anthropic model
                <input
                  required
                  value={context.model}
                  onChange={(e) =>
                    onChange({ ...value, context: { ...context, model: e.target.value } })
                  }
                />
              </label>
              <NumberField
                label="Context output tokens"
                value={context.maxTokens}
                onChange={(maxTokens) => onChange({ ...value, context: { ...context, maxTokens } })}
              />
              <NumberField
                label="Context timeout (seconds)"
                max={600}
                value={context.timeout}
                onChange={(timeout) => onChange({ ...value, context: { ...context, timeout } })}
              />
            </div>
            <Toggle
              label="Cache shared document prefix"
              value={context.cacheDocument}
              onChange={(cacheDocument) =>
                onChange({ ...value, context: { ...context, cacheDocument } })
              }
            />
            <label>
              Custom instruction (optional)
              <textarea
                value={context.instruction}
                onChange={(e) =>
                  onChange({ ...value, context: { ...context, instruction: e.target.value } })
                }
              />
            </label>
            <label>
              Import instruction text
              <input
                type="file"
                accept=".txt,.md"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f)
                    void f
                      .text()
                      .then((instruction) =>
                        onChange({ ...value, context: { ...context, instruction } }),
                      );
                }}
              />
            </label>
          </>
        )}
      </fieldset>
    </>
  );
}
export function RetrievalControls({
  value,
  onChange,
  evaluation = false,
}: {
  value: AskConfig;
  onChange: (value: AskConfig) => void;
  evaluation?: boolean;
}) {
  const r = value.rerank;
  return (
    <>
      <fieldset>
        <legend>Retrieval</legend>
        <div className="fields">
          {!evaluation && (
            <label>
              Method
              <select
                value={value.method}
                onChange={(e) =>
                  onChange({ ...value, method: retrievalMethodSchema.parse(e.target.value) })
                }
              >
                {retrievalMethodSchema.options.map((x) => (
                  <option key={x}>{x}</option>
                ))}
              </select>
            </label>
          )}
          <label>
            Hybrid lexical branch
            <select
              value={value.lexicalMethod}
              onChange={(e) =>
                onChange({ ...value, lexicalMethod: lexicalMethodSchema.parse(e.target.value) })
              }
            >
              {lexicalMethodSchema.options.map((x) => (
                <option key={x}>{x}</option>
              ))}
            </select>
          </label>
          {!evaluation && (
            <NumberField
              label="Final sources (top k)"
              value={value.topK}
              onChange={(topK) => onChange({ ...value, topK })}
            />
          )}
          <NumberField
            label="Candidates per branch"
            value={value.candidateK}
            onChange={(candidateK) => onChange({ ...value, candidateK })}
          />
          <NumberField
            label="RRF constant"
            value={value.rrfConstant}
            onChange={(rrfConstant) => onChange({ ...value, rrfConstant })}
          />
          <NumberField
            label="Generation / rerank timeout (seconds)"
            max={600}
            value={value.timeout}
            onChange={(timeout) => onChange({ ...value, timeout })}
          />
        </div>
      </fieldset>
      <fieldset>
        <legend>Reranking</legend>
        <Toggle
          label="Rerank candidates before selecting sources"
          value={r.enabled}
          onChange={(enabled) => onChange({ ...value, rerank: { ...r, enabled } })}
        />
        {r.enabled && (
          <div className="fields">
            <label>
              Provider
              <select
                value={r.provider}
                onChange={(e) =>
                  onChange({
                    ...value,
                    rerank: { ...r, provider: rerankProviderSchema.parse(e.target.value) },
                  })
                }
              >
                {rerankProviderSchema.options.map((x) => (
                  <option key={x}>{x}</option>
                ))}
              </select>
            </label>
            <label>
              Reranking model
              <input
                required
                value={r.model}
                onChange={(e) => onChange({ ...value, rerank: { ...r, model: e.target.value } })}
              />
            </label>
          </div>
        )}
      </fieldset>
      {!evaluation && (
        <fieldset>
          <legend>Full corpus alternative</legend>
          <Toggle
            label="Send complete selected documents; bypass retrieval"
            value={value.fullContext}
            onChange={(fullContext) => onChange({ ...value, fullContext })}
          />
          {value.fullContext && (
            <NumberField
              label="Full-context budget (code points)"
              value={value.fullContextMaxChars}
              onChange={(fullContextMaxChars) => onChange({ ...value, fullContextMaxChars })}
            />
          )}
          <p className="hint">
            Full context cannot use reranking. The character budget is not a model token-window
            guarantee.
          </p>
        </fieldset>
      )}
    </>
  );
}
