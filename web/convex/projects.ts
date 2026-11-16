import { v } from 'convex/values';
import { query, mutation } from './_generated/server';
import { ingestion, ask } from './validators';
import {
  DEFAULT_INGESTION,
  DEFAULT_ASK,
  ingestionConfigSchema,
  askConfigSchema,
} from '../../src/domain/config';

export const list = query({
  args: {},
  handler: async (ctx) => await ctx.db.query('projects').order('desc').collect(),
});
export const create = mutation({
  args: { name: v.string() },
  handler: async (ctx, args) => {
    const name = args.name.trim();
    if (!name || name.length > 100) throw new Error('Project name must be 1–100 characters');
    return await ctx.db.insert('projects', {
      name,
      ingestion: DEFAULT_INGESTION,
      retrieval: DEFAULT_ASK,
      createdAt: Date.now(),
    });
  },
});
export const saveSettings = mutation({
  args: { projectId: v.id('projects'), ingestion, retrieval: ask },
  handler: async (ctx, args) => {
    if (!(await ctx.db.get(args.projectId))) throw new Error('Project not found');
    await ctx.db.patch(args.projectId, {
      ingestion: ingestionConfigSchema.parse(args.ingestion),
      retrieval: askConfigSchema.parse(args.retrieval),
    });
  },
});
// Existing documents stay unassigned until explicitly moved into a project.
export const importDocument = mutation({
  args: { projectId: v.id('projects'), documentId: v.id('documents') },
  handler: async (ctx, args) => {
    if (!(await ctx.db.get(args.projectId))) throw new Error('Project not found');
    const doc = await ctx.db.get(args.documentId);
    if (!doc || doc.projectId) throw new Error('Only unassigned documents can be imported');
    if (doc.pendingRun) throw new Error('Wait for indexing to finish before importing');
    await ctx.db.patch(doc._id, { projectId: args.projectId });
  },
});
