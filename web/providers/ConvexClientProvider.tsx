'use client';

import { ConvexProvider, ConvexReactClient } from 'convex/react';
import type { ReactNode } from 'react';

const deploymentUrl = process.env.NEXT_PUBLIC_CONVEX_URL;
if (!deploymentUrl)
  throw new Error('Missing NEXT_PUBLIC_CONVEX_URL. Run `npx convex dev` in web first.');
const convex = new ConvexReactClient(deploymentUrl);

export function ConvexClientProvider({ children }: { children: ReactNode }) {
  return <ConvexProvider client={convex}>{children}</ConvexProvider>;
}
