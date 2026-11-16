import { expect, test } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { Markdown } from '../features/rag/Markdown';

test('answers render Markdown, tables and code without enabling raw HTML', () => {
  const html = renderToStaticMarkup(
    <Markdown
      text={
        '## Findings\n\n**Evidence** [1]\n\n| Source | Result |\n| --- | --- |\n| Notes | Yes |\n\n```js\nconst answer = 42;\n```\n\n<script>alert(1)</script>\n\n[unsafe](javascript:alert%281%29)'
      }
    />,
  );
  expect(html).toContain('<h2>Findings</h2>');
  expect(html).toContain('<strong>Evidence</strong>');
  expect(html).toContain('<table>');
  expect(html).toContain('language-js');
  expect(html).not.toContain('<script>');
  expect(html).not.toContain('href="javascript:');
});
