import js from '@eslint/js';
import ts from 'typescript-eslint';

export default ts.config(
  js.configs.recommended,
  ...ts.configs.strictTypeChecked,
  {
    files: ['**/*.ts'],
    languageOptions: {
      parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname },
    },
    rules: {
      '@typescript-eslint/restrict-template-expressions': ['error', { allowNumber: true }],
      '@typescript-eslint/no-confusing-void-expression': 'off',
    },
  },
  {
    files: ['src/domain/**/*.ts'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            'node:*',
            '../infrastructure/*',
            '../storage/*',
            '../pipelines/*',
            '../cli/*',
            '../embeddings/*',
            '../retrieval/*',
            '../ingestion/*',
            '../generation/*',
            '../evaluation/*',
          ],
        },
      ],
    },
  },
  {
    files: ['src/**/*.ts'],
    ignores: ['src/cli/**', 'src/index.ts', 'src/domain/**'],
    rules: { 'no-restricted-imports': ['error', { patterns: ['../cli/*', './cli/*'] }] },
  },
  {
    files: [
      'src/ingestion/**/*.ts',
      'src/embeddings/**/*.ts',
      'src/retrieval/**/*.ts',
      'src/generation/**/*.ts',
      'src/storage/**/*.ts',
    ],
    rules: { 'no-restricted-imports': ['error', { patterns: ['../cli/*', '../pipelines/*'] }] },
  },
  { ignores: ['dist/**', 'web/**', 'node_modules/**'] },
);
