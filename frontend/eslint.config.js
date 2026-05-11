import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
    },
  },
  {
    /*
     * shadcn-style UI primitives re-export Radix subcomponents (e.g. Tabs.Root)
     * and Context providers ship with their own hook for ergonomics. Both patterns
     * trip react-refresh's "only-export-components" rule even though they're fine
     * at runtime. Restrict the scope rather than disabling globally.
     */
    files: ['src/components/ui/**/*.{ts,tsx}', 'src/state/**/*.{ts,tsx}'],
    rules: {
      'react-refresh/only-export-components': 'off',
    },
  },
])
