import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        waltr: {
          bg: '#0b1220',
          panel: '#131c2e',
          accent: '#3b82f6',
          muted: '#64748b',
        },
      },
    },
  },
  plugins: [],
};
export default config;
