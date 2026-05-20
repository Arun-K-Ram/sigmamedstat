/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        trust: {
          excellent: '#10b981',
          good: '#84cc16',
          degraded: '#f59e0b',
          poor: '#ef4444',
          critical: '#7f1d1d',
        }
      }
    },
  },
  plugins: [],
}