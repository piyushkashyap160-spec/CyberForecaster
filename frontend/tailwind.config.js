/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        base: {
          bg: "#0a0e14",        // near-black, slightly warm, not pure #000
          surface: "#12161f",   // card/panel background
          surfaceHover: "#171c27",
          border: "#232838",    // subtle 1px borders, not glow
          borderActive: "#2f3648",
        },
        text: {
          primary: "#e6e8ec",
          secondary: "#8b93a7",
          muted: "#5a6275",
        },
        accent: {
          DEFAULT: "#5b8def",   // one restrained blue accent
          hover: "#4b7ddf",
          subtle: "rgba(91, 141, 239, 0.1)",
        },
        severity: {
          critical: "#f0506e",  // muted red
          high: "#f0a050",      // muted amber
          medium: "#e0c050",    // muted yellow
          low: "#5b8def",       // accent blue
          normal: "#3ecf8e",    // muted green
        }
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Menlo', 'monospace']
      }
    },
  },
  plugins: [],
}
