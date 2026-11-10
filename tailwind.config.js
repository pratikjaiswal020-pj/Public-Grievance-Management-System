/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/static/**/*.html", "./app/static/**/*.js"],
  theme: {
    extend: {
      colors: {
        primary: "#00236f", "on-primary": "#ffffff",
        "primary-container": "#1e3a8a", "on-primary-container": "#90a8ff",
        "primary-fixed": "#dce1ff", "primary-fixed-dim": "#b6c4ff",
        secondary: "#006a61", "on-secondary": "#ffffff",
        "secondary-container": "#86f2e4", "on-secondary-container": "#006f66",
        tertiary: "#d97706", "tertiary-container": "#fef3c7", "on-tertiary-container": "#92400e",
        surface: "#faf8ff", "on-surface": "#131b2e",
        "surface-container-lowest": "#ffffff", "surface-container-low": "#f2f3ff",
        "surface-container": "#eaedff", "surface-container-high": "#e2e7ff",
        "on-surface-variant": "#444651", outline: "#757682", "outline-variant": "#c5c5d3",
        error: "#ba1a1a", "error-container": "#ffdad6", "on-error-container": "#93000a"
      },
      fontFamily: {
        sans: ["Public Sans", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"]
      }
    }
  }
};
