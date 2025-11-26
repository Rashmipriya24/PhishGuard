/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cyber: {
          dark: '#0a0e27',
          darker: '#050811',
          blue: '#00d9ff',
          green: '#00ff88',
          red: '#ff0055',
          purple: '#b829ff',
          gray: '#1a1f3a',
        }
      },
      boxShadow: {
        'cyber': '0 0 20px rgba(0, 217, 255, 0.3)',
        'cyber-green': '0 0 20px rgba(0, 255, 136, 0.3)',
        'cyber-red': '0 0 20px rgba(255, 0, 85, 0.3)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'scan': 'scan 2s ease-in-out infinite',
      },
      keyframes: {
        scan: {
          '0%, 100%': { transform: 'translateY(-100%)' },
          '50%': { transform: 'translateY(100%)' },
        }
      }
    },
  },
  plugins: [],
}
