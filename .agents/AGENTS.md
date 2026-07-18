# Project Agent Rules

## Tailwind CSS v4 Configuration
- When initializing or writing configuration for Tailwind CSS, ALWAYS check the version.
- **Tailwind v4** does NOT use `tailwind.config.js`. 
- **Tailwind v4** uses `@import "tailwindcss";` in the main CSS file, replacing the old `@tailwind base;`, `@tailwind components;`, and `@tailwind utilities;` directives.
- When configuring Tailwind v4 for a **Vite** project, do not use PostCSS. Instead, install the new Vite plugin: `npm install tailwindcss @tailwindcss/vite` and add it to `vite.config.ts`. You do not need `postcss.config.js`.

## PyTorch Security
- When loading models or tensors using `torch.load()`, ALWAYS include the `weights_only=True` parameter to prevent arbitrary code execution vulnerabilities, especially when dealing with federated or external model weights.
