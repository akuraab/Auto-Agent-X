# Auto-Agent-X Frontend

This is the frontend for the Auto-Agent-X project, built with React, Vite, and Tailwind CSS.

## Getting Started

1.  Install dependencies:
    ```bash
    npm install
    ```

2.  Start the development server:
    ```bash
    npm run dev
    ```

3.  The application will be available at `http://localhost:5173`.

## Configuration

-   **Vite**: Configured in `vite.config.ts`.
-   **Tailwind CSS**: Configured in `tailwind.config.js` and `postcss.config.js`.
-   **Proxy**: API requests to `/api` are proxied to `http://localhost:8000` (backend).
