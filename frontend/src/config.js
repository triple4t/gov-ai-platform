/**
 * API base URL from environment. Set VITE_API_URL when building for production.
 * Example: VITE_API_URL=https://api.example.com npm run build
 */
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const API_BASE = `${BASE_URL}/api/v1`;

/** WebSocket base (ws:// or wss://) for streaming endpoints */
export function getWebSocketBase() {
  return BASE_URL.replace(/^http/, 'ws');
}
