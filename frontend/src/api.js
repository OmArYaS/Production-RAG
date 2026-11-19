// API client for the Hybrid RAG backend.
// Base URL is relative; the dev server proxies /api to the backend.

const base = import.meta.env.VITE_API_BASE || "/api";

async function _request(path, options = {}) {
  const resp = await fetch(base + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let data = null;
  try {
    data = await resp.json();
  } catch {
    data = null;
  }
  if (!resp.ok) {
    const message = data && data.detail ? data.detail : `HTTP ${resp.status}`;
    throw new Error(message);
  }
  return data;
}

export const api = {
  health: () => _request("/health"),
  listDocuments: () => _request("/documents"),
  uploadDocument: async (file) => {
    const fd = new FormData();
    fd.append("file", file);
    const resp = await fetch(base + "/documents", { method: "POST", body: fd });
    let data = null;
    try {
      data = await resp.json();
    } catch {
      data = null;
    }
    if (!resp.ok) {
      throw new Error((data && data.detail) || `HTTP ${resp.status}`);
    }
    return data;
  },
  deleteDocument: (docId) =>
    _request(`/documents/${encodeURIComponent(docId)}`, { method: "DELETE" }),
  askQuestion: (question) =>
    _request("/query", {
      method: "POST",
      body: JSON.stringify({ question }),
    }),
  listHistory: () => _request("/history"),
};
