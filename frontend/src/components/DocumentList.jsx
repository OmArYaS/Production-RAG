import { useEffect, useState } from "react";
import { api } from "../api";

export default function DocumentList() {
  const [docs, setDocs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  const refresh = async () => {
    const data = await api.listDocuments();
    setDocs(data);
  };

  useEffect(() => {
    refresh().catch((e) => setError(e.message));
  }, []);

  const onUpload = async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      await api.uploadDocument(file);
      await refresh();
      e.target.value = "";
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const onDelete = async (docId) => {
    setError("");
    try {
      await api.deleteDocument(docId);
      await refresh();
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <section>
      <h2>Documents</h2>
      <div className="upload-row">
        <input type="file" onChange={onUpload} accept=".txt,.md,.pdf" />
        <button disabled={uploading}>{uploading ? "Uploading…" : "Upload"}</button>
        <span className="meta-line">Supported: .txt, .md, .pdf</span>
      </div>
      {error && <div style={{ color: "#c0392b", margin: "8px 0" }}>{error}</div>}
      {docs.length === 0 && !error && <p className="empty">No documents indexed yet.</p>}
      {docs.map((d) => (
        <div className="doc-row" key={d.doc_id}>
          <div>
            <div className="doc-name">{d.source_name}</div>
            <div className="doc-meta">
              {d.chunk_count} chunk(s) · {d.content_type} · {d.updated_at}
            </div>
          </div>
          <button onClick={() => onDelete(d.doc_id)}>Delete</button>
        </div>
      ))}
    </section>
  );
}
