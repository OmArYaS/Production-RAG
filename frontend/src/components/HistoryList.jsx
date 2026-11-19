import { useEffect, useState } from "react";
import { api } from "../api";

export default function HistoryList() {
  const [entries, setEntries] = useState([]);
  const [error, setError] = useState("");

  const refresh = async () => {
    const data = await api.listHistory();
    setEntries(data);
  };

  useEffect(() => {
    refresh().catch((e) => setError(e.message));
  }, []);

  return (
    <section>
      <h2>Q&A History</h2>
      {error && <div style={{ color: "#c0392b", margin: "8px 0" }}>{error}</div>}
      {entries.length === 0 && <p className="empty">No questions asked yet.</p>}
      {entries.map((e) => (
        <div className="history-entry" key={e.id}>
          <div className="he-q">Q: {e.question}</div>
          <div className="he-a">A: {e.answer}</div>
          {e.citations && e.citations.length > 0 && (
            <div className="he-a">
              <strong>Sources:</strong>{" "}
              {e.citations
                .map((c) => `${c.source_name}${c.page_number != null ? ` (p. ${c.page_number})` : ""}`)
                .join(", ")}
            </div>
          )}
          <div className="meta-line">
            {e.model} · {e.latency_ms.toFixed(0)} ms · {e.abstained ? "abstained" : "answered"}
          </div>
        </div>
      ))}
    </section>
  );
}
