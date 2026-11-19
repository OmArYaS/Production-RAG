import { useState } from "react";
import { api } from "../api";

export default function QuestionBox() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const ask = async () => {
    const q = question.trim();
    if (!q) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const resp = await api.askQuestion(q);
      setResult(resp);
      setQuestion("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderCitations = (citations) => (
    <div className="citations">
      <strong>Citations:</strong>
      {citations.map((c) => (
        <div className="citation" key={c.chunk_id}>
          <div className="c-meta">
            [{c.index}] {c.source_name}
            {c.page_number != null ? ` (page ${c.page_number})` : ""}
            {c.section_heading ? ` — ${c.section_heading}` : ""}
          </div>
          <div className="c-snippet">{c.snippet}</div>
        </div>
      ))}
    </div>
  );

  return (
    <section>
      <h2>Ask a question</h2>
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask something grounded in your documents…"
        rows={3}
      />
      <div style={{ marginTop: 10 }}>
        <button onClick={ask} disabled={loading || !question.trim()}>
          {loading ? "Thinking…" : "Ask"}
        </button>
      </div>
      {error && <div style={{ color: "#c0392b", margin: "8px 0" }}>{error}</div>}
      {result && (
        <div className="answer-box">
          {result.abstained && (
            <div className="abstain">
              Could not answer from the retrieved documents (abstained).
            </div>
          )}
          <div style={{ marginTop: 6 }}>{result.answer}</div>
          {result.citations && result.citations.length > 0 && renderCitations(result.citations)}
          <div className="meta-line" style={{ marginTop: 6 }}>
            Model: {result.model} · {result.hits_count} chunk(s) retrieved ·{" "}
            {result.latency_ms.toFixed(0)} ms
          </div>
        </div>
      )}
    </section>
  );
}
