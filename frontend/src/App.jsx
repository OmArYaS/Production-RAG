import { useState } from "react";
import DocumentList from "./components/DocumentList";
import QuestionBox from "./components/QuestionBox";
import HistoryList from "./components/HistoryList";

export default function App() {
  return (
    <div>
      <header className="app">
        <h1>Hybrid Document & Knowledge Assistant</h1>
        <span className="subtitle">
          Local RAG: upload documents, ask grounded questions.
        </span>
      </header>
      <main>
        <DocumentList />
        <QuestionBox />
        <HistoryList />
      </main>
    </div>
  );
}
