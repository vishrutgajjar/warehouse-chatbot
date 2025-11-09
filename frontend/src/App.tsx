import React, { useState } from "react";

export default function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [thinking, setThinking] = useState(false);

  const uploadFile = async () => {
  if (!file) return alert("Select a PDF first!");
  setUploading(true);

  try {
    const form = new FormData();
    form.append("file", file);

    const res = await fetch("http://localhost:8000/upload", {
      method: "POST",
      body: form,
    });

    // Try all possible return formats
    let data;
    try {
      data = await res.json();
    } catch {
      data = { message: await res.text() };
    }

    if (res.ok) {
      alert(data.message || "✅ PDF uploaded successfully!");
    } else {
      alert("❌ Upload failed: " + (data.message || res.statusText));
    }
  } catch (err) {
    console.error("UPLOAD ERROR:", err);
    alert("❌ Error uploading file. Check console for details.");
  } finally {
    setUploading(false);
  }
};



  const askQuestion = async () => {
    if (!query.trim()) return;
    setThinking(true);
    setAnswer("");
    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      setAnswer(data.answer || "No answer found.");
    } catch (err) {
      console.error(err);
      setAnswer("❌ Error contacting backend.");
    } finally {
      setThinking(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 20, fontFamily: "sans-serif" }}>
      <h1>🏭 Warehouse Chatbot</h1>

      <div>
        <input type="file" accept="application/pdf" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <button onClick={uploadFile} disabled={uploading} style={{ marginLeft: 10 }}>
          {uploading ? "Uploading..." : "Upload SOP PDF"}
        </button>
      </div>

      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask a question about your SOP..."
        style={{ width: 400, height: 100 }}
      />

      <button onClick={askQuestion} disabled={thinking || !query.trim()}>
        {thinking ? "🤔 Thinking..." : "Ask"}
      </button>

      <h3>Answer:</h3>
      <p style={{ whiteSpace: "pre-wrap" }}>{answer}</p>
    </div>
  );
}
