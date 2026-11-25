import { useEffect, useRef, useState } from "react";
import { ask } from "../api.js";

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const bottom = useRef(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const submit = async (e) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    setQuestion("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setBusy(true);
    try {
      const { answer, sources } = await ask(q);
      setMessages((m) => [...m, { role: "assistant", text: answer, sources }]);
    } catch (err) {
      setMessages((m) => [...m, { role: "error", text: err.message }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="chat">
      <div className="messages">
        {messages.length === 0 && <p className="muted">Upload a document, then ask a question about it.</p>}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <p>{m.text}</p>
            {m.sources?.length > 0 && (
              <details>
                <summary>{m.sources.length} sources</summary>
                {m.sources.map((s, n) => (
                  <blockquote key={n}>
                    <strong>[{n + 1}] {s.document}</strong> · chunk {s.chunk_index} · score {s.score.toFixed(2)}
                    <br />
                    {s.text}
                  </blockquote>
                ))}
              </details>
            )}
          </div>
        ))}
        {busy && <div className="msg assistant muted">Thinking…</div>}
        <div ref={bottom} />
      </div>
      <form onSubmit={submit}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about your documents…"
          maxLength={1000}
          aria-label="Question"
        />
        <button disabled={busy || !question.trim()}>Ask</button>
      </form>
    </section>
  );
}
