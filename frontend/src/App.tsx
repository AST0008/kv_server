import { useState } from "react";

import "./App.css";
import { flushSync } from "react-dom";

function App() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();

    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || isStreaming) return;

    try {
      setError(null);
      setResponse("");
      flushSync(() => setIsStreaming(true));

      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: trimmedQuestion }),
      });

      if (!res.ok || !res.body) {
        throw new Error(`Request failed with status ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by double newlines
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const event of events) {
          const dataLines = event
            .split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trimStart());

          if (dataLines.length === 0) continue;

          let data = dataLines.join("\n");

          if (!data) continue;
          console.log("Received chunk:", JSON.stringify(data));
          if (data === "[DONE]") {
            flushSync(() => setIsStreaming(false));
            return;
          }


          await new Promise((resolve) => setTimeout(resolve, 10));
          // Force immediate render for each token
          flushSync(() => {
            setResponse((prev) => prev + data);
          });
        }
      }
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Failed to fetch response",
      );
    } finally {
      flushSync(() => setIsStreaming(false));
    }
  }

  return (
    <main className="app-shell">
      <section className="chat-card">
        <h1 className="title">KV Chat</h1>
        <p className="subtitle">
          Ask a question and watch the answer stream in real time.
        </p>

        <form className="prompt-form" onSubmit={handleSubmit}>
          <input
            className="prompt-input"
            type="text"
            name="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask anything..."
            disabled={isStreaming}
          />
          <button
            className="submit-button"
            type="submit"
            disabled={isStreaming || question.trim().length === 0}
          >
            {isStreaming ? "Streaming..." : "Send"}
          </button>
        </form>

        {error ? <p className="error-text">{error}</p> : null}

        <div id="response" className="response-box" aria-live="polite">
          {response || (isStreaming ? "" : "Response will appear here...")}
          {isStreaming ? <span className="cursor">▍</span> : null}
        </div>
      </section>
    </main>
  );
}

export default App;
