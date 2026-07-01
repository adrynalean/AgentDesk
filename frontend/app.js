/* AgentDesk chat UI — React 18 without a build step (htm for JSX-free templates). */
import { h, render } from "https://esm.sh/preact@10.22.0";
import { useState, useRef, useEffect } from "https://esm.sh/preact@10.22.0/hooks";
import htm from "https://esm.sh/htm@3.1.1";

const html = htm.bind(h);

function Message({ m }) {
  return html`
    <div class="msg ${m.role}">
      <div class="role">${m.role === "user" ? "You" : "AgentDesk"}</div>
      <div>${m.text}</div>
      ${m.meta && html`
        <div class="meta">
          ${m.meta.plan?.length > 1 && html`
            <div><b>Plan:</b> ${m.meta.plan.map(p => html`<span class="chip">${p}</span>`)}</div>`}
          ${m.meta.tool_calls?.length > 0 && html`
            <div><b>Tools:</b> ${m.meta.tool_calls.map(t =>
              html`<span class="chip tool">${t.action}(${t.action_input}) → ${t.observation}</span>`)}
            </div>`}
          ${m.meta.sources?.length > 0 && html`
            <div><b>Sources:</b> ${[...new Set(m.meta.sources)].map(s =>
              html`<span class="chip">${s}</span>`)}</div>`}
        </div>`}
    </div>`;
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);

  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [messages]);

  async function send(e) {
    e.preventDefault();
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setBusy(true);
    setMessages(ms => [...ms, { role: "user", text: q }]);
    try {
      const r = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const data = await r.json();
      setMessages(ms => [...ms, { role: "agent", text: data.answer, meta: data }]);
    } catch (err) {
      setMessages(ms => [...ms, { role: "agent", text: `error: ${err}` }]);
    } finally {
      setBusy(false);
    }
  }

  return html`
    <h1>Agent<span>Desk</span></h1>
    <p class="sub">multi-agent RAG · planner → retriever → executor · tool-calling</p>
    ${messages.map(m => html`<${Message} m=${m} />`)}
    ${busy && html`<div class="msg thinking">planning → retrieving → executing…</div>`}
    <div ref=${endRef}></div>
    <form onSubmit=${send}>
      <div class="row">
        <input value=${input} onInput=${e => setInput(e.target.value)}
               placeholder="Ask about the corpus, or try: what is 42 * (3 + 1)?" />
        <button disabled=${busy}>Send</button>
      </div>
    </form>`;
}

render(html`<${App} />`, document.getElementById("root"));
