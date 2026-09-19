import os
import uuid
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

GROQ_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

SYSTEM_PROMPT = (
    "You are J.A.R.V.I.S., a dry-witted, loyal, and highly capable personal "
    "assistant. Keep replies concise, clear, and genuinely useful. Speak with "
    "quiet confidence, address the user directly, and avoid unnecessary filler."
)

MAX_HISTORY_MESSAGES = 20

app = FastAPI(title="J.A.R.V.I.S. MVP")

sessions: Dict[str, List[dict]] = {}


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": GROQ_MODEL,
        "llm_configured": bool(GROQ_API_KEY),
        "active_sessions": len(sessions),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set on the server. Add it in your Render environment variables.",
        )

    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty.")

    session_id = req.session_id or str(uuid.uuid4())
    history = sessions.setdefault(session_id, [])

    history.append({"role": "user", "content": req.message})
    del history[:-MAX_HISTORY_MESSAGES]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={"model": GROQ_MODEL, "messages": messages},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502, detail=f"LLM provider error ({e.response.status_code}): {e.response.text[:300]}"
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Could not reach LLM provider: {e}")

    data = resp.json()
    reply = data["choices"][0]["message"]["content"]

    history.append({"role": "assistant", "content": reply})
    del history[:-MAX_HISTORY_MESSAGES]

    return ChatResponse(session_id=session_id, reply=reply)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<title>J.A.R.V.I.S.</title>
<style>
  :root {
    --bg: #0a0e14;
    --panel: #10161f;
    --border: #1c2733;
    --accent: #4cc9f0;
    --accent-dim: #2b5a6e;
    --warn: #ffb454;
    --text: #dce6ef;
    --text-dim: #6f8399;
    --danger: #ff5c5c;
    color-scheme: dark;
  }
  * { box-sizing: border-box; }
  html, body {
    height: 100%; margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  body {
    display: flex; flex-direction: column;
    background-image:
      radial-gradient(circle at 15% 0%, rgba(76, 201, 240, 0.08), transparent 40%),
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: auto, 28px 28px, 28px 28px;
    background-position: 0 0, -1px -1px, -1px -1px;
  }
  header {
    display: flex; align-items: center; gap: 10px;
    padding: calc(env(safe-area-inset-top, 0px) + 14px) 16px 14px;
    border-bottom: 1px solid var(--border);
    background: rgba(10, 14, 20, 0.85); backdrop-filter: blur(6px);
    position: sticky; top: 0; z-index: 10;
  }
  .dot {
    width: 9px; height: 9px; border-radius: 50%; background: var(--accent);
    box-shadow: 0 0 8px 2px rgba(76, 201, 240, 0.7); flex-shrink: 0;
  }
  .dot.offline { background: var(--danger); box-shadow: 0 0 8px 2px rgba(255, 92, 92, 0.7); }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
  .dot.pulse { animation: pulse 1.6s ease-in-out infinite; }
  header h1 { font-size: 15px; letter-spacing: 0.06em; margin: 0; font-weight: 600; }
  header .status {
    margin-left: auto; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 11px; color: var(--text-dim); text-align: right;
  }
  main {
    flex: 1; overflow-y: auto; padding: 18px 14px 10px;
    display: flex; flex-direction: column; gap: 12px;
  }
  .msg {
    max-width: 82%; padding: 10px 13px; border-radius: 4px; line-height: 1.45;
    font-size: 15px; white-space: pre-wrap; word-wrap: break-word;
  }
  .msg.assistant {
    align-self: flex-start; background: var(--panel); border: 1px solid var(--border);
    border-left: 2px solid var(--accent);
  }
  .msg.user {
    align-self: flex-end; background: rgba(76, 201, 240, 0.1);
    border: 1px solid var(--accent-dim); border-right: 2px solid var(--warn);
  }
  .msg.system {
    align-self: center; color: var(--text-dim);
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px; background: none; border: none; padding: 4px;
  }
  .msg .label {
    display: block; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 10px; letter-spacing: 0.08em; color: var(--text-dim); margin-bottom: 4px;
  }
  .thinking span {
    display: inline-block; width: 5px; height: 5px; margin-right: 3px;
    border-radius: 50%; background: var(--accent); animation: pulse 1s infinite;
  }
  .thinking span:nth-child(2) { animation-delay: 0.15s; }
  .thinking span:nth-child(3) { animation-delay: 0.3s; }
  footer {
    display: flex; gap: 8px; padding: 12px 14px calc(env(safe-area-inset-bottom, 0px) + 12px);
    border-top: 1px solid var(--border); background: rgba(10, 14, 20, 0.85); backdrop-filter: blur(6px);
  }
  #input {
    flex: 1; resize: none; background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; color: var(--text); padding: 11px 12px; font-size: 15px;
    font-family: inherit; max-height: 120px;
  }
  #input:focus { outline: none; border-color: var(--accent); }
  #send {
    background: var(--accent); color: #06131c; border: none; border-radius: 4px;
    padding: 0 18px; font-weight: 600; font-size: 14px; letter-spacing: 0.02em; cursor: pointer; flex-shrink: 0;
  }
  #send:disabled { opacity: 0.4; cursor: default; }
  #send:active:not(:disabled) { transform: translateY(1px); }
</style>
</head>
<body>

<header>
  <div class="dot pulse" id="statusDot"></div>
  <h1>J.A.R.V.I.S.</h1>
  <div class="status" id="statusText">CONNECTING…</div>
</header>

<main id="messages">
  <div class="msg system">SESSION INITIALIZED</div>
</main>

<footer>
  <textarea id="input" rows="1" placeholder="Say something…" autofocus></textarea>
  <button id="send">SEND</button>
</footer>

<script>
  const messagesEl = document.getElementById('messages');
  const inputEl = document.getElementById('input');
  const sendEl = document.getElementById('send');
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');

  const SESSION_KEY = 'jarvis_session_id';
  let sessionId = localStorage.getItem(SESSION_KEY) || null;

  function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    if (role !== 'system') {
      const label = document.createElement('span');
      label.className = 'label';
      label.textContent = role === 'user' ? 'YOU' : 'JARVIS';
      div.appendChild(label);
    }
    const body = document.createElement('div');
    body.textContent = text;
    div.appendChild(body);
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  function addThinking() {
    const div = document.createElement('div');
    div.className = 'msg assistant thinking';
    div.innerHTML = '<span class="label">JARVIS</span><span></span><span></span><span></span>';
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  async function checkHealth() {
    try {
      const res = await fetch('/health');
      const data = await res.json();
      if (data.llm_configured) {
        statusDot.classList.remove('offline');
        statusText.textContent = data.model.toUpperCase();
      } else {
        statusDot.classList.add('offline');
        statusText.textContent = 'NO API KEY SET';
      }
    } catch (e) {
      statusDot.classList.add('offline');
      statusText.textContent = 'UNREACHABLE';
    }
  }

  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;

    inputEl.value = '';
    inputEl.style.height = 'auto';
    sendEl.disabled = true;
    addMessage('user', text);
    const thinkingEl = addThinking();

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });

      const data = await res.json();
      thinkingEl.remove();

      if (!res.ok) {
        addMessage('system', 'ERROR: ' + (data.detail || res.statusText));
        return;
      }

      sessionId = data.session_id;
      localStorage.setItem(SESSION_KEY, sessionId);
      addMessage('assistant', data.reply);
    } catch (e) {
      thinkingEl.remove();
      addMessage('system', 'CONNECTION FAILED: ' + e.message);
    } finally {
      sendEl.disabled = false;
      inputEl.focus();
    }
  }

  sendEl.addEventListener('click', sendMessage);
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = inputEl.scrollHeight + 'px';
  });

  checkHealth();
</script>

</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE
