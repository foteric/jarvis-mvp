import json
import os
import uuid
from typing import Dict, List, Optional

import asyncpg
import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

GROQ_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GROQ_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
GROQ_URL = "https://openrouter.ai/api/v1/chat/completions"
# A shared passphrase that gates the /api/chat endpoint. Without this, anyone
# who finds the public Render URL could use your Gemini quota. If this is left
# unset, auth is disabled and /health reports that clearly.
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

# Neon Postgres connection string. If unset, chat history falls back to an
# in-memory dict (same behavior as before) so nothing breaks -- it just won't
# survive a restart until this is configured.
DATABASE_URL = os.environ.get("DATABASE_URL", "")

SYSTEM_PROMPT = (
    "You are J.A.R.V.I.S., a dry-witted, loyal, and highly capable personal "
    "assistant. Keep replies concise, clear, and genuinely useful. Speak with "
    "quiet confidence, address the user directly, and avoid unnecessary filler."
)

MAX_HISTORY_MESSAGES = 20

app = FastAPI(title="J.A.R.V.I.S. MVP")

# Fallback store, used only if DATABASE_URL isn't configured.
_memory_sessions: Dict[str, List[dict]] = {}
db_pool: Optional[asyncpg.pool.Pool] = None


@app.on_event("startup")
async def on_startup():
    global db_pool
    if DATABASE_URL:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    history JSONB NOT NULL DEFAULT '[]'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )


@app.on_event("shutdown")
async def on_shutdown():
    if db_pool:
        await db_pool.close()


async def get_history(session_id: str) -> List[dict]:
    if db_pool:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT history FROM sessions WHERE session_id = $1", session_id)
            if row and row["history"]:
                value = row["history"]
                return json.loads(value) if isinstance(value, str) else value
            return []
    return _memory_sessions.get(session_id, [])


async def save_history(session_id: str, history: List[dict]) -> None:
    if db_pool:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (session_id, history, updated_at)
                VALUES ($1, $2::jsonb, now())
                ON CONFLICT (session_id)
                DO UPDATE SET history = $2::jsonb, updated_at = now()
                """,
                session_id,
                json.dumps(history),
            )
    else:
        _memory_sessions[session_id] = history


async def count_sessions() -> int:
    if db_pool:
        async with db_pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM sessions")
    return len(_memory_sessions)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": GROQ_MODEL,
        "llm_configured": bool(GROQ_API_KEY),
        "auth_enabled": bool(APP_PASSWORD),
        "db_configured": bool(DATABASE_URL),
        "active_sessions": await count_sessions(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, x_app_password: Optional[str] = Header(None, alias="X-App-Password")):
    if APP_PASSWORD and x_app_password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid or missing passphrase.")

    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set on the server. Add it in your Render environment variables.",
        )

    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty.")

    session_id = req.session_id or str(uuid.uuid4())
    history = await get_history(session_id)

    history.append({"role": "user", "content": req.message})
    history = history[-MAX_HISTORY_MESSAGES:]

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
    history = history[-MAX_HISTORY_MESSAGES:]
    await save_history(session_id, history)

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
  #speakToggle {
    background: none; border: 1px solid var(--border); border-radius: 4px;
    color: var(--text-dim); font-size: 14px; line-height: 1; padding: 6px 8px;
    margin-left: 8px; cursor: pointer; flex-shrink: 0;
  }
  #speakToggle.active { color: var(--accent); border-color: var(--accent-dim); }
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
  #mic {
    background: var(--panel); border: 1px solid var(--border); border-radius: 4px;
    color: var(--accent); font-size: 18px; width: 44px; flex-shrink: 0; cursor: pointer;
  }
  #mic.listening {
    background: var(--danger); border-color: var(--danger); color: #fff;
    animation: pulse 1s infinite;
  }
  #mic:disabled { opacity: 0.3; cursor: default; }
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

  .auth-overlay {
    position: fixed; inset: 0; background: rgba(10, 14, 20, 0.96);
    display: none; align-items: center; justify-content: center; z-index: 100; padding: 20px;
  }
  .auth-box {
    background: var(--panel); border: 1px solid var(--border); border-left: 2px solid var(--accent);
    border-radius: 6px; padding: 24px 20px; width: 100%; max-width: 320px; text-align: center;
  }
  .auth-box h2 { margin: 0 0 4px; font-size: 16px; letter-spacing: 0.06em; }
  .auth-box p { margin: 0 0 16px; color: var(--text-dim); font-size: 13px; }
  #authInput {
    width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
    color: var(--text); padding: 10px 12px; font-size: 15px; margin-bottom: 12px;
    text-align: center; letter-spacing: 0.15em;
  }
  #authInput:focus { outline: none; border-color: var(--accent); }
  #authSubmit {
    width: 100%; background: var(--accent); color: #06131c; border: none; border-radius: 4px;
    padding: 10px; font-weight: 600; font-size: 14px; letter-spacing: 0.04em; cursor: pointer;
  }
  .auth-error { color: var(--danger); font-size: 12px; margin-top: 10px; min-height: 14px; }
</style>
</head>
<body>

<div class="auth-overlay" id="authOverlay">
  <div class="auth-box">
    <h2>J.A.R.V.I.S.</h2>
    <p>Enter your passphrase to continue</p>
    <input id="authInput" type="password" placeholder="Passphrase" autocomplete="off" />
    <button id="authSubmit">UNLOCK</button>
    <div class="auth-error" id="authError"></div>
  </div>
</div>

<header>
  <div class="dot pulse" id="statusDot"></div>
  <h1>J.A.R.V.I.S.</h1>
  <div class="status" id="statusText">CONNECTING…</div>
  <button id="speakToggle" title="Toggle voice output">🔇</button>
</header>

<main id="messages">
  <div class="msg system">SESSION INITIALIZED</div>
</main>

<footer>
  <button id="mic" title="Voice input">🎤</button>
  <textarea id="input" rows="1" placeholder="Say something…" autofocus></textarea>
  <button id="send">SEND</button>
</footer>

<script>
  const messagesEl = document.getElementById('messages');
  const inputEl = document.getElementById('input');
  const sendEl = document.getElementById('send');
  const micEl = document.getElementById('mic');
  const speakToggleEl = document.getElementById('speakToggle');
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');
  const authOverlay = document.getElementById('authOverlay');
  const authInput = document.getElementById('authInput');
  const authSubmit = document.getElementById('authSubmit');
  const authError = document.getElementById('authError');

  const SESSION_KEY = 'jarvis_session_id';
  const AUTH_KEY = 'jarvis_app_password';
  let sessionId = localStorage.getItem(SESSION_KEY) || null;
  let appPassword = localStorage.getItem(AUTH_KEY) || '';

  function showAuthOverlay(message) {
    authOverlay.style.display = 'flex';
    authError.textContent = message || '';
    authInput.value = '';
    authInput.focus();
  }
  function hideAuthOverlay() {
    authOverlay.style.display = 'none';
  }

  authSubmit.addEventListener('click', () => {
    const val = authInput.value.trim();
    if (!val) return;
    appPassword = val;
    localStorage.setItem(AUTH_KEY, appPassword);
    hideAuthOverlay();
  });
  authInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') authSubmit.click();
  });

  // --- Voice output (text-to-speech) ---
  let voiceEnabled = localStorage.getItem('jarvis_voice_enabled') === 'true';

  function updateSpeakToggleUI() {
    speakToggleEl.textContent = voiceEnabled ? '🔊' : '🔇';
    speakToggleEl.classList.toggle('active', voiceEnabled);
  }
  updateSpeakToggleUI();

  speakToggleEl.addEventListener('click', () => {
    voiceEnabled = !voiceEnabled;
    localStorage.setItem('jarvis_voice_enabled', voiceEnabled);
    updateSpeakToggleUI();
    if (!voiceEnabled && window.speechSynthesis) window.speechSynthesis.cancel();
  });

  function speak(text) {
    if (!voiceEnabled || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  }

  // --- Voice input (speech-to-text) ---
  const SpeechRecognitionImpl = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null;

  if (SpeechRecognitionImpl) {
    recognition = new SpeechRecognitionImpl();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => micEl.classList.add('listening');
    recognition.onend = () => micEl.classList.remove('listening');
    recognition.onerror = () => micEl.classList.remove('listening');
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      inputEl.value = transcript;
      sendMessage();
    };

    micEl.addEventListener('click', () => {
      if (window.speechSynthesis) window.speechSynthesis.cancel();
      try { recognition.start(); } catch (e) { /* already listening */ }
    });
  } else {
    micEl.disabled = true;
    micEl.title = 'Voice input not supported in this browser';
  }

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
      if (data.auth_enabled && !appPassword) {
        showAuthOverlay();
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
        headers: {
          'Content-Type': 'application/json',
          'X-App-Password': appPassword,
        },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });

      if (res.status === 401) {
        thinkingEl.remove();
        localStorage.removeItem(AUTH_KEY);
        appPassword = '';
        inputEl.value = text;
        showAuthOverlay('Incorrect passphrase. Try again.');
        return;
      }

      const data = await res.json();
      thinkingEl.remove();

      if (!res.ok) {
        addMessage('system', 'ERROR: ' + (data.detail || res.statusText));
        return;
      }

      sessionId = data.session_id;
      localStorage.setItem(SESSION_KEY, sessionId);
      addMessage('assistant', data.reply);
      speak(data.reply);
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


@app.get("/")
async def index():
    return HTMLResponse(content=HTML_PAGE, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})
