import asyncio
import json
import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from thenvoi import Agent
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.types import PlatformMessage

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SupervisorDashboard")

# ---------- WebSocket connection manager ----------
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

# ---------- Band Adapter (pure relay – no LLM) ----------
class RelayAdapter(SimpleAdapter):
    def __init__(self):
        super().__init__(history_converter=None)
        self.room_tools: dict[str, AgentToolsProtocol] = {}
        self.room_id: str | None = None
        self.agent_name: str | None = None

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        self.agent_name = agent_name
        logger.info(f"✅ Supervisor connected as {agent_name}")

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history,
        participants_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
        contacts_msg: str | None = None,
    ) -> None:
        self.room_tools[room_id] = tools
        self.room_id = room_id

        if self.agent_name and msg.sender_name == self.agent_name:
            return

        await manager.broadcast(json.dumps({
            "type": "chat",
            "sender": msg.sender_name,
            "content": msg.content,
            "timestamp": msg.created_at.isoformat() if hasattr(msg, 'created_at') else None,
        }))
        logger.info(f"📨 Relayed from {msg.sender_name}: {msg.content[:80]}...")

    async def on_cleanup(self, room_id: str) -> None:
        self.room_tools.pop(room_id, None)
        if self.room_id == room_id:
            self.room_id = None

# ---------- FastAPI app ----------
app = FastAPI(title="Band Supervisor Dashboard")
adapter = RelayAdapter()
agent: Agent | None = None

@app.on_event("startup")
async def startup():
    global agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=os.getenv("SUPERVISOR_AGENT_ID"),
        api_key=os.getenv("SUPERVISOR_API_KEY"),
        ws_url=os.getenv("THENVOI_WS_URL"),
        rest_url=os.getenv("THENVOI_REST_URL"),
    )
    asyncio.create_task(agent.run())
    logger.info("🚀 Supervisor agent started, waiting for room messages...")

@app.on_event("shutdown")
async def shutdown():
    if agent:
        await agent.stop()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                if payload.get("type") == "document":
                    content = payload.get("content", "").strip()
                    if content and adapter.room_id and adapter.room_tools.get(adapter.room_id):
                        tools = adapter.room_tools[adapter.room_id]
                        try:
                            # Only mention the Document Triager (Agent 1)
                            target_agent = "@hexo/document-triager"
                            await tools.send_message(content, mentions=[target_agent])
                            logger.info(f"📝 Posted document to {target_agent}: {content[:80]}...")
                        except Exception as e:
                            logger.error(f"Failed to send message: {e}")
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "content": f"Failed to post: {str(e)}"
                            }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "content": "Agent not ready or no room active"
                        }))
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "content": "Invalid JSON"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/")
async def get_chat_ui():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Band War Room – Document Supervisor</title>
        <style>
            * {
                box-sizing: border-box;
            }
            body {
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                background: #0a0c15;
                margin: 0;
                padding: 20px;
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .chat-container {
                width: 1000px;
                max-width: 100%;
                height: 90vh;
                background: #141824;
                border-radius: 24px;
                box-shadow: 0 20px 35px rgba(0,0,0,0.5);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                border: 1px solid #2a2f42;
            }
            .chat-header {
                background: #1e243b;
                padding: 18px 24px;
                color: white;
                font-weight: 600;
                font-size: 1.25rem;
                border-bottom: 1px solid #2a2f42;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .chat-header span:first-child {
                font-size: 1.5rem;
            }
            .messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 12px;
                background: #0f111a;
            }
            .message {
                max-width: 80%;
                padding: 10px 16px;
                border-radius: 20px;
                line-height: 1.45;
                word-wrap: break-word;
                font-size: 0.9rem;
                animation: fadeIn 0.2s ease;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(5px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .message-user {
                background: #1e88e5;
                color: white;
                align-self: flex-start;
                border-bottom-left-radius: 4px;
            }
            .message-agent {
                background: #2c6e5c;
                color: white;
                align-self: flex-end;
                border-bottom-right-radius: 4px;
            }
            .message-other {
                background: #23283e;
                color: #e0e4f0;
                align-self: flex-start;
                border-bottom-left-radius: 4px;
            }
            .sender {
                font-size: 0.7rem;
                font-weight: 600;
                margin-bottom: 4px;
                opacity: 0.8;
                letter-spacing: 0.3px;
            }
            .input-area {
                padding: 18px;
                background: #141824;
                border-top: 1px solid #2a2f42;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            textarea {
                background: #0a0c15;
                border: 1px solid #2a2f42;
                color: #f0f3ff;
                padding: 14px;
                border-radius: 20px;
                font-family: inherit;
                font-size: 0.9rem;
                resize: vertical;
                outline: none;
                transition: 0.2s;
            }
            textarea:focus {
                border-color: #1e88e5;
                box-shadow: 0 0 0 2px rgba(30,136,229,0.2);
            }
            button {
                background: #1e88e5;
                color: white;
                border: none;
                padding: 10px 22px;
                border-radius: 40px;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s;
                width: 140px;
                align-self: flex-end;
                font-size: 0.9rem;
            }
            button:hover {
                background: #0b5e7e;
            }
            .status {
                padding: 8px 18px;
                font-size: 0.75rem;
                color: #8e98b3;
                background: #0a0c15;
                text-align: center;
                border-top: 1px solid #2a2f42;
            }
            .status.connected {
                color: #6fbf73;
            }
            .status.disconnected {
                color: #f28b82;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <span>📄</span> M&A War Room · Supervisor Dashboard
            </div>
            <div class="messages" id="messages">
                <div class="message message-other">
                    <div class="sender">⚙️ System</div>
                    Waiting for room connection...
                </div>
            </div>
            <div class="input-area">
                <textarea id="documentInput" rows="3" placeholder="Paste a document (or any message) here…&#10;Press Ctrl+Enter to send."></textarea>
                <button id="sendBtn">📤 Post to Room</button>
            </div>
            <div class="status" id="status">🔌 Connecting to supervisor agent...</div>
        </div>

        <script>
            const messagesDiv = document.getElementById('messages');
            const documentInput = document.getElementById('documentInput');
            const sendBtn = document.getElementById('sendBtn');
            const statusSpan = document.getElementById('status');
            let ws = null;

            function addMessage(sender, content, type = 'other') {
                const msgDiv = document.createElement('div');
                msgDiv.className = `message message-${type}`;
                const senderSpan = document.createElement('div');
                senderSpan.className = 'sender';
                senderSpan.textContent = sender;
                const contentSpan = document.createElement('div');
                contentSpan.textContent = content;
                msgDiv.appendChild(senderSpan);
                msgDiv.appendChild(contentSpan);
                messagesDiv.appendChild(msgDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                ws = new WebSocket(wsUrl);
                ws.onopen = () => {
                    statusSpan.innerHTML = '✅ Connected · relaying live chat';
                    statusSpan.classList.add('connected');
                    statusSpan.classList.remove('disconnected');
                };
                ws.onclose = () => {
                    statusSpan.innerHTML = '❌ Disconnected – retrying in 3s...';
                    statusSpan.classList.add('disconnected');
                    statusSpan.classList.remove('connected');
                    setTimeout(connectWebSocket, 3000);
                };
                ws.onerror = (err) => {
                    console.error('WebSocket error', err);
                };
                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.type === 'chat') {
                            const sender = data.sender;
                            const content = data.content;
                            let style = 'other';
                            if (sender.toLowerCase().includes('user')) style = 'user';
                            else if (sender.toLowerCase().includes('supervisor')) style = 'agent';
                            addMessage(sender, content, style);
                        } else if (data.type === 'error') {
                            addMessage('⚠️ System', `Error: ${data.content}`, 'other');
                        }
                    } catch(e) {
                        console.error('Parse error', e);
                    }
                };
            }

            sendBtn.onclick = () => {
                const content = documentInput.value.trim();
                if (!content) return;
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'document', content: content }));
                    documentInput.value = '';
                    addMessage('You (pending)', content, 'agent');
                } else {
                    addMessage('System', 'Not connected – please wait for WebSocket', 'other');
                }
            };

            documentInput.addEventListener('keydown', (e) => {
                if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                    e.preventDefault();
                    sendBtn.click();
                }
            });

            connectWebSocket();
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)