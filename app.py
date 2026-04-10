from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from environment import ComplianceAuditorEnv
from models import Action, TaskDifficulty, Observation
import uuid

app = FastAPI()
sessions = {}

class ResetRequest(BaseModel):
    difficulty: str = "easy"

class StepRequest(BaseModel):
    session_id: str
    action: Action

# --- NEW HTML landing page ---
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Compliance Auditor</title><meta charset="UTF-8"></head>
    <body style="font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 1rem;">
        <h1>🛡️ Compliance Auditor Environment</h1>
        <p>OpenEnv environment for GDPR/CCPA compliance auditing.</p>
        <h2>Endpoints</h2>
        <ul><li><code>POST /reset</code></li><li><code>POST /step</code></li><li><code>GET /state/{session_id}</code></li></ul>
        <h2>Test with PowerShell</h2>
        <pre style="background:#f4f4f4;padding:1rem;">$body = '{"difficulty":"easy"}'
Invoke-RestMethod -Uri "https://xcoder18-project1.hf.space/reset" -Method Post -ContentType "application/json" -Body $body</pre>
        <p><a href="https://huggingface.co/spaces/xcoder18/project1/blob/main/README.md">Full documentation</a></p>
    </body>
    </html>
    """
# --- Existing endpoints ---
@app.post("/reset")
def reset(req: ResetRequest):
    try:
        diff = TaskDifficulty(req.difficulty)
    except:
        raise HTTPException(400, "Invalid difficulty")
    env = ComplianceAuditorEnv(difficulty=diff)
    obs = env.reset()
    session_id = str(uuid.uuid4())
    sessions[session_id] = env
    return {"session_id": session_id, "observation": obs.dict()}

@app.post("/step")
def step(req: StepRequest):
    env = sessions.get(req.session_id)
    if not env:
        raise HTTPException(404, "Session not found")
    obs, reward, done, info = env.step(req.action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state/{session_id}")
def get_state(session_id: str):
    env = sessions.get(session_id)
    if not env:
        raise HTTPException(404, "Session not found")
    return env.state()
