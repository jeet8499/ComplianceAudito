from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from environment import ComplianceAuditorEnv
from models import Action, TaskDifficulty
import uuid

app = FastAPI()
sessions = {}

# --- HTML landing page (optional) ---
@app.get("/", response_class=HTMLResponse)
def root():
    return """<html><body><h1>Compliance Auditor</h1><p>API endpoints: POST/GET /reset, POST /step, GET /state/{id}</p></body></html>"""

# --- Reset: works with GET (query param) or POST (body or empty) ---
@app.get("/reset")
async def reset_get(difficulty: str = Query("easy")):
    return await reset_logic(difficulty)

@app.post("/reset")
async def reset_post(request: Request):
    difficulty = "easy"
    try:
        body = await request.body()
        if body:
            data = await request.json()
            difficulty = data.get("difficulty", "easy")
    except:
        pass
    return await reset_logic(difficulty)

async def reset_logic(difficulty: str):
    try:
        diff = TaskDifficulty(difficulty)
    except:
        raise HTTPException(400, "Invalid difficulty")
    env = ComplianceAuditorEnv(difficulty=diff)
    obs = env.reset()
    session_id = str(uuid.uuid4())
    sessions[session_id] = env
    return {"session_id": session_id, "observation": obs.dict()}

# --- Step: expects session_id in query param, action in body ---
class StepAction(BaseModel):
    action_type: str
    clause_index: int | None = None
    is_violation: bool | None = None

@app.post("/step")
async def step(action: StepAction, session_id: str = Query(...)):
    env = sessions.get(session_id)
    if not env:
        raise HTTPException(404, "Session not found")
    # Convert StepAction to your Action model
    from models import Action as ActionModel
    act = ActionModel(action_type=action.action_type, clause_index=action.clause_index, is_violation=action.is_violation)
    obs, reward, done, info = env.step(act)
    return [obs.dict(), reward, done, info]

# --- State endpoint ---
@app.get("/state/{session_id}")
def get_state(session_id: str):
    env = sessions.get(session_id)
    if not env:
        raise HTTPException(404, "Session not found")
    return env.state()
