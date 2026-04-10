from fastapi import FastAPI, HTTPException
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