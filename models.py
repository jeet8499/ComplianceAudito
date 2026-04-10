from pydantic import BaseModel
from typing import List, Optional, Literal
from enum import Enum

class Regulation(str, Enum):
    GDPR = "gdpr"
    CCPA = "ccpa"

class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class Clause(BaseModel):
    index: int
    text: str

class Observation(BaseModel):
    task_id: str
    difficulty: TaskDifficulty
    regulation: Regulation
    document_title: str
    clauses: List[Clause]
    step_count: int
    max_steps: int
    remaining_clauses: List[int]

class Action(BaseModel):
    action_type: Literal["flag", "skip", "finalize"]
    clause_index: Optional[int] = None
    is_violation: Optional[bool] = None

class RewardInfo(BaseModel):
    reward: float
    done: bool
    info: dict