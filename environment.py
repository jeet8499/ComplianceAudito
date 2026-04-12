import uuid
from typing import Tuple, Dict, Any
from models import Observation, Action, TaskDifficulty, Regulation, Clause
from graders import TASK_GROUND_TRUTH, grade

class ComplianceAuditorEnv:
    def __init__(self, difficulty: TaskDifficulty = TaskDifficulty.EASY):
        self.difficulty = difficulty
        self.task_id = None
        self.regulation = None
        self.document_title = ""
        self.clauses = []
        self.max_steps = 20
        self.step_count = 0
        self.flags = {}
        self._init_task()

    def _init_task(self):
        if self.difficulty == TaskDifficulty.EASY:
            self.task_id = "easy_1"
        elif self.difficulty == TaskDifficulty.MEDIUM:
            self.task_id = "medium_1"
        else:
            self.task_id = "hard_1"
        task_data = TASK_GROUND_TRUTH[self.task_id]
        self.regulation = Regulation(task_data["regulation"])
        self.document_title = f"Sample {self.regulation.value.upper()} Policy"
        self.clauses = [Clause(index=i, text=txt) for i, txt in enumerate(task_data["clause_texts"])]

    def reset(self) -> Observation:
        self._init_task()
        self.step_count = 0
        self.flags = {}
        return self._get_obs()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = -0.01
        done = False
        info = {}

        if action.action_type == "flag":
            if action.clause_index is None or action.is_violation is None:
                reward -= 0.1
                info["error"] = "flag requires clause_index and is_violation"
            else:
                idx = action.clause_index
                if 0 <= idx < len(self.clauses):
                    if idx in self.flags:
                        reward -= 0.05
                    else:
                        self.flags[idx] = action.is_violation
                        truth = TASK_GROUND_TRUTH[self.task_id]["violations"]
                        expected = truth.get(idx, False)
                        expected_bool = expected > 0.5 if isinstance(expected, float) else expected
                        if action.is_violation == expected_bool:
                            reward += 0.3
                        else:
                            reward -= 0.1
                else:
                    reward -= 0.2
                    info["error"] = "invalid clause index"
        elif action.action_type == "skip":
            pass
        elif action.action_type == "finalize":
            done = True
            final_score = grade(self.task_id, self.flags)
            info["final_score"] = final_score
            reward += final_score
        else:
            reward -= 0.2
            info["error"] = "unknown action_type"

        if self.step_count >= self.max_steps and not done:
            done = True
            info["final_score"] = grade(self.task_id, self.flags)
            reward += info["final_score"]
            info["message"] = "Max steps reached"

        obs = self._get_obs()
        return obs, reward, done, info

    def _get_obs(self) -> Observation:
        reviewed = list(self.flags.keys())
        remaining = [c.index for c in self.clauses if c.index not in reviewed]
        return Observation(
            task_id=self.task_id,
            difficulty=self.difficulty,
            regulation=self.regulation,
            document_title=self.document_title,
            clauses=self.clauses,
            step_count=self.step_count,
            max_steps=self.max_steps,
            remaining_clauses=remaining
        )

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty.value,
            "regulation": self.regulation.value,
            "flags": self.flags,
            "step_count": self.step_count
        }
